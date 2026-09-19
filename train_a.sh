#!/usr/bin/env bash
# train_a.sh —— 批量执行 CONFIGS/tasks/afinetune 下的 33 个音频下游任务。
#
# 用法：
#   bash train_a.sh --dry-run  # 校验并打印 33 条命令
#   bash train_a.sh            # 串行执行
#   bash train_a.sh 2          # 两个并发槽位；同一槽位内串行
#
# 注意：每个任务自身会用 nproc_per_node 张 GPU。并发数大于 1 时，多个任务会
# 共享同一组可见 GPU；只有显存足够时才这样使用。

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
用法：
  bash train_a.sh --dry-run  只校验配置并打印生成的 33 条命令
  bash train_a.sh [并发数]  执行全部任务；并发数默认为 1

切换为新的 AEmo-JEPA 预训练 checkpoint 时，修改脚本顶部 EXPERIMENT_META：
  checkpoint=OUTPUT/pretrain_a/aemojepa-plus-large/FaVoR-2to4s-full-e2v/latest.pt
  run_name=aemojepa-plus-large-full-e2v
  feature_mode=topk_average
EOF
}

die() {
  printf 'train_a.sh: %s\n' "$*" >&2
  exit 2
}

# ==================== 实验元数据（通常只改这里）====================
# 当前默认跑官方 emotion2vec+ large 基线。
# 新版 pretrain_a 输出同样使用完整 emotion2vec 结构，因此只需换 checkpoint；
# 对 AEmo-JEPA checkpoint，建议同时把 feature_mode 改成 topk_average。
declare -A EXPERIMENT_META=(
  [nproc_per_node]="2"
  [master_port_base]="12400"
  [checkpoint]="CKPT/emotion2vec_plus_large/model.pt"
  [output_root]="OUTPUT/finetune_a"
  [run_name]="emotion2vec-plus-large"
  [backbone_mode]="emotion2vec"
  [feature_mode]="last"
)

# 所有任务共享的附加覆盖。每项必须是完整的 key=value。
# 示例：EXTRA_SET=("meta.seed=7" "optimization.epochs=30")
EXTRA_SET=()

# ==================== 33 个音频下游任务 ====================
# 格式：任务分组|任务名|可选逐任务覆盖（多个覆盖用分号分隔）
TASK_META=(
  "cls|ASVP-ESD-emotion|"
  "cls|ASVP-ESD-intensity|"
  "cls|Androids-Corpus-label|"
  "cls|CASIA-emotion|"
  "cls|CMDC-label|"
  "cls|CREMA-D-emotion|"
  "cls|CSEMOTIONS-emotion|"
  "cls|EMNS-emotion|"
  "cls|ESD-Chinese-emotion|"
  "cls|EmotionTalk-emotion|"
  "cls|IEMOCAP-emotion|"
  "cls|MER2023-emotion|"
  "cls|MER242526-emotion|"
  "cls|MODMA-label|"
  "cls|RAVDESS-emotion|"
  "cls|RAVDESS-intensity|"
  "mlcls|CNSCED-emotion|"
  "mlcls|M3ED-emotion|"
  "mlcls|MER242526-26openset|"
  "reg|AVEC2014-PHQ|"
  "reg|CMDC-HAMD|"
  "reg|CMDC-PHQ|"
  "reg|CNSCED-intensity|"
  "reg|CREMA-D-intensity|"
  "reg|EATD-Corpus-SDS|"
  "reg|EMNS-intensity|"
  "reg|EMOVIE-pos_intensity|"
  "reg|IEMOCAP-activation|"
  "reg|IEMOCAP-dominance|"
  "reg|IEMOCAP-valence|"
  "reg|MER2023-pos_intensity|"
  "reg|MER242526-pos_intensity|"
  "reg|PDCH-HAMD|"
)

CMDS=()
declare -A SEEN_TASKS=()

validate_metadata() {
  local key checkpoint last_port config_count
  local required=(
    nproc_per_node master_port_base checkpoint output_root run_name
    backbone_mode feature_mode
  )
  for key in "${required[@]}"; do
    [[ -v "EXPERIMENT_META[$key]" ]] || die "EXPERIMENT_META 缺少字段: $key"
  done
  [[ "${EXPERIMENT_META[nproc_per_node]}" =~ ^[1-9][0-9]*$ ]] \
    || die "nproc_per_node 必须是正整数"
  [[ "${EXPERIMENT_META[master_port_base]}" =~ ^[1-9][0-9]*$ ]] \
    || die "master_port_base 必须是正整数"
  [[ "${EXPERIMENT_META[backbone_mode]}" =~ ^(emotion2vec|aemojepa)$ ]] \
    || die "backbone_mode 只能是 emotion2vec 或 aemojepa"
  [[ "${EXPERIMENT_META[feature_mode]}" =~ ^(last|topk_average)$ ]] \
    || die "feature_mode 只能是 last 或 topk_average"
  [[ -n "${EXPERIMENT_META[output_root]}" ]] || die "output_root 不能为空"
  [[ -n "${EXPERIMENT_META[run_name]}" ]] || die "run_name 不能为空"
  [[ "${EXPERIMENT_META[run_name]}" != */* ]] || die "run_name 不能包含 /"

  checkpoint=${EXPERIMENT_META[checkpoint]}
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || die "checkpoint 不存在: $checkpoint"
  ((${#TASK_META[@]} == 33)) || die "TASK_META 应为 33 项，当前为 ${#TASK_META[@]}"
  config_count=$(find CONFIGS/tasks/afinetune -type f -name '*.yaml' | wc -l)
  ((config_count == 33)) || die "CONFIGS/tasks/afinetune 应有 33 个 YAML，当前为 $config_count"
  last_port=$((EXPERIMENT_META[master_port_base] + ${#TASK_META[@]} - 1))
  ((last_port <= 65535)) || die "生成的 master_port 超出 65535: $last_port"

  for key in "${EXTRA_SET[@]}"; do
    [[ "$key" == *=* ]] || die "EXTRA_SET 项不是 key=value: $key"
  done
}

render_command() {
  local index=$1 row=$2
  local group task task_overrides config folder port command override
  local -a argv task_set=()

  IFS='|' read -r group task task_overrides <<<"$row"
  case "$group" in
    cls|mlcls|reg) ;;
    *) die "TASK_META 第 $((index + 1)) 行分组无效: $group" ;;
  esac
  [[ "$task" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "TASK_META 第 $((index + 1)) 行任务名无效: $task"
  [[ -z "${SEEN_TASKS[$task]+x}" ]] || die "TASK_META 任务重复: $task"
  SEEN_TASKS[$task]=1

  config="CONFIGS/tasks/afinetune/$group/$task.yaml"
  [[ -f "$config" ]] || die "任务配置不存在: $config"
  folder="${EXPERIMENT_META[output_root]}/$task/${EXPERIMENT_META[run_name]}"
  port=$((EXPERIMENT_META[master_port_base] + index))

  argv=(
    torchrun
    --master_port "$port"
    --nproc_per_node "${EXPERIMENT_META[nproc_per_node]}"
    -m app.main
    --fname "$config"
    --set
    "meta.read_checkpoint=${EXPERIMENT_META[checkpoint]}"
    "folder=$folder"
    "model.backbone.mode=${EXPERIMENT_META[backbone_mode]}"
    "model.features.mode=${EXPERIMENT_META[feature_mode]}"
  )
  argv+=("${EXTRA_SET[@]}")

  if [[ -n "$task_overrides" ]]; then
    IFS=';' read -r -a task_set <<<"$task_overrides"
    for override in "${task_set[@]}"; do
      [[ "$override" == *=* ]] \
        || die "任务 $task 的逐任务覆盖不是 key=value: $override"
    done
    argv+=("${task_set[@]}")
  fi

  printf -v command '%q ' "${argv[@]}"
  CMDS+=("${command% }")
}

validate_metadata
for index in "${!TASK_META[@]}"; do
  render_command "$index" "${TASK_META[$index]}"
done

TOTAL=${#CMDS[@]}
((TOTAL == 33)) || die "应生成 33 条命令，实际生成 $TOTAL 条"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
(( $# <= 1 )) || { usage >&2; exit 2; }

N=${1:-1}
[[ "$N" =~ ^[1-9][0-9]*$ ]] || die "并发数必须是正整数，当前为: $N"
((N > TOTAL)) && N=$TOTAL

if [[ "$DRY_RUN" == true ]]; then
  printf '音频下游任务: %d\n' "$TOTAL"
  printf 'checkpoint: %s\n' "${EXPERIMENT_META[checkpoint]}"
  printf 'run_name: %s\n' "${EXPERIMENT_META[run_name]}"
  printf 'backbone: %s | features: %s\n\n' \
    "${EXPERIMENT_META[backbone_mode]}" "${EXPERIMENT_META[feature_mode]}"
  for index in "${!CMDS[@]}"; do
    printf '[%02d] %s\n' "$((index + 1))" "${CMDS[$index]}"
  done
  exit 0
fi

STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="OUTPUT/train_a_sh_logs/$STAMP"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
: > "$QUEUE_LOG"

qlog() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"; }

job_name() {
  local fname base
  fname=$(sed -n 's/.*--fname[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p' <<<"$1")
  base=${fname##*/}
  printf '%s' "${base%.yaml}"
}

run_one() {
  local slot=$1 idx=$2 cmd=$3
  local name log rc start
  name=$(job_name "$cmd")
  log="$LOGDIR/$(printf '%02d' "$idx")-${name}.log"
  start=$SECONDS
  qlog "[槽$slot] 开始 $idx/$TOTAL $name"
  printf '# %s\n# 命令: %s\n\n' "$(date '+%F %T')" "$cmd" > "$log"
  bash -c "$cmd" >>"$log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$LOGDIR/.rc-$idx"
  if ((rc == 0)); then
    qlog "[槽$slot] 完成 $idx/$TOTAL $name ($((SECONDS - start))s)"
  else
    qlog "[槽$slot] 失败 $idx/$TOTAL $name 退出码=$rc ($((SECONDS - start))s)"
  fi
}

on_interrupt() {
  trap - INT TERM
  printf '\n'
  qlog "收到中断信号，终止全部槽位及其训练进程"
  kill 0 2>/dev/null
  exit 130
}

printf '候选音频任务 %d 个，并发槽位 %d 个\n' "$TOTAL" "$N"
((N > 1)) && printf '注意: %d 个任务共享同一组 GPU，请确认显存余量以免 OOM\n' "$N"
printf 'checkpoint: %s\nrun_name: %s\nbackbone: %s | features: %s\n日志目录: %s\n\n' \
  "${EXPERIMENT_META[checkpoint]}" "${EXPERIMENT_META[run_name]}" \
  "${EXPERIMENT_META[backbone_mode]}" "${EXPERIMENT_META[feature_mode]}" "$LOGDIR"

pids=()
for ((slot = 1; slot <= N; slot++)); do
  (
    for ((idx = slot; idx <= TOTAL; idx += N)); do
      run_one "$slot" "$idx" "${CMDS[$((idx - 1))]}"
    done
  ) &
  pids+=("$!")
done

trap on_interrupt INT TERM
for pid in "${pids[@]}"; do
  wait "$pid"
done
trap - INT TERM

printf '\n==================== 队列结果 ====================\n'
failed=0
for ((idx = 1; idx <= TOTAL; idx++)); do
  rc=$(cat "$LOGDIR/.rc-$idx" 2>/dev/null)
  name=$(job_name "${CMDS[$((idx - 1))]}")
  if [[ "$rc" == "0" ]]; then
    printf '  [OK]   %02d/%02d  %s\n' "$idx" "$TOTAL" "$name"
  else
    printf '  [FAIL] %02d/%02d  %s  (退出码=%s)\n' \
      "$idx" "$TOTAL" "$name" "${rc:-未执行}"
    failed=$((failed + 1))
  fi
done
printf '\n总耗时 %d 分钟 | 日志: %s\n' "$((SECONDS / 60))" "$LOGDIR"

if ((failed > 0)); then
  printf '有 %d 个任务未成功，请检查 %s/queue.log 和对应任务日志。\n' "$failed" "$LOGDIR"
  exit 1
fi
printf '全部 33 个音频下游任务成功。\n'

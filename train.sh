#!/usr/bin/env bash
# train.sh —— 由元数据生成训练命令，再按并发槽位执行
#
# 用法:
#   bash train.sh --dry-run  # 只打印生成的命令，不训练、不创建日志
#   bash train.sh            # 串行跑（1 个槽位，最安全）
#   bash train.sh 2          # 同时跑 2 个任务
#   bash train.sh 4          # 同时跑 4 个任务
#
# 日常只需要修改下面两个元数据区：
#   1. EXPERIMENT_META：一次实验共享的 checkpoint、输出名、卡数等；
#   2. TASK_META：任务类型、任务名，以及可选的逐任务 --set 覆盖。
# render_command() 是唯一的命令模板，不再手写和复制 15 条 torchrun 命令。
#
# 调度: N 个槽位，第 j 个槽位依次执行列表第 j、j+N、j+2N … 条命令。
# 槽位内部严格串行、槽位之间并发。单个任务失败只记录，不中断其余队列；
# 全部结束后统一汇总，存在失败时脚本以非 0 退出。
#
# ⚠ 每条命令都使用 nproc_per_node 指定的同一组可见 GPU。N>1 表示多个任务共享
# 这些 GPU，而不是每个任务独占一组 GPU。启动前请先用 nvidia-smi 检查显存。

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
用法:
  bash train.sh --dry-run  只校验元数据并打印生成的训练命令
  bash train.sh [并发数]  执行训练；并发数默认为 1，且必须是正整数
EOF
}

die() {
  printf 'train.sh: %s\n' "$*" >&2
  exit 2
}

# ==================== 实验元数据（通常只改这里）====================
# 切换实验只需改变 checkpoint 与 run_name；output_root/model_name 一般保持不变。
# scratch 示例：
#   [checkpoint]=""
#   [run_name]="FaVoR-112px-48f-8fps-scratch"
# 官方 V-JEPA 2.1 示例：
#   [checkpoint]="CKPT/vjepa2/vitl.pt"
#   [run_name]="FaVoR-112px-48f-8fps-vjepaori"
declare -A EXPERIMENT_META=(
  [nproc_per_node]="2"
  [master_port_base]="12344"
  [checkpoint]="OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt"
  [output_root]="OUTPUT/finetune_v"
  [model_name]="vitl16"
  [run_name]="FaVoR-112px-48f-8fps-0-32-bestrankme"
)

# 所有任务共享的额外 --set 覆盖；每项必须是一个完整的 key=value。
# 例如：EXTRA_SET=("meta.seed=7" "optimization.epochs=20")
EXTRA_SET=()

# ==================== 任务元数据（按列表顺序执行）====================
# 格式：任务分组|任务名|可选逐任务覆盖
# 分组只能是 cls / reg / mlcls；第三列用分号分隔多个 key=value，可留空。
# 例："cls|RAVDESS-emotion|meta.seed=7;optimization.epochs=20"
TASK_META=(
  "cls|CREMA-D-emotion|"
  "cls|EmotionTalk-emotion|"
  "cls|IEMOCAP-emotion|"
  "cls|MER2023-emotion|"
  "cls|MER242526-emotion|"
  "cls|RAVDESS-emotion|"
  "cls|RAVDESS-intensity|"
  "reg|AVEC2014-PHQ|"
  "reg|CREMA-D-intensity|"
  "reg|IEMOCAP-activation|"
  "reg|IEMOCAP-dominance|"
  "reg|IEMOCAP-valence|"
  "reg|MER2023-pos_intensity|"
  "reg|MER242526-pos_intensity|"
  "mlcls|MER242526-26openset|"
)

# ==================== 命令模板与元数据校验 ====================
CMDS=()
declare -A SEEN_TASKS=()

validate_metadata() {
  local required key checkpoint last_port
  required=(nproc_per_node master_port_base checkpoint output_root model_name run_name)
  for key in "${required[@]}"; do
    [[ -v "EXPERIMENT_META[$key]" ]] || die "EXPERIMENT_META 缺少字段: $key"
  done
  [[ "${EXPERIMENT_META[nproc_per_node]}" =~ ^[1-9][0-9]*$ ]] \
    || die "nproc_per_node 必须是正整数"
  [[ "${EXPERIMENT_META[master_port_base]}" =~ ^[1-9][0-9]*$ ]] \
    || die "master_port_base 必须是正整数"
  [[ -n "${EXPERIMENT_META[output_root]}" ]] || die "output_root 不能为空"
  [[ -n "${EXPERIMENT_META[model_name]}" ]] || die "model_name 不能为空"
  [[ -n "${EXPERIMENT_META[run_name]}" ]] || die "run_name 不能为空"
  [[ "${EXPERIMENT_META[run_name]}" != */* ]] || die "run_name 不能包含 /"
  checkpoint=${EXPERIMENT_META[checkpoint]}
  [[ -z "$checkpoint" || -f "$checkpoint" ]] \
    || die "checkpoint 不存在: $checkpoint"
  ((${#TASK_META[@]} > 0)) || die "TASK_META 不能为空"
  last_port=$((EXPERIMENT_META[master_port_base] + ${#TASK_META[@]} - 1))
  ((last_port <= 65535)) || die "生成的 master_port 超出 65535: $last_port"
}

# 唯一的命令模板：把一行任务元数据渲染成经过 shell 转义的 torchrun 命令。
render_command() {
  local index=$1 row=$2
  local group config_group task task_overrides config folder port command override
  local -a argv task_set=()

  IFS='|' read -r group task task_overrides <<<"$row"
  case "$group" in
    cls) config_group=classification ;;
    reg) config_group=regression ;;
    mlcls) config_group=multilabel ;;
    *) die "TASK_META 第 $((index + 1)) 行分组无效: $group" ;;
  esac
  [[ "$task" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die "TASK_META 第 $((index + 1)) 行任务名无效: $task"
  [[ -z "${SEEN_TASKS[$task]+x}" ]] || die "TASK_META 任务重复: $task"
  SEEN_TASKS[$task]=1

  config="CONFIGS/tasks/finetune/video/$config_group/${task//_/-}.yaml"
  [[ -f "$config" ]] || die "任务配置不存在: $config"
  folder="${EXPERIMENT_META[output_root]}/${EXPERIMENT_META[model_name]}/$task/${EXPERIMENT_META[run_name]}"
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

  # %q 使空字符串、空格和 shell 特殊字符都能安全传给后续的 bash -c。
  printf -v command '%q ' "${argv[@]}"
  CMDS+=("${command% }")
}

validate_metadata
for index in "${!TASK_META[@]}"; do
  render_command "$index" "${TASK_META[$index]}"
done

TOTAL=${#CMDS[@]}
((TOTAL > 0)) || die "没有生成任何训练命令"

# ==================== 参数解析与 dry-run ====================
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
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  usage >&2
  die "并发数需为正整数，当前传入: $N"
fi
((N > TOTAL)) && N=$TOTAL

if [[ "$DRY_RUN" == true ]]; then
  printf '实验: %s\n' "${EXPERIMENT_META[run_name]}"
  printf 'checkpoint: %s\n' "${EXPERIMENT_META[checkpoint]:-(scratch / empty)}"
  printf '由 %d 条任务元数据生成 %d 条命令：\n\n' "${#TASK_META[@]}" "$TOTAL"
  for index in "${!CMDS[@]}"; do
    printf '[%02d] %s\n' "$((index + 1))" "${CMDS[$index]}"
  done
  exit 0
fi

# ==================== 并发执行器（一般不用改）====================
STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="OUTPUT/train_sh_logs/$STAMP"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
: > "$QUEUE_LOG"

qlog() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"; }

# 从命令里取 --fname 的 basename 作为任务名（仅用于日志与汇总）。
job_name() {
  local fname base
  fname=$(sed -n 's/.*--fname[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p' <<<"$1")
  base=${fname##*/}
  printf '%s' "${base%.yaml}"
}

# 跑一条命令，退出码落盘到 .rc-<编号>，供汇总阶段读取。
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

printf '候选命令 %d 条，并发槽位 %d 个\n' "$TOTAL" "$N"
((N > 1)) && printf '注意: %d 个任务共享同一组 GPU，请确认显存余量以免 OOM\n' "$N"
printf '实验: %s\ncheckpoint: %s\n' \
  "${EXPERIMENT_META[run_name]}" "${EXPERIMENT_META[checkpoint]:-(scratch / empty)}"
printf '日志目录: %s\n\n' "$LOGDIR"

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
  printf '有 %d 个任务未成功，逐个排查: %s/queue.log 与同目录下的 .log\n' \
    "$failed" "$LOGDIR"
  exit 1
fi
printf '全部成功。\n'

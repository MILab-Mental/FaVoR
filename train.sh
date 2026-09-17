#!/usr/bin/env bash
# train.sh —— 候选训练命令的「并发队列」执行器
#
# 用法:
#   bash train.sh          # 串行跑（1 个槽位，最安全）
#   bash train.sh 2        # 同时跑 2 个任务
#   bash train.sh 4        # 同时跑 4 个任务
#
# 调度: N 个槽位，第 j 个槽位依次执行列表第 j、j+N、j+2N … 条命令。
#   槽位内部严格按列表顺序、前一个结束才启动下一个；槽位之间并发。
#   例 N=2 → [槽1] 1→3→5… ，[槽2] 2→4→6…  （任意时刻最多 2 个任务在跑）
#
# ⚠ 本机只有 2 张卡（cuda:0 / cuda:1，由 app/main.py 按 LOCAL_RANK 分配），
#   下面每条命令用的都是这一对卡 —— N>1 是「多个任务共享同一对卡」（靠显存多路
#   复用），不是各占一对卡。跑之前先看余量:
#     nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
#   两张卡各 98 GiB；单任务约占 20+ GiB，N 取 2 左右较稳妥，N 越大越可能 OOM。
#
# 失败处理: 单个任务失败只记录、不中断队列，全部跑完后汇总并以非 0 退出。
# 跳过某个任务: 把 CMDS 数组里那一行注释掉即可（编号会自动重排）。
# 日志: OUTPUT/train_sh_logs/<时间戳>/ 下一个任务一个 .log（含完整命令），
#       另有 queue.log 记录调度事件（谁在何时开始 / 结束 / 退出码）。
#
# 这 15 条与 run.sh 末尾「vjepaori 基线」段完全一致 —— run.sh 是逐条手工执行、
# 无调度；train.sh 负责批量与并发。

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== 候选命令（按列表顺序执行）====================
CMDS=(
"torchrun --master_port 12345 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/CREMA-D-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/CREMA-D-emotion/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12347 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/EmotionTalk-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/EmotionTalk-emotion/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12350 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/IEMOCAP-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-emotion/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12352 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER2023-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/MER2023-emotion/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12355 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12357 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/RAVDESS-emotion/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12358 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-intensity.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/RAVDESS-intensity/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12344 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/AVEC2014-PHQ.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/AVEC2014-PHQ/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12346 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/CREMA-D-intensity.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/CREMA-D-intensity/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12348 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-activation.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-activation/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12349 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-dominance.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-dominance/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12351 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-valence.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-valence/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12353 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER2023-pos_intensity.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/MER2023-pos_intensity/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12356 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER242526-pos_intensity.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/MER242526-pos_intensity/FaVoR-112px-48f-8fps-0-32-bestrankme"
"torchrun --master_port 12354 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/mlcls/MER242526-26openset.yaml --set meta.read_checkpoint=OUTPUT/pretrain_v/vitl16/FaVoR-112px-64f16-rankme/best_rankme.pt folder=OUTPUT/finetune_v/vitl16/MER242526-26openset/FaVoR-112px-48f-8fps-0-32-bestrankme"
)

# CMDS=(
# "torchrun --master_port 12345 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/CREMA-D-emotion.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/CREMA-D-emotion/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12347 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/EmotionTalk-emotion.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/EmotionTalk-emotion/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12350 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/IEMOCAP-emotion.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/IEMOCAP-emotion/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12352 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER2023-emotion.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/MER2023-emotion/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12355 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12357 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/RAVDESS-emotion/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12358 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-intensity.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/RAVDESS-intensity/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12344 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/AVEC2014-PHQ.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/AVEC2014-PHQ/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12346 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/CREMA-D-intensity.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/CREMA-D-intensity/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12348 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-activation.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/IEMOCAP-activation/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12349 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-dominance.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/IEMOCAP-dominance/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12351 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-valence.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/IEMOCAP-valence/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12353 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER2023-pos_intensity.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/MER2023-pos_intensity/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12356 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER242526-pos_intensity.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/MER242526-pos_intensity/FaVoR-112px-48f-8fps-scratch"
# "torchrun --master_port 12354 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/mlcls/MER242526-26openset.yaml --set meta.read_checkpoint='' folder=OUTPUT/finetune_v/vitl16/MER242526-26openset/FaVoR-112px-48f-8fps-scratch"
# )

# CMDS=(
# "torchrun --master_port 12345 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/CREMA-D-emotion.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/CREMA-D-emotion/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12347 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/EmotionTalk-emotion.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/EmotionTalk-emotion/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12350 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/IEMOCAP-emotion.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-emotion/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12352 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER2023-emotion.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/MER2023-emotion/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12355 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/MER242526-emotion.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/MER242526-emotion/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12357 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-emotion.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/RAVDESS-emotion/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12358 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/cls/RAVDESS-intensity.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/RAVDESS-intensity/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12344 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/AVEC2014-PHQ.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/AVEC2014-PHQ/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12346 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/CREMA-D-intensity.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/CREMA-D-intensity/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12348 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-activation.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-activation/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12349 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-dominance.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-dominance/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12351 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/IEMOCAP-valence.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/IEMOCAP-valence/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12353 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER2023-pos_intensity.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/MER2023-pos_intensity/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12356 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/reg/MER242526-pos_intensity.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/MER242526-pos_intensity/FaVoR-112px-48f-8fps-vjepaori"
# "torchrun --master_port 12354 --nproc_per_node=2 -m app.main --fname CONFIGS/tasks/vfinetune/mlcls/MER242526-26openset.yaml --set meta.read_checkpoint=CKPT/vjepa2/vitl.pt folder=OUTPUT/finetune_v/vitl16/MER242526-26openset/FaVoR-112px-48f-8fps-vjepaori"
# )




# ==================== 执行器（一般不用改）====================
TOTAL=${#CMDS[@]}
N=${1:-1}

if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  printf '用法: bash train.sh [并发数]\n  并发数需为正整数，当前传入: %s\n' "$N" >&2
  exit 2
fi
(( N > TOTAL )) && N=$TOTAL

STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="OUTPUT/train_sh_logs/$STAMP"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
: > "$QUEUE_LOG"

qlog() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$QUEUE_LOG"; }

# 从命令里取 --fname 的 basename 作为任务名（仅用于日志与汇总）
job_name() {
  local fname base
  fname=$(sed -n 's/.*--fname[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p' <<<"$1")
  base=${fname##*/}
  printf '%s' "${base%.yaml}"
}

# 跑一条命令，退出码落盘到 .rc-<编号>，供汇总阶段读取
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
  if (( rc == 0 )); then
    qlog "[槽$slot] 完成 $idx/$TOTAL $name ($(( SECONDS - start ))s)"
  else
    qlog "[槽$slot] 失败 $idx/$TOTAL $name 退出码=$rc ($(( SECONDS - start ))s)"
  fi
}

on_interrupt() {
  trap - INT TERM
  printf '\n'
  qlog "收到中断信号，终止全部槽位及其训练进程"
  kill 0 2>/dev/null   # 当前进程组：本脚本 + 各槽位 + torchrun 及其 worker
  exit 130
}

printf '候选命令 %d 条，并发槽位 %d 个\n' "$TOTAL" "$N"
(( N > 1 )) && printf '注意: %d 个任务共享同一对卡，先确认显存余量（nvidia-smi）以免 OOM\n' "$N"
printf '日志目录: %s\n\n' "$LOGDIR"

pids=()
for (( slot=1; slot<=N; slot++ )); do
  (
    for (( idx=slot; idx<=TOTAL; idx+=N )); do
      run_one "$slot" "$idx" "${CMDS[$((idx-1))]}"
    done
  ) &
  pids+=("$!")
done

trap on_interrupt INT TERM
for pid in "${pids[@]}"; do wait "$pid"; done
trap - INT TERM

printf '\n==================== 队列结果 ====================\n'
failed=0
for (( idx=1; idx<=TOTAL; idx++ )); do
  rc=$(cat "$LOGDIR/.rc-$idx" 2>/dev/null)
  name=$(job_name "${CMDS[$((idx-1))]}")
  if [[ "$rc" == "0" ]]; then
    printf '  [OK]   %02d/%02d  %s\n' "$idx" "$TOTAL" "$name"
  else
    printf '  [FAIL] %02d/%02d  %s  (退出码=%s)\n' "$idx" "$TOTAL" "$name" "${rc:-未执行}"
    failed=$((failed + 1))
  fi
done
printf '\n总耗时 %d 分钟 | 日志: %s\n' "$(( SECONDS / 60 ))" "$LOGDIR"

if (( failed > 0 )); then
  printf '有 %d 个任务未成功，逐个排查: %s/queue.log 与同目录下的 .log\n' "$failed" "$LOGDIR"
  exit 1
fi
printf '全部成功。\n'

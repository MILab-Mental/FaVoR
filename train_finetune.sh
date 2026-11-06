#!/usr/bin/env bash
# 读取任务 CSV，按顺序训练；直接调用 torchrun，不生成 YAML，也不套 Python。
# 常用：指定显卡、权重、任务范围；DRY_RUN=1 可以先预览命令。
# CUDA_VISIBLE_DEVICES=0 SCOPE=mental CHECKPOINT=/path/model.pt \
#   RUN_NAME=my-run DRY_RUN=1 bash train_finetune.sh a
# a=音频，v=视频，va=音视频；末尾可加 data.batch_size=16 等覆盖参数。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
用法：CUDA_VISIBLE_DEVICES=0 CHECKPOINT=/path/model.pt RUN_NAME=experiment
      bash train_finetune.sh a|v|va [KEY=VALUE ...]

环境变量（不填则使用默认值）：
  CHECKPOINT  必填，已有的预训练权重路径；预览也需要
  RUN_NAME    实验名，默认取权重文件名（去掉扩展名）
  SCOPE       all / mental / nonmental，默认 all
  TASKS       任务名筛选，支持逗号和通配符，例如 'DSM-*,IEMOCAP-*'
  TYPE        all / classification / regression / multi_label_classification
  TASK_CSV    任务列表，默认 CONFIGS/tasks/finetune/tasks.csv
  DRY_RUN     1=只打印命令，0=开始训练，默认 0

多种筛选条件同时生效。显卡数量根据 CUDA_VISIBLE_DEVICES 自动确定。
参数优先级：默认配置 < CSV < CHECKPOINT < 命令末尾的 KEY=VALUE。
训练失败立即停止；所有选中任务先检查完，再开始第一项训练。
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
# 1. 读取启动参数，并检查权重、筛选条件。
mode=${1:-}
case "$mode" in a|v|va) shift ;; *) usage >&2; exit 1 ;; esac
extra=("$@")
for item in "${extra[@]}"; do
  [[ $item == ?*=* ]] || die "Override must be KEY=VALUE: $item"
done
checkpoint=${CHECKPOINT:-}
[[ -n $checkpoint && -f $checkpoint ]] || die 'Set CHECKPOINT to an existing pretrained checkpoint.'
run=${RUN_NAME:-$(basename -- "${checkpoint%.*}")}
[[ $run =~ ^[a-zA-Z0-9_.-]+$ ]] || die 'RUN_NAME may contain letters, numbers, dot, underscore and hyphen.'
catalog=${TASK_CSV:-CONFIGS/tasks/finetune/tasks.csv}
scope=${SCOPE:-all}
type=${TYPE:-all}
dry_run=${DRY_RUN:-0}
case "$scope" in all|mental|nonmental) ;; *) die "Invalid SCOPE: $scope" ;; esac
case "$type" in all|classification|regression|multi_label_classification) ;; *) die "Invalid TYPE: $type" ;; esac
[[ $dry_run == 0 || $dry_run == 1 ]] || die 'DRY_RUN must be 0 or 1.'
[[ -f $catalog ]] || die "Missing catalog: $catalog"

# CSV 解析细节：支持带引号的逗号、双引号转义和 Windows 换行；不支持跨行字段。
# 这里不执行 CSV 中的 Shell 代码，参数始终通过数组传递。
csv_parse() {
  local line=${1%$'\r'} field='' quoted=0 closed=0 ch next i
  CSV_FIELDS=()
  for ((i=0; i<${#line}; i++)); do
    ch=${line:i:1}; next=${line:i+1:1}
    if ((quoted)); then
      if [[ $ch == '"' ]]; then
        if [[ $next == '"' ]]; then field+='"'; i=$((i+1))
        else quoted=0; closed=1; fi
      else field+=$ch; fi
    elif [[ $ch == ',' ]]; then
      CSV_FIELDS+=("$field"); field=''; closed=0
    elif [[ $ch == '"' ]]; then
      [[ -z $field && $closed == 0 ]] || die "Invalid CSV quote at line $line_number"
      quoted=1
    else
      ((closed == 0)) || die "Characters after CSV closing quote at line $line_number"
      field+=$ch
    fi
  done
  ((quoted == 0)) || die "Unclosed quote/multiline CSV field at line $line_number"
  CSV_FIELDS+=("$field")
}

matches_task() {
  local pattern
  [[ -z ${TASKS:-} ]] && return 0
  local -a patterns
  IFS=',' read -r -a patterns <<< "$TASKS"
  for pattern in "${patterns[@]}"; do
    [[ $1 == $pattern ]] && return 0
  done
  return 1
}

# 将一行 CSV 转成 row[列名]，后续代码就不用记列号了。
read_row() {
  csv_parse "$1"
  ((${#CSV_FIELDS[@]} == ${#header[@]})) || die "Wrong number of CSV fields at line $line_number"
  row=()
  local i
  for i in "${!header[@]}"; do row[${header[i]}]=${CSV_FIELDS[i]}; done
}

# 2. 把一条任务记录组装成命令；这里只检查参数，不启动训练。
build_command() {
  local id=${row[task_id]} default folder dataset_path dataset_yaml item
  local -a overrides=()
  [[ ${row[num_class]} =~ ^[1-9][0-9]*$ ]] || die "Invalid num_class for $id"
  [[ -f ${row[csv_path]} ]] || die "Missing dataset CSV for $id: ${row[csv_path]}"
  [[ -n ${row[label_column]} && -n ${row[output_dir]} ]] || die "Missing label/output for $id"
  case ${row[task_type]} in classification|regression|multi_label_classification) ;; *) die "Invalid task type for $id" ;; esac
  case ${row[profile]} in clip|instance) ;; *) die "Invalid profile for $id" ;; esac
  default="CONFIGS/tasks/finetune/defaults/$mode/${row[profile]}/default.yaml"
  [[ -f $default ]] || die "Missing default: $default"
  folder=${row[output_dir]//\{modality\}/$mode}
  folder=${folder//\{task_id\}/$id}
  folder=${folder//\{run\}/$run}
  # 数据集路径是 YAML 列表：单引号在 YAML 中写成两个单引号。
  dataset_path=${row[csv_path]//\'/\'\'}
  dataset_yaml="['$dataset_path']"
  cmd=(torchrun --standalone --nproc_per_node=gpu -m app.main --fname "$default" --set
       "data.datasets=$dataset_yaml" 'data.datasets_weights=[1]'
       "data.task=${row[task_type]}" "data.num_class=${row[num_class]}"
       "data.label_column=${row[label_column]}" "folder=$folder")
  if [[ $mode == va ]]; then
    [[ ${row[best_metric_mode]} == min || ${row[best_metric_mode]} == max ]] || die "Invalid best_metric_mode for $id"
    cmd+=("finetune.best_metric=${row[best_metric]}" "finetune.best_metric_mode=${row[best_metric_mode]}")
  fi
  # CSV 的任务特定参数用分号分隔。
  if [[ -n ${row[overrides]} ]]; then
    IFS=';' read -r -a overrides <<< "${row[overrides]}"
    for item in "${overrides[@]}"; do
      [[ $item == ?*=* ]] || die "Invalid CSV override for $id: $item"
      cmd+=("$item")
    done
  fi
  # 最后加入权重和命令行参数，保证覆盖优先级。
  if [[ $mode == va ]]; then cmd+=("finetune.pretrained_checkpoint=$checkpoint")
  else cmd+=("meta.read_checkpoint=$checkpoint"); fi
  cmd+=("${extra[@]}")
}

# 3. 先扫描完整 CSV：筛选任务、检查配置，再保存待运行的 CSV 行。
# 队列保存原始行，执行时重新组装命令，不再拆分/拼接一大串命令参数。
declare -a CSV_FIELDS=() header=() queue=() cmd=()
declare -A row seen
line_number=0
while IFS= read -r line || [[ -n $line ]]; do
  line_number=$((line_number+1))
  if ((line_number == 1)); then line=${line#$'\xef\xbb\xbf'}; fi
  [[ -n ${line%$'\r'} ]] || continue
  if ((line_number == 1)); then
    csv_parse "$line"
    header=("${CSV_FIELDS[@]}")
    expected='enabled task_id dataset scope task_type num_class csv_path label_column modalities profile output_dir best_metric best_metric_mode overrides'
    [[ ${header[*]} == "$expected" ]] || die "Unexpected CSV header; expected: $expected"
    continue
  fi
  read_row "$line"
  id=${row[task_id]}
  [[ -n $id && ! ${seen[$id]+exists} ]] || die "Empty/duplicate task_id at line $line_number: $id"
  seen[$id]=1
  [[ ${row[enabled]} == 0 || ${row[enabled]} == 1 ]] || die "Invalid enabled for $id"
  [[ ${row[enabled]} == 1 ]] || continue
  [[ $scope == all || ${row[scope]} == "$scope" ]] || continue
  [[ $type == all || ${row[task_type]} == "$type" ]] || continue
  [[ "|${row[modalities]}|" == *"|$mode|"* ]] || continue
  matches_task "$id" || continue
  build_command
  queue+=("$line")
done < "$catalog"
((${#queue[@]} > 0)) || die 'No tasks selected; check SCOPE, TYPE, TASKS and modality.'
printf 'Selected %d tasks; modality=%s; run=%s; GPUs=%s\n' "${#queue[@]}" "$mode" "$run" "${CUDA_VISIBLE_DEVICES:-all visible}"

# 4. 按 CSV 原顺序训练；预览只打印，真实运行遇到失败就停止。
for i in "${!queue[@]}"; do
  read_row "${queue[i]}"
  build_command
  printf '\n[%d/%d] %s\n' "$((i+1))" "${#queue[@]}" "${row[task_id]}"
  printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ $dry_run == 0 ]]; then
    "${cmd[@]}" || {
      status=$?
      printf 'Failed: %s (exit %d); queue stopped.\n' "${row[task_id]}" "$status" >&2
      exit "$status"
    }
  fi
done

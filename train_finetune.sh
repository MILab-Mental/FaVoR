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
  CHECKPOINT  预训练权重路径；普通任务必填，mental-auto 可使用路线默认值
  VA_ROUTE    音视频路线：anxiety-best / depression-best / total-best；
              仅用于 va，可选 SCOPE=mental / mental_core / nonmental，默认 mental
  MENTAL_AUTO VA_ROUTE 的兼容别名；默认权重可由 CHECKPOINT 覆盖
  RUN_NAME    实验名，默认取权重文件名（去掉扩展名）
  SCOPE       all / mental / mental_core / nonmental，默认 all
  TASKS       任务名筛选，支持逗号和通配符，例如 'DSM-*,IEMOCAP-*'
  TYPE        all / classification / regression / multi_label_classification
  TASK_CSV    任务列表，默认 CONFIGS/tasks/finetune/tasks.csv
  DRY_RUN     1=只打印命令，0=开始训练，默认 0

多种筛选条件同时生效。显卡数量根据 CUDA_VISIBLE_DEVICES 自动确定。
参数优先级：默认配置 < CSV < CHECKPOINT < 命令末尾的 KEY=VALUE。
训练失败后继续下一个任务；全部任务结束后如有失败则返回非零状态。
仅跳过带完成标记的任务；无标记的旧目录会保留并从头重跑。
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
[[ -z ${VA_ROUTE:-} || -z ${MENTAL_AUTO:-} || $VA_ROUTE == "$MENTAL_AUTO" ]] || die 'VA_ROUTE and MENTAL_AUTO disagree.'
mental_auto=${VA_ROUTE:-${MENTAL_AUTO:-}}
checkpoint=${CHECKPOINT:-}
if [[ -n $mental_auto ]]; then
  [[ $mode == va ]] || die 'MENTAL_AUTO requires modality va.'
  case "$mental_auto" in anxiety-best|depression-best|total-best) ;; *) die "Invalid MENTAL_AUTO: $mental_auto" ;; esac
  route_config="CONFIGS/tasks/finetune/audio-video/mental-auto/$mental_auto.yaml"
  [[ -f $route_config ]] || die "Missing mental-auto route: $route_config"
  if [[ -z $checkpoint ]]; then
    checkpoint=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["finetune"]["pretrained_checkpoint"])' "$route_config")
  fi
  run=${RUN_NAME:-$mental_auto}
else
  run=${RUN_NAME:-$(basename -- "${checkpoint%.*}")}
fi
[[ -n $checkpoint && -f $checkpoint ]] || die 'Set CHECKPOINT to an existing pretrained checkpoint.'
[[ $run =~ ^[a-zA-Z0-9_.-]+$ ]] || die 'RUN_NAME may contain letters, numbers, dot, underscore and hyphen.'
catalog=${TASK_CSV:-CONFIGS/tasks/finetune/tasks.csv}
scope=${SCOPE:-$([[ -n $mental_auto ]] && printf mental || printf all)}
[[ -z $mental_auto || $scope == mental || $scope == mental_core || $scope == nonmental ]] || die 'VA_ROUTE requires SCOPE=mental, mental_core, or nonmental.'
type=${TYPE:-all}
dry_run=${DRY_RUN:-0}
case "$scope" in all|mental|mental_core|nonmental) ;; *) die "Invalid SCOPE: $scope" ;; esac
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
  local id=${row[task_id]} default folder output_group dataset_path dataset_yaml item
  local -a overrides=()
  [[ ${row[num_class]} =~ ^[1-9][0-9]*$ ]] || die "Invalid num_class for $id"
  [[ -f ${row[csv_path]} ]] || die "Missing dataset CSV for $id: ${row[csv_path]}"
  [[ -n ${row[label_column]} && -n ${row[output_dir]} ]] || die "Missing label/output for $id"
  case ${row[task_type]} in classification|regression|multi_label_classification) ;; *) die "Invalid task type for $id" ;; esac
  case ${row[profile]} in clip|instance) ;; *) die "Invalid profile for $id" ;; esac
  case ${row[scope]} in
    nonmental) output_group=ordinary ;;
    mental) output_group=mental ;;
    *) die "Invalid scope for $id: ${row[scope]}" ;;
  esac
  default="CONFIGS/tasks/finetune/defaults/$mode/${row[profile]}/default.yaml"
  if [[ -n $mental_auto ]]; then
    [[ ${row[scope]} != mental || ${row[profile]} == instance ]] || die "Invalid mental-auto task profile for $id"
    default=$route_config
  fi
  [[ -f $default ]] || die "Missing default: $default"
  folder=${row[output_dir]//\{modality\}/$mode}
  folder=${folder//\{catalog\}/$output_group}
  folder=${folder//\{task_id\}/$id}
  folder=${folder//\{run\}/$run}
  job_folder=$folder
  # 数据集路径是 YAML 列表：单引号在 YAML 中写成两个单引号。
  dataset_path=${row[csv_path]//\'/\'\'}
  dataset_yaml="['$dataset_path']"
  cmd=(torchrun --standalone --nproc_per_node=gpu -m app.main --fname "$default" --set
       "data.datasets=$dataset_yaml" 'data.datasets_weights=[1]'
       "data.task=${row[task_type]}" "data.num_class=${row[num_class]}"
       "data.label_column=${row[label_column]}" "folder=$folder")
  if [[ -n $mental_auto && ${row[profile]} == clip ]]; then
    cmd+=("data.num_sentence=0")
  fi
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
      [[ $item != folder=* ]] || job_folder=${item#folder=}
    done
  fi
  # 最后加入权重和命令行参数，保证覆盖优先级。
  if [[ $mode == va ]]; then cmd+=("finetune.pretrained_checkpoint=$checkpoint")
  else cmd+=("meta.read_checkpoint=$checkpoint"); fi
  cmd+=("${extra[@]}")
  for item in "${extra[@]}"; do
    [[ $item != folder=* ]] || job_folder=${item#folder=}
  done
  [[ -n $job_folder ]] || die "Empty output folder for $id"
  legacy_folder=
  if [[ $job_folder == "OUTPUT/finetune_$mode/$output_group/$id/$run" ]]; then
    legacy_folder="OUTPUT/finetune_$mode/catalog/$id/$run"
    if [[ -n $mental_auto && $output_group == mental ]]; then
      legacy_folder="OUTPUT/finetune_va/mental-transfer/$id/$run"
    fi
  fi
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
    expected='enabled task_id dataset scope mental_core task_type num_class csv_path label_column modalities profile output_dir best_metric best_metric_mode overrides'
    [[ ${header[*]} == "$expected" ]] || die "Unexpected CSV header; expected: $expected"
    continue
  fi
  read_row "$line"
  id=${row[task_id]}
  [[ -n $id && ! ${seen[$id]+exists} ]] || die "Empty/duplicate task_id at line $line_number: $id"
  seen[$id]=1
  [[ ${row[enabled]} == 0 || ${row[enabled]} == 1 ]] || die "Invalid enabled for $id"
  [[ ${row[mental_core]} == 0 || ${row[mental_core]} == 1 ]] || die "Invalid mental_core for $id"
  [[ ${row[enabled]} == 1 ]] || continue
  if [[ $scope == mental_core ]]; then
    [[ ${row[scope]} == mental && ${row[mental_core]} == 1 ]] || continue
  elif [[ $scope != all ]]; then
    [[ ${row[scope]} == "$scope" ]] || continue
  fi
  [[ $type == all || ${row[task_type]} == "$type" ]] || continue
  [[ "|${row[modalities]}|" == *"|$mode|"* ]] || continue
  matches_task "$id" || continue
  build_command
  queue+=("$line")
done < "$catalog"
((${#queue[@]} > 0)) || die 'No tasks selected; check SCOPE, TYPE, TASKS and modality.'
printf 'Selected %d tasks; modality=%s; run=%s; GPUs=%s\n' "${#queue[@]}" "$mode" "$run" "${CUDA_VISIBLE_DEVICES:-all visible}"

is_complete() {
  local folder=$1
  [[ -f $folder/.finetune-complete && -s $folder/best.pt && -s $folder/logs/history.csv && -s $folder/logs/best.csv ]]
}

# 4. 仅成功退出且产物完整才标记完成，未完成目录移走以免混用历史记录。
failed=0
for i in "${!queue[@]}"; do
  read_row "${queue[i]}"
  build_command
  printf '\n[%d/%d] %s\n' "$((i+1))" "${#queue[@]}" "${row[task_id]}"
  if is_complete "$job_folder"; then
    printf 'Skipped completed task: %s (%s)\n' "${row[task_id]}" "$job_folder"
    continue
  fi
  if [[ -n $legacy_folder && ! -e $job_folder && ! -L $job_folder ]] && is_complete "$legacy_folder"; then
    printf 'Skipped completed task: %s (%s)\n' "${row[task_id]}" "$legacy_folder"
    continue
  fi
  printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ $dry_run == 0 ]]; then
    if [[ -e $job_folder || -L $job_folder ]]; then
      backup="${job_folder}.incomplete-$(date +%Y%m%d-%H%M%S)"
      suffix=0
      while [[ -e $backup || -L $backup ]]; do
        suffix=$((suffix+1))
        backup="${job_folder}.incomplete-$(date +%Y%m%d-%H%M%S)-$suffix"
      done
      mv -- "$job_folder" "$backup"
      printf 'Preserved incomplete output: %s\n' "$backup"
    fi
    if "${cmd[@]}"; then
      if [[ -s $job_folder/best.pt && -s $job_folder/logs/history.csv && -s $job_folder/logs/best.csv ]]; then
        touch -- "$job_folder/.finetune-complete"
      else
        printf 'Failed: %s (missing expected training output).\n' "${row[task_id]}" >&2
        failed=$((failed+1))
      fi
    else
      status=$?
      printf 'Failed: %s (exit %d); continuing.\n' "${row[task_id]}" "$status" >&2
      failed=$((failed+1))
    fi
  elif [[ -e $job_folder || -L $job_folder ]]; then
    printf 'Would preserve incomplete output and restart: %s\n' "$job_folder"
  fi
done
if ((failed)); then
  printf '\n%d task(s) failed.\n' "$failed" >&2
  exit 1
fi

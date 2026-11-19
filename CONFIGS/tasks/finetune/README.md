# CSV fine-tuning entry

`train_finetune.sh` reads `CONFIGS/tasks/finetune/tasks.csv` and launches `torchrun -m app.main` directly.
The launcher is Bash; it does not call a Python wrapper or generate task YAMLs.
The 177 standard task entry YAMLs are retained under
`CONFIGS/archive/finetune-standard/`; the loader resolves their old paths for
existing commands. The 71 historical task data fragments are archived under
`CONFIGS/archive/finetune-data/`; the config loader maps individual old paths
for special experiments. BASELINES also reads `tasks.csv` for task discovery.
Sampling presets remain here.
The data preparation utilities read `tasks.csv` and no longer generate per-task
YAML files.

## Files and precedence

- `default.yaml`: shared defaults (`meta.save_latest=false`, patience 10).
- `defaults/{a,v,va}/{clip,instance}/default.yaml`: six modality/sampling entries.
  They reuse the existing model, sampling and optimization fragments, but do not
  reference task-specific YAMLs.
- `tasks.csv`: 71 unique tasks (38 mental + 33 nonmental). Audio supports all 71;
  video/VA support the 53 tasks whose CSVs contain both modalities.

Precedence: model/sampling/optimization fragments → shared defaults → entry YAML
→ CSV fields and overrides → selected CHECKPOINT → trailing KEY=VALUE arguments.

`clip` uses individual clips. `instance` groups eight sentences, including mental,
AVEC2014, Androids-Corpus, CMDC, EATD-Corpus and PDCH where applicable. Defaults
match the canonical old task configurations, including seed, batch size, epochs,
sampling, model and VA regression selection by MAE. Audio defaults to batch 16,
video to 64, VA to 4, all **per GPU**. VA defaults to 20 epochs; audio to 100 and
video to 80. These are not imports of the tuned `mental-auto/*-best.yaml` variants.

CHECKPOINT is required for standard runs: `meta.read_checkpoint` for a/v,
`finetune.pretrained_checkpoint` for va. Mental-auto routes include a default
checkpoint; CHECKPOINT overrides it. RUN_NAME distinguishes experiments in the
output path; use different names for different pretrained weights.

## CSV fields

| Field | Meaning |
|---|---|
| enabled | 1 runs the row; 0 disables it |
| task_id | Unique task name, also used by TASKS filters |
| dataset | Dataset identifier from the task registry |
| scope | mental or nonmental |
| task_type | classification, regression, multi_label_classification |
| num_class | Number of classes/labels; regression uses 1 |
| csv_path | Dataset CSV, resolved from the repository root |
| label_column | Exact label column, including underscores |
| modalities | Supported modes separated by `\|`: a or a\|v\|va |
| profile | clip or instance |
| output_dir | Output template; supports {modality}, {task_id}, {run} |
| best_metric / best_metric_mode | VA checkpoint/early-stop metric and min/max |
| overrides | Optional `KEY=VALUE;KEY=VALUE` for task-specific differences |

Example overrides: `data.batch_size=8;optimization.lr=0.0002`.
Values use the existing `app.main --set` syntax. CSV fields containing commas
must be quoted, with internal double quotes doubled; spreadsheet CSV exports
work. Keep each CSV record on one physical line. A semicolon separates override
entries and cannot appear inside an override value. No shell expansion/eval is
performed on CSV content.

Dataset media roots still come from `CONFIGS/datasets.csv`; add a new dataset
there as well as adding its task row here. Split `-1` exclusions in dataset CSVs
are respected by the existing loaders.

## Commands

Run from the repository root, or use an absolute path to the shell script.

Preview the 33 nonmental audio jobs without training:

```bash
CUDA_VISIBLE_DEVICES=0 SCOPE=nonmental DRY_RUN=1 \
CHECKPOINT=/home/data/sdd/mental/OUTPUTS/pretrain_a/emotion2vec-plus-large/FaVoR-2to6s/latest.pt \
RUN_NAME=favor-2to6s bash train_finetune.sh a
```

Run those jobs by removing `DRY_RUN=1`:

```bash
CUDA_VISIBLE_DEVICES=0 SCOPE=nonmental \
CHECKPOINT=/home/data/sdd/mental/OUTPUTS/pretrain_a/emotion2vec-plus-large/FaVoR-2to6s/latest.pt \
RUN_NAME=favor-2to6s bash train_finetune.sh a

CUDA_VISIBLE_DEVICES=1 SCOPE=nonmental \
CHECKPOINT=/home/data/sdc/FAVOR/CKPT/emotion2vec_plus_large/model.pt \
RUN_NAME=emotion2vec-official bash train_finetune.sh a
```

Use different terminals for parallel queues. GPU count follows the visible GPU
IDs (`--nproc_per_node=gpu`). Global batch = per-GPU batch × GPU count (without
gradient accumulation).

Run one video task with an explicit per-GPU batch override:

```bash
CUDA_VISIBLE_DEVICES=0,1 TASKS=MER242526-pos-intensity \
CHECKPOINT=/home/data/sdc/FAVOR/CKPT/vjepa2/vitl.pt RUN_NAME=vjepa2-official \
bash train_finetune.sh v data.batch_size=16
```

Preview mental VA with a different epoch budget and shared preflight cache:

```bash
CUDA_VISIBLE_DEVICES=0,1 SCOPE=mental DRY_RUN=1 \
CHECKPOINT=/home/data/sdd/mental/OUTPUTS/pretrain_va_lejepa/JEPA-e5-latest-C1/step-75000.pt \
RUN_NAME=jepa-75k bash train_finetune.sh va \
  optimization.epochs=100 \
  data.sync_preflight_file=OUTPUT/finetune_va/mental/va_sync_preflight.json
```

Use the tuned mental-auto route for all 38 mental tasks, or add TASKS to select
individual tasks. The three routes are `anxiety-best`, `depression-best` and
`total-best`; the default scope is mental and the modality must be va:

```bash
CUDA_VISIBLE_DEVICES=0,1 MENTAL_AUTO=anxiety-best DRY_RUN=1 \
  bash train_finetune.sh va

CUDA_VISIBLE_DEVICES=0,1 MENTAL_AUTO=depression-best TASKS=DSM-Depression \
  bash train_finetune.sh va
```

The default output is `OUTPUT/finetune_va/mental-transfer/<task>/<route>/`.
An explicit RUN_NAME replaces `<route>`, and CHECKPOINT overrides the route's
initialization. The original 108 entries are archived under
`CONFIGS/archive/finetune-mental-auto/`; their old app.main paths still resolve.
These routes use the transferred 100-epoch, patience-10 settings, not the
original 5-epoch autoresearch runs. Existing unmarked outputs are backed up and
restarted by this launcher; they are not automatically resumed.

Filters: SCOPE=all/mental/nonmental, TYPE=regression (or another task type),
TASKS='DSM-*,IEMOCAP-*'. Filters intersect. TASK_CSV selects another catalog.
Trailing overrides take priority for every selected job. The launcher validates
the selected queue before starting. After a successful process exit, the launcher
marks an output complete only if `best.pt`, `logs/history.csv` and `logs/best.csv`
are nonempty. Subsequent runs skip marked, intact outputs. Any existing output
without the marker (including legacy outputs whose completion cannot be proven)
is moved to a sibling `.incomplete-<timestamp>` directory and trained from
scratch. Failed jobs do not stop the queue; the launcher returns a nonzero status
afterward if any job failed. A preview makes no filesystem changes and validates
command/config construction, not GPU memory, media integrity or checkpoint
compatibility.

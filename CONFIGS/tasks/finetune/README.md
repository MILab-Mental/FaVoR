# CSV fine-tuning entry

`train_finetune.sh` reads `CONFIGS/tasks/finetune/tasks.csv` and launches `torchrun -m app.main` directly.
The launcher is Bash; it does not call a Python wrapper or generate task YAMLs.
Existing task/data configs are retained.

## Files and precedence

- `default.yaml`: shared defaults (`meta.save_latest=false`, patience 10).
- `defaults/{a,v,va}/{clip,instance}/default.yaml`: six modality/sampling entries.
  They reuse the existing model, sampling and optimization fragments, but do not
  reference task-specific YAMLs.
- `tasks.csv`: 71 unique tasks (38 mental + 33 nonmental). Audio supports all 71;
  video/VA support the 53 tasks whose CSVs contain both modalities.

Precedence: model/sampling/optimization fragments → shared defaults → entry YAML
→ CSV fields and overrides → explicit CHECKPOINT → trailing KEY=VALUE arguments.

`clip` uses individual clips. `instance` groups eight sentences, including mental,
AVEC2014, Androids-Corpus, CMDC, EATD-Corpus and PDCH where applicable. Defaults
match the canonical old task configurations, including seed, batch size, epochs,
sampling, model and VA regression selection by MAE. Audio defaults to batch 16,
video to 64, VA to 4, all **per GPU**. VA defaults to 20 epochs; audio to 100 and
video to 80. These are not imports of the tuned `mental-auto/*-best.yaml` variants.

CHECKPOINT is always explicit: `meta.read_checkpoint` for a/v,
`finetune.pretrained_checkpoint` for va. RUN_NAME distinguishes experiments in
the output path; use different names for different pretrained weights.

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

Filters: SCOPE=all/mental/nonmental, TYPE=regression (or another task type),
TASKS='DSM-*,IEMOCAP-*'. Filters intersect. TASK_CSV selects another catalog.
Trailing overrides take priority for every selected job. The launcher validates
the selected queue before starting, stops on the first failed job, and does not
automatically skip existing outputs. Checkpoints/resume follow the chosen
default and explicit overrides. A preview validates command/config construction,
not GPU memory, media integrity or checkpoint compatibility.

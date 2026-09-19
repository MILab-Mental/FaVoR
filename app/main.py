# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import logging
import multiprocessing as mp
import os
import pprint
import socket
import sys
import warnings
from pathlib import Path

# 过滤所有 warning：Python warnings + PyTorch C++ 日志（如 NCCL destroy_process_group）。
# C++ 日志级别必须在导入 torch 之前设置（utils.distributed 会导入 torch）才生效。
warnings.filterwarnings("ignore")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("C10_LOG_LEVEL", "ERROR")

import yaml

from utils.distributed import init_distributed

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debugmode",
    type=bool,
    default=False,
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to \
    debug with checkpointing.",
)
parser.add_argument(
    "--set",
    dest="overrides",
    nargs="+",
    default=[],
    metavar="KEY=VALUE",
    help="Override merged config values, e.g. \
    --set folder=/path/to/out meta.read_checkpoint=/path/to/ckpt.pt . \
    Dotted keys index nested mappings; values are parsed as YAML scalars \
    (int/float/bool/list/dict) and fall back to strings. Overrides are applied \
    after the YAML merge, so they beat every fragment and the entry config. \
    Only keys already present in the merged config are guaranteed to take \
    effect (see CONFIGS/README.md 10.2). Best passed last, for readability.",
)


def app_main(app, args):
    """Load and run the configured pre-training application."""
    logger.info(f"Running pre-training of app: {app}")
    return importlib.import_module(f"app.{app}.train").main(
        args=args
    )


def choose_free_port():
    """Reserve an ephemeral local TCP port number for this launch."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _merge_config(base, override):
    """Recursively merge configuration mappings, with ``override`` winning."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_yaml_fragment(yaml_path, config_path):
    """Resolve a ``yamls`` fragment with portable repository-relative support.

    Resolution order is intentionally deterministic:

    - absolute paths are used unchanged;
    - paths beginning with ``CONFIGS/`` are relative to the repository root;
    - all other relative paths are relative to the entry YAML directory;
    - the other relative interpretation is retained as a compatibility fallback.
    """
    raw = str(yaml_path)
    fragment_path = Path(raw).expanduser()
    if fragment_path.is_absolute():
        candidates = [fragment_path]
    else:
        repository_relative = PROJECT_ROOT / fragment_path
        config_relative = config_path.parent / fragment_path
        if fragment_path.parts and fragment_path.parts[0] == "CONFIGS":
            candidates = [repository_relative, config_relative]
        else:
            candidates = [config_relative, repository_relative]

    unique_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(
        f"YAML fragment {raw!r} referenced by {config_path} was not found; "
        f"tried: {attempted}"
    )


def load_config(fname):
    """Load a YAML config and its ordered ``yamls`` fragments, if present.

    ``CONFIGS/...`` fragment paths are repository-root-relative. Other
    relative paths remain relative to the entry YAML for compatibility.
    """
    config_path = Path(fname).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as y_file:
        config = yaml.load(y_file, Loader=yaml.FullLoader) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    yaml_paths = config.pop("yamls", None)
    if yaml_paths is None:
        return config
    if isinstance(yaml_paths, dict):
        yaml_paths = yaml_paths.values()
    elif isinstance(yaml_paths, (str, Path)):
        yaml_paths = [yaml_paths]
    elif not isinstance(yaml_paths, list):
        raise ValueError("yamls must be a mapping, list, or YAML path")

    params = {}
    for yaml_path in yaml_paths:
        fragment_path = _resolve_yaml_fragment(yaml_path, config_path)
        with fragment_path.open("r", encoding="utf-8") as y_file:
            fragment = yaml.load(y_file, Loader=yaml.FullLoader) or {}
        if not isinstance(fragment, dict):
            raise ValueError(f"Configuration fragment must be a YAML mapping: {fragment_path}")
        params = _merge_config(params, fragment)

    # The entry config is last so run-specific values override shared fragments.
    return _merge_config(params, config)


def _coerce_override_value(raw):
    """Parse a --set value as a YAML scalar/container, keeping plain strings as-is."""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(parsed, (bool, int, float, list, dict)):
        return parsed
    return raw


def apply_overrides(params, overrides):
    """Apply ``KEY=VALUE`` pairs to a merged config; dotted keys index nested mappings.

    Runs after ``load_config`` so command-line values win over every YAML file.
    ``params`` is mutated in place, which keeps the ``params-{app}.yaml`` snapshot
    written by ``process_main`` faithful to what actually ran.
    """
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"--set expects KEY=VALUE, but got {item!r}")
        parts = [part for part in key.strip().split(".") if part]
        if not parts:
            raise ValueError(f"--set expects a non-empty key, but got {item!r}")
        node = params
        walked = []
        for part in parts[:-1]:
            walked.append(part)
            child = node.get(part)
            if not isinstance(child, dict):
                raise KeyError(
                    f"--set {key.strip()}: {'.'.join(walked)} is not a config mapping"
                )
            node = child
        node[parts[-1]] = _coerce_override_value(raw)
    return params


def process_main(rank, fname, world_size, devices, local_rank=None, overrides=None):
    import os

    # Each rank must see exactly ONE GPU, because the trainers hard-code
    # cuda:0 / device_ids=[0] and rely on CUDA_VISIBLE_DEVICES remapping.
    # local_rank comes from torchrun (LOCAL_RANK env); under the mp.Process
    # launcher it defaults to the spawn rank.
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if devices and len(devices) == world_size:
        physical = str(devices[local_rank]).split(":")[-1]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        visible = [v for v in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if v.strip()]
        physical = visible[local_rank] if local_rank < len(visible) else str(local_rank)
    else:
        physical = str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical

    import logging

    from utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = load_config(fname)
    if overrides:
        apply_overrides(params, overrides)
        logger.info("applied --set overrides: %s", " ".join(overrides))
    logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, f"params-{params['app']}.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    try:
        app_main(params["app"], args=params)
    finally:
        # 干净退出分布式进程组，消除 NCCL 的 destroy_process_group 警告。
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    args = parser.parse_args()
    import os

    # torchrun mode: torchrun has already spawned one process per rank and set
    # RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT, so run inline instead
    # of forking our own child processes.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        logger.info(
            "torchrun mode (rank=%s world=%s local=%s)",
            os.environ["RANK"],
            os.environ["WORLD_SIZE"],
            os.environ.get("LOCAL_RANK", "0"),
        )
        process_main(
            rank=int(os.environ["RANK"]),
            fname=args.fname,
            world_size=int(os.environ["WORLD_SIZE"]),
            devices=args.devices,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            overrides=args.overrides,
        )
    else:
        # Manual mp.Process launcher: pick a master port once in the parent so
        # every spawned rank joins the same process group.  Do not use a fixed
        # default: multiple local jobs may coexist.
        master_port = os.environ.get("MASTER_PORT") or str(choose_free_port())
        os.environ["MASTER_PORT"] = master_port
        logger.info("Using local distributed rendezvous port %s", master_port)
        if args.debugmode:
            process_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"],
                         overrides=args.overrides)
        else:
            num_gpus = len(args.devices)
            mp.set_start_method("spawn")
            for rank in range(num_gpus):
                mp.Process(target=process_main,
                           args=(rank, args.fname, num_gpus, args.devices, None, args.overrides)).start()

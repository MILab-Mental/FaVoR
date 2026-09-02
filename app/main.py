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


def load_config(fname):
    """Load a YAML config and its ordered ``yamls`` fragments, if present."""
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
        fragment_path = Path(yaml_path).expanduser()
        if not fragment_path.is_absolute():
            fragment_path = config_path.parent / fragment_path
        fragment_path = fragment_path.resolve()
        with fragment_path.open("r", encoding="utf-8") as y_file:
            fragment = yaml.load(y_file, Loader=yaml.FullLoader) or {}
        if not isinstance(fragment, dict):
            raise ValueError(f"Configuration fragment must be a YAML mapping: {fragment_path}")
        params = _merge_config(params, fragment)

    # The entry config is last so run-specific values override shared fragments.
    return _merge_config(params, config)


def process_main(rank, fname, world_size, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

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
    # Select once in the parent so every spawned rank joins the same process
    # group.  Do not use a fixed default: multiple local jobs may coexist.
    import os

    master_port = os.environ.get("MASTER_PORT") or str(choose_free_port())
    os.environ["MASTER_PORT"] = master_port
    logger.info("Using local distributed rendezvous port %s", master_port)
    if args.debugmode:
        process_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
    else:
        num_gpus = len(args.devices)
        mp.set_start_method("spawn")
        for rank in range(num_gpus):
            mp.Process(target=process_main, args=(rank, args.fname, num_gpus, args.devices)).start()

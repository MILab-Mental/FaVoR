"""Pinned official checkouts, scoped imports, and narrowly defined API compatibility.

Checkouts live beside weights, not inside FAVOR's `models`/`utils` packages.
"""
import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
import yaml

from BASELINES.common.runtime import WEIGHTS_ROOT, path

LOCK = WEIGHTS_ROOT / 'sources.yaml'


def checkout(name):
    entry = yaml.safe_load(LOCK.read_text())[name]
    root = path(entry.get('path', WEIGHTS_ROOT / 'sources' / name))
    if not root.is_dir():
        raise FileNotFoundError(f'{root}: run python -m BASELINES.common.official --fetch {name}')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != entry['revision']:
        raise ValueError(f'{name} source revision {actual} != pinned {entry["revision"]}')
    return root


@contextlib.contextmanager
def official(name, subdir='.', keep_packages=()):
    root = checkout(name) / subdir
    roots = {p.stem for p in root.glob('*.py')} | {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')}
    roots.difference_update(keep_packages)
    saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in roots}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(root))
    # These repos use timm 0.3's qk_scale=None. Modern timm uses the same
    # head_dim**-0.5 default; non-default legacy scales are rejected.
    import timm.models.vision_transformer as vit
    old_block, old_attention, old_patch = vit.Block, vit.Attention, vit.PatchEmbed
    def compatible(base):
        class Legacy(base):
            def __init__(self, *args, qk_scale=None, **kwargs):
                if qk_scale is not None:
                    raise ValueError('Non-default qk_scale requires the original timm environment')
                if 'drop' in kwargs and base is old_block:
                    kwargs['proj_drop'] = kwargs.pop('drop')
                super().__init__(*args, **kwargs)
        return Legacy
    vit.Block, vit.Attention = compatible(old_block), compatible(old_attention)
    old_six = sys.modules.get('torch._six')
    six = types.ModuleType('torch._six')
    six.inf = float('inf')
    sys.modules['torch._six'] = six
    had_float = hasattr(np, 'float')
    if not had_float:
        np.float = float
    old_home = os.environ.get('TORCH_HOME')
    def import_module(module):
        # Official module imports may seed global RNGs. Preserve the run seed.
        python_rng, numpy_rng, torch_rng = __import__('random').getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None
        result = importlib.import_module(module)
        # The original VideoMAE defaults to reentrant checkpointing, which is
        # incompatible with DDP unused-parameter traversal. Preserve its math
        # while selecting PyTorch's supported non-reentrant implementation.
        if name=='VideoMAE' and module in ('modeling_finetune','modeling_pretrain'):
            from functools import partial
            from torch.utils.checkpoint import checkpoint
            result.checkpoint = types.SimpleNamespace(checkpoint=partial(checkpoint,use_reentrant=False))
        __import__('random').setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
        return result
    try:
        yield import_module
    finally:
        vit.Block, vit.Attention, vit.PatchEmbed = old_block, old_attention, old_patch
        if not had_float:
            del np.float
        if old_six is None:
            sys.modules.pop('torch._six', None)
        else:
            sys.modules['torch._six'] = old_six
        if old_home is None:
            os.environ.pop('TORCH_HOME', None)
        else:
            os.environ['TORCH_HOME'] = old_home
        sys.path.remove(str(root))
        for key in list(sys.modules):
            if key.split('.')[0] in roots:
                del sys.modules[key]
        sys.modules.update(saved)


def fetch(names):
    entries = yaml.safe_load(LOCK.read_text())
    for name in names:
        entry = entries[name]
        root = path(entry.get('path', WEIGHTS_ROOT / 'sources' / name))
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', entry['url'], str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'checkout', entry['revision']], check=True)
        checkout(name)
        print(name, entry['revision'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', nargs='+', required=True)
    fetch(parser.parse_args().fetch)

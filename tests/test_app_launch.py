"""A failed rank must exit without waiting on peers during collective cleanup."""
from unittest.mock import Mock

import pytest
import torch.distributed as dist

from app import main
from utils import logging as log_utils


@pytest.mark.parametrize('fails', [False, True])
def test_process_main_cleans_up_only_successful_runs(monkeypatch, fails):
    logger = Mock()
    monkeypatch.setattr(log_utils, 'get_logger', lambda **kwargs: logger)
    monkeypatch.setattr(main, 'load_config', lambda path: {'app': 'finetune_va'})
    monkeypatch.setattr(main, 'init_distributed', lambda **kwargs: (2, 1))
    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    cleanup = Mock()
    monkeypatch.setattr(dist, 'destroy_process_group', cleanup)
    def run(*args, **kwargs):
        if fails:
            raise RuntimeError('missing media')
    monkeypatch.setattr(main, 'app_main', run)
    if fails:
        with pytest.raises(RuntimeError, match='missing media'):
            main.process_main(1, 'unused.yaml', 2, ['cuda:0', 'cuda:1'])
        cleanup.assert_not_called()
        logger.exception.assert_called_once()
    else:
        main.process_main(1, 'unused.yaml', 2, ['cuda:0', 'cuda:1'])
        cleanup.assert_called_once()
        logger.exception.assert_not_called()

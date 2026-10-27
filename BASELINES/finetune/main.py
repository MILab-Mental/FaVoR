from BASELINES.common.runtime import arguments
from BASELINES.finetune.trainer import main

if __name__ == '__main__':
    args, cfg = arguments('finetune')
    main(cfg, args)

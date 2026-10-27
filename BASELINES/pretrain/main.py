from BASELINES.common.runtime import arguments
from BASELINES.pretrain.trainer import main

if __name__ == '__main__':
    args, cfg = arguments('pretrain')
    main(cfg, args)

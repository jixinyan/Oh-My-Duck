"""Run the official RSL trainer with project logging settings on every rank."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from common.tracking import settings


def main():
    defaults = settings()
    os.environ['WANDB_MODE'] = defaults['mode']
    # mjlab's native torchrunx launch copies CUDA/Torch variables, but not W&B.
    # Extend its public environment allowlist; PPO, task and launcher are original.
    import torchrunx
    torchrunx.DEFAULT_ENV_VARS_FOR_COPY += ('WANDB_MODE', 'WANDB_PROJECT', 'WANDB_DIR',
        'WANDB_RUN_GROUP', 'WANDB_RUN_ID', 'WANDB_TAGS', 'WANDB_ENTITY', 'WANDB_USERNAME')
    runpy.run_module('mjlab.scripts.train', run_name='__main__')


if __name__ == '__main__':
    main()

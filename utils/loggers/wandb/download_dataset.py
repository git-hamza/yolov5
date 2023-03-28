import argparse

from wandb_utils import WandbLogger

from utils.general import LOGGER
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory


def create_dataset_artifact(opt):
    logger = WandbLogger(opt, None, job_type='Dataset Retrieval')  # TODO: return value unused
    if not logger.wandb:
        LOGGER.info("install wandb using `pip install wandb` to log the dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/training_data_wandb.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='firefly-ball-detector', help='name of W&B Project')
    parser.add_argument('--entity', default='firefly-balldetector', help='W&B entity')
    parser.add_argument('--name', type=str, default='get dataset', help='name of W&B run')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    opt = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job
    opt.single_cls = False

    create_dataset_artifact(opt)

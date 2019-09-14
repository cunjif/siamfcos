import logging
from sfc.utils.distributed import get_word_size
from sfc.data.dataset import TrkDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from sfc.config.defaults import cfg


logger = logging.getLogger('global')


def build_data_loader():
    logger.info("build train dataset")
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_word_size() > 1:
        train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            sampler=train_sampler
        )
    return train_loader
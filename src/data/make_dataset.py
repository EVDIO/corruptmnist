# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset


class CorruptMnist(Dataset):
    def __init__(self, train, input_filepath=None, output_filepath=None):

        if train:
            content = []
            for i in range(8):
                content.append(np.load(os.path.join(input_filepath,
                                                    'train_{}.npz'.format(i)), allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content]))
            data = data.reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load(os.path.join(input_filepath, 'test.npz'),
                              allow_pickle=True)
            data = torch.tensor(content['images'])
            data = data.reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])

        self.data = data
        self.targets = targets

    def __len__(self):
        return self.targets.numel()

    def __getitem__(self, idx):

        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train_data = CorruptMnist(True, input_filepath=input_filepath,
                              output_filepath=output_filepath)
    test_data = CorruptMnist(False, input_filepath=input_filepath,
                             output_filepath=output_filepath)
    print(type(train_data))
    torch.save(train_data, output_filepath + "/train_set.pt")
    torch.save(test_data, output_filepath + "/test_set.pt")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

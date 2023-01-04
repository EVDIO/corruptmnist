# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import torch
import os
from torchvision import transforms

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # initialize lists to hold the training and test data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # load the training data
    for i in range(8):
        npz_file = np.load(os.path.join(input_filepath, 'train_{}.npz'.format(i)))
        data = npz_file['images']
        labels = npz_file['labels']
        train_data.append(data)
        train_labels.append(labels)
    
    # load the test data
    npz_file = np.load(os.path.join(input_filepath, 'test.npz'))
    test_data = npz_file['images']
    test_labels = npz_file['labels']

    # concatenate the training data and labels into single arrays
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    
    # apply normalization transform to the data
    normalize_transform = transforms.Normalize((0.5,), (0.5,))
    train_data = normalize_transform(torch.from_numpy(train_data))
    test_data = normalize_transform(torch.from_numpy(test_data))
    
    # convert labels to PyTorch tensors
    train_labels = torch.from_numpy(train_labels)
    test_labels = torch.from_numpy(test_labels)

    # save the normalized data and labels to the save directory
    torch.save((train_data, train_labels), os.path.join(output_filepath, 'train_set.pt'))
    torch.save((test_data, test_labels), os.path.join(output_filepath, 'test_set.pt'))

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

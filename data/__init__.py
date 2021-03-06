"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    if phase == 'train':
        num_workers = dataset_opt['num_workers']
        batch_size = dataset_opt['batch_size']
        shuffle = dataset_opt['use_shuffle']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, drop_last=True,
                                           pin_memory=False)
    elif phase == 'val':
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt, phase):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'REDS':
        from data.REDS_dataset import REDSDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt, phase)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
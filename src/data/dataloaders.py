
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.preprocessing import minmax_scale
from torchvision import transforms

from src.data.Audio.CustomTensor import CustomTensorDataset
from src.data.Audio.TorchLoader import TorchLoader


def dataloader(args, log):
    audio = TorchLoader(path_source=args.source_folder, log=log, sample_rate=args.sample_rate)
    x_train, train_y, x_val, val_y, x_test, test_y = audio.load_data(
        source_path=args.source_folder).load_files().save_data(
        destination_path=args.source_folder).get_data()

    x_train = minmax_scale(x_train)
    x_val = minmax_scale(x_val)
    x_test = minmax_scale(x_test)

    x_train = np.reshape(x_train, (-1, args.in_channels, args.x, args.y))
    x_val = np.reshape(x_val, (-1, args.in_channels, args.x, args.y))
    x_test = np.reshape(x_test, (-1, args.in_channels, args.x, args.y))

    class PadNumpy(object):

        def __init__(self, type_padding):
            self._type_padding = type_padding

        def __call__(self, sample):
            real_shape = sample.shape
            if real_shape[2] == 49:
                sample = np.pad(sample, ((0, 0), (0, 0), (8, 7)), self._type_padding)
            if real_shape[2] == 40:
                sample = np.pad(sample, ((0, 0), (0, 0), (12, 12)), self._type_padding)
            if real_shape[1] == 25:
                sample = np.pad(sample, ((0, 0), (20, 19), (0, 0)), self._type_padding)
            if real_shape[2] == 99:
                sample = np.pad(sample, ((0, 0), (0, 0), (15, 14)), self._type_padding)
            if real_shape[2] == 80:
                sample = np.pad(sample, ((0, 0), (0, 0), (24, 24)), self._type_padding)
            if real_shape[1] == 50:
                sample = np.pad(sample, ((0, 0), (39, 39), (0, 0)), self._type_padding)
            return sample

    data_transform_pad = transforms.Compose([
        PadNumpy("constant"),
    ])

    train_dataset = CustomTensorDataset(torch.from_numpy(x_train), torch.from_numpy(x_train),
                                        transform=data_transform_pad)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    validation_dataset = CustomTensorDataset(torch.from_numpy(x_val), torch.from_numpy(x_val),
                                             transform=data_transform_pad)
    val_loader = data_utils.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    test_dataset = CustomTensorDataset(torch.from_numpy(x_test), torch.from_numpy(x_test), transform=data_transform_pad)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

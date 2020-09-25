"""
TLSTM. Turing Learning system to generate trajectories
Copyright (C) 2018  Alessandro Zonta (a.zonta@vu.nl)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from src.data.Audio.DataLoader import LoadData

import glob
import os
from os import path
import torch.nn.functional as F

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


class TorchLoader(LoadData):
    """
    laoding data using torch functions
    """

    def __init__(self, path_source, log, sample_rate):
        super().__init__(path_source, log, sample_rate)

    def load_files(self):
        """
        load file from disk
        :return: self
        """
        # train set
        train_set_folder = "{}/train_set".format(self._path_source)
        if not torch.is_tensor(self._x_train_set):
            self._x_train_set, self._y_train_set = self._load_audio_and_files(folder=train_set_folder,
                                                                              new_sample_rate=self._sample_rate)
        # val set
        val_set_folder = "{}/val_set".format(self._path_source)
        if not torch.is_tensor(self._x_val_set):
            self._x_val_set, self._y_val_set = self._load_audio_and_files(folder=val_set_folder,
                                                                          new_sample_rate=self._sample_rate)
        # test set
        test_set_folder = "{}/test_set".format(self._path_source)
        if not torch.is_tensor(self._x_test_set):
            self._x_test_set, self._y_test_set = self._load_audio_and_files(folder=test_set_folder,
                                                                            new_sample_rate=self._sample_rate)
        self._log.info("Data read from source")
        return self

    def save_data(self, destination_path):
        """
        save loaded dataset for fast loading
        :param destination_path: path where to save data
        :return: self
        """
        saved = False
        if not path.exists("{}/training_set_t_{}.npz".format(destination_path, self._sample_rate)):
            np.savez("{}/training_set_t_{}".format(destination_path, self._sample_rate), self._x_train_set.numpy(),
                     self._y_train_set)
            saved = True
        if not path.exists("{}/test_set_t_{}.npz".format(destination_path, self._sample_rate)):
            np.savez("{}/test_set_t_{}".format(destination_path, self._sample_rate), self._x_test_set.numpy(),
                     self._y_test_set)
            saved = True
        if not path.exists("{}/validation_set_t_{}.npz".format(destination_path, self._sample_rate)):
            np.savez("{}/validation_set_t_{}".format(destination_path, self._sample_rate), self._x_val_set.numpy(),
                     self._y_val_set)
            saved = True
        if saved:
            self._log.info("Data saved")
        return self

    def load_data(self, source_path):
        """
        load data from memory if file exist
        :param source_path: path where data is stored
        :return: self
        """
        if path.exists("{}/test_set_t_{}.npz".format(source_path, self._sample_rate)):
            tr = np.load("{}/test_set_t_{}.npz".format(source_path, self._sample_rate), allow_pickle=True)
            self._x_train_set = torch.from_numpy(tr["arr_0"])
            self._x_train_set = self._x_train_set.view(self._x_train_set.size()[0], -1)
            self._y_train_set = tr["arr_1"]
        if path.exists("{}/test_set_t_{}.npz".format(source_path, self._sample_rate)):
            tr = np.load("{}/test_set_t_{}.npz".format(source_path, self._sample_rate), allow_pickle=True)
            self._x_test_set = torch.from_numpy(tr["arr_0"])
            self._x_test_set = self._x_test_set.view(self._x_test_set.size()[0], -1)
            self._y_test_set = tr["arr_1"]
        if path.exists("{}/test_set_t_{}.npz".format(source_path, self._sample_rate)):
            tr = np.load("{}/test_set_t_{}.npz".format(source_path, self._sample_rate), allow_pickle=True)
            self._x_val_set = torch.from_numpy(tr["arr_0"])
            self._x_val_set = self._x_val_set.view(self._x_val_set.size()[0], -1)
            self._y_val_set = tr["arr_1"]
        return self

    @staticmethod
    def _load_audio_and_files(folder, new_sample_rate):
        """
        finds all the files .wav
        if stereo signal -> transform to mono
        sample the data following the indicated sampling rate
        normalise the data to the interval [-1,1]
        pads the data to the max_dimensions
        :param folder: path files wav
        :param new_sample_rate: sampling rate used to read the data
        :return: audio signals and class
        """

        def _normalize(tensor):
            # Subtract the mean, and scale to the interval [-1,1]
            tensor_minusmean = tensor - tensor.mean()
            return tensor_minusmean / tensor_minusmean.abs().max()

        folders = os.listdir(folder)
        max_dimension = 0
        data_list = []
        data_list_y = []
        for i in range(len(folders)):
            audio_files = glob.glob("{}/{}/*.wav".format(folder, folders[i]))
            for file in tqdm(audio_files, desc="loading {}, {}".format(folder, folders[i])):
                data_list_y.append(i)
                waveform, sample_rate = torchaudio.load(file)
                # if stereo signal make mono
                if waveform.size()[0] == 2:
                    waveform = (waveform.sum(axis=0) / 2).view(1, -1)
                waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
                # Let's normalize to the full interval [-1,1]
                waveform = _normalize(waveform)
                max_dimension = max(max_dimension, waveform.size()[1])
                data_list.append(waveform)
        current_set = []
        for el in data_list:
            el_here = F.pad(el, (0, max_dimension - el.size()[1]))
            current_set.append(el_here)
        x_set = torch.stack(current_set)
        return x_set, data_list_y

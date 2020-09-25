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
from abc import abstractmethod
from os import path

import numpy as np

"""
class coming from different project, supports different way of loading data
"""


class LoadData(object):
    def __init__(self, path_source, log, sample_rate):
        self._path_source = path_source
        self._x_train_set = []
        self._y_train_set = []
        self._x_val_set = []
        self._y_val_set = []
        self._x_test_set = []
        self._y_test_set = []
        self._log = log
        self._sample_rate = sample_rate

    def get_data(self):
        """
        Get training, test, and validation sets
        :return:
        """
        return self._x_train_set, self._y_train_set, self._x_val_set, self._y_val_set, self._x_test_set, self._y_test_set

    def save_data(self, destination_path):
        """
        save loaded dataset for fast loading
        :param destination_path: path where to save data
        :return: self
        """
        saved = False
        if not path.exists("{}/training_set_{}.npz".format(destination_path, self._sample_rate)):
            np.savez("{}/training_set_{}".format(destination_path, self._sample_rate), self._x_train_set,
                     self._y_train_set)
            saved = True
        if not path.exists("{}/test_set_{}.npz".format(destination_path, self._sample_rate)):
            np.savez("{}/test_set_{}".format(destination_path, self._sample_rate), self._x_test_set, self._y_test_set)
            saved = True
        if not path.exists("{}/validation_set_{}.npz".format(destination_path, self._sample_rate)):
            np.savez("{}/validation_set_{}".format(destination_path, self._sample_rate), self._x_val_set,
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
        if path.exists("{}/training_set_{}.npz".format(source_path, self._sample_rate)):
            tr = np.load("{}/training_set_{}.npz".format(source_path, self._sample_rate), allow_pickle=True)
            self._x_train_set = tr["arr_0"]
            self._y_train_set = tr["arr_1"]
        if path.exists("{}/test_set_{}.npz".format(source_path, self._sample_rate)):
            tr = np.load("{}/test_set_{}.npz".format(source_path, self._sample_rate), allow_pickle=True)
            self._x_test_set = tr["arr_0"]
            self._y_test_set = tr["arr_1"]
        if path.exists("{}/validation_set_{}.npz".format(source_path, self._sample_rate)):
            tr = np.load("{}/validation_set_{}.npz".format(source_path, self._sample_rate), allow_pickle=True)
            self._x_val_set = tr["arr_0"]
            self._y_val_set = tr["arr_1"]
        return self

    @abstractmethod
    def load_files(self):
        """
        abstract method
        loading of the single files
        :return:
        """
        return

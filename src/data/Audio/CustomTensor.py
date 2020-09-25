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
from torch.utils.data.dataset import Dataset


class CustomTensorDataset(Dataset):
    """Dataset wrapping tensors.
    Each sample will be retrieved by indexing tensors along the first dimension.
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        from PIL import Image

        X, y = self.tensors
        X_i, y_i, = X[index], y[index]

        if self.transform:
            X_i = self.transform(X_i.numpy())
        return X_i, y_i

    def __len__(self):
        return self.tensors[0].size(0)
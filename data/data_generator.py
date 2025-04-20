import torch
from skimage.io import imread
from torch.utils import data


class SegmentationDataSet(data.Dataset):
    
    ''' used for picking input-target pairs and preforms transformations on the data '''

    def __init__(self, inputs: list, targets: list, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # getting sample
        sample_input = self.inputs[index]
        sample_target = self.targets[index]

        # load input and target
        x, y = imread(sample_input), imread(sample_target)

        # preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

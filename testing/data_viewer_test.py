"""
run the following commands in shell

ipython
run testing\data_viewer_test.py

"""


from data import DatasetViewer
import numpy as np
from data import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01
from data import SegmentationDataSet1
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib

root = pathlib.Path.cwd() / 'Carvana'


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / 'Input')
targets = get_filenames_of_path(root / 'Target')

# training transformations and augmentations
transforms = ComposeDouble([
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])


random_seed = 42

# split dataset into training set and validation set
train_size = 0.8 

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, targets, train_size=train_size, random_state=random_seed, shuffle=True
)

# create training dataset and dataloader
train_dataset = SegmentationDataSet1(inputs=train_inputs,
                                      targets=train_targets,
                                      transform=transforms)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=2,
                              shuffle=True)

# create validation dataset and dataloader
val_dataset = SegmentationDataSet1(inputs=val_inputs,
                                    targets=val_targets,
                                    transform=transforms)

val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=2,
                                shuffle=False)

dataset_viewer_training = DatasetViewer(train_dataset) 
dataset_viewer_training.napari()

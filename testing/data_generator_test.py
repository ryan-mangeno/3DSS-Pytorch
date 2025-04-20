from data import SegmentationDataSet
from data import ComposeDouble
from data import FunctionWrapperDouble
from data import create_dense_target
from data import normalize_01
import numpy as np
from torch.utils.data import DataLoader
from skimage.transform import resize

inputs = [r'data\input\pic_01.png', r'data\input\pic_02.png']
targets = [r'data\target\pic_01.png', r'data\target\pic_02.png']

training_dataset = SegmentationDataSet(inputs=inputs,
                                       targets=targets,
                                       transform=None)

training_dataloader = DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)


x = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
y = np.random.randint(10, 15, size=(128, 128), dtype=np.uint8)


transforms = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(128, 128, 3)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          output_shape=(128, 128),
                          order=0,
                          anti_aliasing=False,
                          preserve_range=True),
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])


tar = np.array([
    [5, 5, 9],
    [9, 5, 15]
])

print(f'Dense Array: {create_dense_target(tar)}')


x_t, y_t = transforms(x, y)

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'x_t: shape: {x_t.shape}  type: {x_t.dtype}')
print(f'x_t = min: {x_t.min()}; max: {x_t.max()}')

print(f'y = shape: {y.shape}; class: {np.unique(y)}')
print(f'y_t = shape: {y_t.shape}; class: {np.unique(y_t)}')
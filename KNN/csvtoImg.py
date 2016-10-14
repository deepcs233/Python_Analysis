import pandas as pd
from PIL import Image
import numpy

TRAIN_NUM = 42000

data = pd.read_csv('train.csv')
train_data = data.values[0:TRAIN_NUM,1:]

pixels=train_data[10].reshape(28,28).astype('uint8')
print pixels.shape
img=Image.fromarray(pixels)
img=img.convert('RGB')
img.save('Pixel_number.jpeg')

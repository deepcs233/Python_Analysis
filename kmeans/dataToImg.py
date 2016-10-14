import numpy
from PIL import Image
import json

with open('numbers_rocord.txt','r') as f:
	t=f.read()
	d=json.loads(t)

for k in range(len(d)):
    pixels=numpy.array(d[k]).reshape(28,28)
    pixel_strong=numpy.arange(28*28*16).reshape(28*4,28*4)
    for i in range(len(pixel_strong)):
        for j in range(len(pixel_strong[0])):
            
            pixel_strong[i][j]=((pixels[i/4,j/4]))
            if pixel_strong[i][j]>80:
                pixel_strong[i][j]=255
            else:
                pixel_strong[i][j]=0
    print pixel_strong.shape
    img=Image.fromarray(pixel_strong)
    img=img.convert('RGB')
    img.save('Pixel'+str(k)+'.jpeg')


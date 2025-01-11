import numpy as np
from PIL import Image

file = Image.open('dog.png')
file.show()

image = np.array(file)

print(image.shape)

# 直接访问像素点颜色
print(image[100, 100])

red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]

Image.fromarray(red).show()
Image.fromarray(green).show()
Image.fromarray(blue).show()

# 直接混合图片
# conbine = a1 * 0.2 + a2 * 0.8
# 显示图片需要换为unit8
# combine.astype(np.unit8)


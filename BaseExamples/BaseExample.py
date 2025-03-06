from PIL import Image
import numpy as np

arr = np.zeros((600, 800, 3), dtype=np.uint8)

# full black image
for x in range(600):
    for y in range(800):
        arr[x][y] = 0

image = Image.fromarray(arr)
image.save("black.png")

# full green image
for x in range(600):
    for y in range(800):
        arr[x][y] = [0, 255, 0]

image = Image.fromarray(arr)
image.save("green.png")

# gradient image
for x in range(600):
    for y in range(800):
        tmp = (x + y) % 256
        arr[x][y] = [tmp, 0, 0]

image = Image.fromarray(arr)
image.save("gradient.png")

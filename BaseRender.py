from Parser import *
from PIL import Image, ImageOps
import numpy as np
import math
import random


def calculate_bounding_rectangle(x0, x1, x2, y0, y1, y2, width, height):
    xmin = math.floor(max(min(x0, x1, x2), 0))
    xmax = math.ceil(min(max(x0, x1, x2), width))
    ymin = math.floor(max(min(y0, y1, y2), 0))
    ymax = math.ceil(min(max(y0, y1, y2), height))
    return xmin, xmax, ymin, ymax


def calculate_barycentric_coordinates(x, y, x0, x1, x2, y0, y1, y2):
    denom = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if denom == 0:
        return None, None, None

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


if __name__ == "__main__":
    vertices, faces = parse_obj("model.obj")
    for i in range(len(vertices)):
        vertices[i] = [1000 + 10000 * vertices[i][0], 500 + 10000 * vertices[i][1], vertices[i][2]]

    width = 2000
    height = 2000
    color = [0, 255, 0]
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for el in faces:
        v1, v2, v3 = (vertices[x - 1] for x in el)
        color = random.randint(0, 255)
        xmin, xmax, ymin, ymax = calculate_bounding_rectangle(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], width, height)

        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                l1, l2, l3 = calculate_barycentric_coordinates(x, y, v1[0], v2[0], v3[0], v1[1], v2[1], v3[1])
                if (l1 >= 0) & (l2 >= 0) & (l3 >= 0):
                    arr[y, x] = color

    img = Image.fromarray(arr)
    img = ImageOps.flip(img)
    img.save("result.png")

from Parser import *
from PIL import Image, ImageOps
import numpy as np
import math
import random


def calculate_bounding_rect(x0, x1, x2, y0, y1, y2, width, height):
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


def calculate_perpendicular(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))


def calculate_cos(n):
    l = np.array([0, 0, -1])
    dot_product = np.dot(n, l)
    norm_n = np.linalg.norm(n)
    norm_l = np.linalg.norm(l)
    return dot_product / (norm_n * norm_l)


if __name__ == "__main__":
    vertices, faces = parse_obj("model.obj")
    for i in range(len(vertices)):
        vertices[i] = [1000 + 10000 * vertices[i][0], 500 + 10000 * vertices[i][1], 10000 * vertices[i][2]]

    width = 2000
    height = 2000
    color = [0, 255, 0]
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), -np.inf, dtype=np.float64)

    # заяц смотрит на нас потому что я поменял вектор взгляда на [0, 0, -1]
    # и z_buffer теперь заполняется -inf, а меняется если текущий z > z_buffer
    for el in faces:
        v1, v2, v3 = (vertices[x - 1] for x in el)

        n = calculate_perpendicular(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2])
        cos = calculate_cos(n)

        if cos < 0:
            xmin, xmax, ymin, ymax = calculate_bounding_rect(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], width, height)
            color = -255 * cos
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    l1, l2, l3 = calculate_barycentric_coordinates(x, y, v1[0], v2[0], v3[0], v1[1], v2[1], v3[1])
                    if (l1 >= 0) & (l2 >= 0) & (l3 >= 0):
                        z = l1 * v1[2] + l2 * v2[2] + l3 * v3[2]
                        if z > z_buffer[y, x]:
                            arr[y, x] = color
                            z_buffer[y, x] = z

    img = Image.fromarray(arr)
    img = ImageOps.flip(img)
    img.save("result.png")

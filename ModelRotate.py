from Parser import *
from PIL import Image, ImageOps
import numpy as np
import math


def calculate_bounding_rect(x0, x1, x2, y0, y1, y2, width, height):
    xmin = math.floor(max(min(x0, x1, x2), 0))
    xmax = math.ceil(min(max(x0, x1, x2), width))
    ymin = math.floor(max(min(y0, y1, y2), 0))
    ymax = math.ceil(min(max(y0, y1, y2), height))
    return xmin, xmax, ymin, ymax


def calculate_barycentric_coordinates(x, y, x0, x1, x2, y0, y1, y2):
    if (denom := (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) == 0:
        return None, None, None
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def calculate_perpendicular(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    return np.cross(np.array([x1 - x2, y1 - y2, z1 - z2]), np.array([x1 - x0, y1 - y0, z1 - z0]))


def calculate_cos(n):
    l = np.array([0, 0, 1])
    dot_product = np.dot(n, l)
    norm_n = np.linalg.norm(n)
    norm_l = np.linalg.norm(l)
    return dot_product / (norm_n * norm_l)


def get_rotation_matrix(alpha, beta, gamma):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)]
    ])

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    Rz = np.array([
        [np.cos(gamma), np.sin(gamma), 0],
        [-np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def transform_vertices(vertices, alpha, beta, gamma, shift, trans):
    R = get_rotation_matrix(alpha, beta, gamma)
    # поворот -> shift модели -> скалирование -> trans
    return np.array([10000 * (R @ np.array(el) + np.array(shift)) + trans for el in vertices])


if __name__ == "__main__":
    vertices, faces = parse_obj("model.obj")

    # Определить min и max координату для выставления сдвига модели (0.1 и 0.05)
    # print(min([el[0] for el in vertices]))
    # print(max([el[0] for el in vertices]))

    width = 2000
    height = 2000
    color = [0, 255, 0]
    trans = [0, -0.05, 0]
    shift = [width // 2, height // 2, 0]

    vertices = transform_vertices(vertices, 0, 0, 0, trans, shift)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float64)

    for el in faces:
        v1, v2, v3 = (vertices[x - 1] for x in el)

        n = calculate_perpendicular(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2])
        cos = calculate_cos(n)

        if cos < 0:
            xmin, xmax, ymin, ymax = calculate_bounding_rect(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], width, height)
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    l1, l2, l3 = calculate_barycentric_coordinates(x, y, v1[0], v2[0], v3[0], v1[1], v2[1], v3[1])
                    if (l1 >= 0) & (l2 >= 0) & (l3 >= 0):
                        z = l1 * v1[2] + l2 * v2[2] + l3 * v3[2]
                        if z < z_buffer[y, x]:
                            arr[y, x] = [c * -cos for c in color]
                            z_buffer[y, x] = z

    img = Image.fromarray(arr)
    img = ImageOps.flip(img)
    img.save("result.png")

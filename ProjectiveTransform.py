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


def transform_vertices(v, alpha, beta, gamma, sh_z, sc, tr):
    R = get_rotation_matrix(alpha, beta, gamma)
    v = np.array(v) + [0, -0.05, 0]
    v = np.array([R @ el for el in v])
    v += [0, 0, sh_z]
    v[:, :2] = np.array([[el[0] / el[2], el[1] / el[2]] for el in v])
    v = v * sc + tr
    return v


if __name__ == "__main__":
    vertices, faces = parse_obj("model.obj")

    # Определить min и max координату для выставления сдвига модели
    # print(min([el[2] for el in vertices]))
    # print(max([el[2] for el in vertices]))

    # Чтобы было 3 картинки с разной дальностью до обьекта
    # можно менять scale и shift_z (3 координата)
    # Значения для примеров, похожих на лекционные: 0.1/1000 0.2/2000 5/5_000
    width = 2000
    height = 2000
    scale = 1_000  # тут
    color = [0, 255, 0]
    shift_z = 0.1  # и тут
    trans = [width // 2, height // 2, 0]

    tr_vertices = transform_vertices(vertices, 0, -20, 0, shift_z, scale, trans)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float64)

    for el in faces:
        v1, v2, v3 = (vertices[x - 1] for x in el)
        tv1, tv2, tv3 = (tr_vertices[x - 1] for x in el)

        n = calculate_perpendicular(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2])
        cos = calculate_cos(n)

        if cos < 0:
            xmin, xmax, ymin, ymax = calculate_bounding_rect(tv1[0], tv2[0], tv3[0], tv1[1], tv2[1], tv3[1], width,
                                                             height)
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    l1, l2, l3 = calculate_barycentric_coordinates(x, y, tv1[0], tv2[0], tv3[0], tv1[1], tv2[1], tv3[1])
                    if (l1 >= 0) & (l2 >= 0) & (l3 >= 0):
                        z = l1 * tv1[2] + l2 * tv2[2] + l3 * tv3[2]
                        if z < z_buffer[y, x]:
                            arr[y, x] = [c * -cos for c in color]
                            z_buffer[y, x] = z

    img = Image.fromarray(arr)
    img = ImageOps.flip(img)
    img.save("result.png")

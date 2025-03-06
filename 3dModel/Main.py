from Parser import *
from PIL import Image, ImageOps
from Lines.LineRendering import bresenham_line
import numpy as np

if __name__ == "__main__":
    vertices, faces = parse_obj("model.obj")
    for i in range(len(vertices)):
        vertices[i] = [int(1000 + 10000 * vertices[i][0]), int(500 + 10000 * vertices[i][1]), vertices[i][2]]

    width = 2000
    height = 2000
    color = [0, 255, 0]

    arr = np.zeros((width, height, 3), dtype=np.uint8)

    # Можно убрать первый или второй for, чтобы посмотреть только линии или только вершины
    for el in vertices:
        arr[el[1], el[0]] = color

    for el in faces:
        v1, v2, v3 = (x - 1 for x in el)
        bresenham_line(arr, vertices[v1][0], vertices[v1][1], vertices[v2][0], vertices[v2][1], color)
        bresenham_line(arr, vertices[v1][0], vertices[v1][1], vertices[v3][0], vertices[v3][1], color)
        bresenham_line(arr, vertices[v2][0], vertices[v2][1], vertices[v3][0], vertices[v3][1], color)

    img = Image.fromarray(arr)
    img = ImageOps.flip(img)
    img.save("result.png")

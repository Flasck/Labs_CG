from LineRendering import *
import numpy as np
from PIL import Image
import math

if __name__ == "__main__":
    arr = np.zeros((200, 200), dtype=np.uint8)

    color = 255
    start_x = start_y = 100
    for i in range(0, 13):
        end_x = round(100 + 95 * math.cos((2 * math.pi * i) / 13))
        end_y = round(100 + 95 * math.sin((2 * math.pi * i) / 13))

        # Можно раскомментировать нужное и проверить файлик star.png

        # dotted_line(arr, start_x, start_y, end_x, end_y, 50, color)
        # dotted_line_v2(arr, start_x, start_y, end_x, end_y, color)
        # x_loop_line(arr, start_x, start_y, end_x, end_y, color)
        # x_loop_line_hotfix_1(arr, start_x, start_y, end_x, end_y, color)
        # x_loop_line_hotfix_2(arr, start_x, start_y, end_x, end_y, color)
        # x_loop_line_v2(arr, start_x, start_y, end_x, end_y, color)
        # x_loop_line_v2_no_y_calc(arr, start_x, start_y, end_x, end_y, color)
        # x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(arr, start_x, start_y, end_x, end_y, color)
        bresenham_line(arr, start_x, start_y, end_x, end_y, color)

    image = Image.fromarray(arr)
    image.save("star.png")

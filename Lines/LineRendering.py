import numpy as np
import math


def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1. / count
    for t in np.arange(0, 1, step):
        x = round((1 - t) * x0 + t * x1)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def dotted_line_v2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    step = 1. / count
    for t in np.arange(0, 1, step):
        x = round((1 - t) * x0 + t * x1)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line(image, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line_hotfix_1(image, x0, y0, x1, y1, color):
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        image[(y, x)] = color


def x_loop_line_hotfix_2(image, x0, y0, x1, y1, color):
    changed = False
    if abs(x1 - x0) < abs(y1 - y0):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        changed = True

    for x in range(x0, x1):
        t = ((x - x0) / (x1 - x0))
        y = round((1 - t) * y0 + t * y1)
        image[(x, y) if changed else (y, x)] = color


def x_loop_line_v2(image, x0, y0, x1, y1, color):
    changed = False
    if abs(x1 - x0) < abs(y1 - y0):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        changed = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        image[(x, y) if changed else (y, x)] = color


def x_loop_line_v2_no_y_calc(image, x0, y0, x1, y1, color):
    changed = False
    if abs(x1 - x0) < abs(y1 - y0):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        changed = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    y_edge = 0.
    dy = abs((y1 - y0) / (x1 - x0))
    y_step = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        image[(x, y) if changed else (y, x)] = color
        y_edge += dy
        if y_edge > 0.5:
            y_edge -= 1.
            y += y_step


def x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(image, x0, y0, x1, y1, color):
    changed = False
    if abs(x1 - x0) < abs(y1 - y0):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        changed = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    y_edge = 0.
    dy = 2. * (x1 - x0) * abs((y1 - y0) / (x1 - x0))
    y_step = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        image[(x, y) if changed else (y, x)] = color
        y_edge += dy
        if y_edge > 2. * (x1 - x0) * 0.5:
            y_edge -= 2. * (x1 - x0) * 1.
            y += y_step


def bresenham_line(image, x0, y0, x1, y1, color):
    changed = False
    if abs(x1 - x0) < abs(y1 - y0):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        changed = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    y_edge = 0
    dy = 2 * abs(y1 - y0)
    y_step = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        image[(x, y) if changed else (y, x)] = color
        y_edge += dy
        if y_edge > (x1 - x0):
            y_edge -= 2 * (x1 - x0)
            y += y_step

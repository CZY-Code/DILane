import cv2
import os
import os.path as osp
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]

def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T

def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img,
                 tuple(p1),
                 tuple(p2),
                 color=(255, 255, 255),
                 thickness=width)
    return img

def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious

def culane_metric_v3(pred, #原图基准上的所有（x,y）车道线
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(590, 1640, 3)): #此处每次处理的数据是单张图像
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred) #多检
        fn = 0 if len(pred) != 0 else len(anno) #漏检
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (4, 50, 2)

    ious = discrete_cross_iou(interp_pred,
                            interp_anno,
                            width=width,
                            img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious) #此处对一张图的数据进行线间匹配

    return ious, row_ind, col_ind 


def imshow_lanes(img, lanes, lanes_gt, show=False, out_file=None, width=4):

    lanes_xys_gt = []
    for _, lane in enumerate(lanes_gt):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys_gt.append(xys)
    #lanes_xys_gt.sort(key=lambda xys : xys[0][0])
    for idx, xys in enumerate(lanes_xys_gt):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[0], thickness=width)

    ious, row_ind, col_ind = culane_metric_v3(lanes_gt, lanes)
    dist = ious[row_ind, col_ind]

    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    #lanes_xys.sort(key=lambda xys : xys[0][0])
    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            if idx in col_ind and dist[col_ind==idx]>0.5: #tp
                cv2.line(img, xys[i - 1], xys[i], COLORS[1], thickness=width)
            else: #fp
                cv2.line(img, xys[i - 1], xys[i], COLORS[2], thickness=width)

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)
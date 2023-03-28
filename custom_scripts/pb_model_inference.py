import time
import glob
import cv2
import tensorflow as tf
import torch
import numpy as np
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from tqdm import tqdm


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess_(im):

    img = letterbox(im, (640, 640), stride=32, auto=False)[0]
    # Convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img

def wrap_frozen_graph(gd, inputs, outputs):
    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
    ge = x.graph.as_graph_element
    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))


if __name__ == "__main__":
    FP16 = False
    dir_path = "data/images"
    weights = "checkpoints/yolov5n.pb"
    gd = tf.Graph().as_graph_def()  # graph_def
    with open(weights, 'rb') as f:
        gd.ParseFromString(f.read())
    frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
    paths_list = glob.glob(dir_path + "/*.jpg")
    paths_list*=1

    time_list = []
    loop_count = 1
    for idx in tqdm(range(loop_count)):
        start_time = time.time()
        img_list = []
        orignalimgs_shapes_list = []

        for file_ in paths_list:
            im_orig = cv2.imread(file_)
            orignalimgs_shapes_list.append(im_orig.shape)
            tens = preprocess_(im_orig)
            img_list.append(tens)

        im = np.stack(img_list)  # add images
        im = tf.cast(im, tf.float16) if FP16 else tf.cast(im, tf.float32) # uint8 to fp16/32
        # im = im.astype(np.float16) if FP16 else im.astype(np.float32)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        y = frozen_func(x=im).numpy()
        h, w = im.shape[1:3][::-1]
        y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels
        # pred = torch.tensor(y, device=torch.device('cuda:0'))
        if idx==0:
            print("Model pred shape =========== ",y.shape)

        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        for i, det in enumerate(pred):  # per image
            # gn = torch.tensor(im1_orig.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # print("torch image shape = ", torch_img_list[i].shape)
            # print("original image shape = ", orignalimgs_shapes_list[i])
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[1:3][::-1], det[:, :4], orignalimgs_shapes_list[i]).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if idx==0:
                        print("XYXY = ",xyxy)
                        print("conf = ",conf)
                        print("cls = ",cls)
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # print("XYWH = ",xywh)

        end_time = time.time() - start_time
        time_list.append(end_time)

    print(f"Frame Processing Time: {sum(time_list)/loop_count:.3f}")
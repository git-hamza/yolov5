import json
import math
import os
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.dataloaders import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights, device):
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        LOGGER.info(f'Loading {weights} for TensorRT inference...')
        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False  # default updated below
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        batch_size = bindings['images'].shape[0]

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


path = '1.jpg'
device = '0'  # 'cpu' or 0
weights = 'checkpoints/bestv5x-b1.engine'
imgsz = (640, 640)
stride = 32
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
imgs = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights, device)
print("OK")



videos_path = "videos"
output_vids = "output_vids"
all_videos = os.listdir(videos_path)

for curr_video in all_videos:
    vid_name = curr_video.split(".")[0]
    save_path_det = os.path.join(os.path.join(output_vids, vid_name), "detected")
    save_path_undet = os.path.join(os.path.join(output_vids, vid_name), "undetected")

    if not os.path.exists(save_path_det):
        os.makedirs(save_path_det)
    if not os.path.exists(save_path_undet):
        os.makedirs(save_path_undet)

    cap = cv2.VideoCapture(os.path.join(videos_path, curr_video))

    frame_no = 0
    total_time = 0
    while (cap.isOpened()):
        ret, img0 = cap.read()
        if ret == True:
            frame_no+=1

            import time
            start = time.time()
            img = letterbox(img0, imgsz, stride=stride, auto=False)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            imgs = [img]
            im = np.expand_dims(imgs, axis=0)[0]
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
            end = time.time()
            time_taken = end - start
            total_time+=time_taken
            if len(pred[0]):
                for i, det in enumerate(pred):  # per image
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # print("XYXY = ", xyxy)
                            # print("conf = ", conf)
                            # print("cls = ", cls)

                            x1,y1,x2,y2 = xyxy
                            cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.imwrite(os.path.join(save_path_det, f"{vid_name}_fr_{frame_no}.jpg"), img0)
            else:
                cv2.imwrite(os.path.join(save_path_undet, f"{vid_name}_fr_{frame_no}.jpg"), img0)

        else:
            break

    print(f"Total time = {total_time}")
    cap.release()

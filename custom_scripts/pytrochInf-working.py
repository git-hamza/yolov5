import onnxruntime as rt
import time
import glob
import cv2
import yaml
import torch
import torch.nn as nn
import numpy as np
from utils.augmentations import letterbox
from models.common import Conv
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from tqdm import tqdm


def preprocess(im, device):
    # im = cv2.resize(im, (640,640), interpolation=cv2.INTER_LINEAR)
    im = letterbox(im, (640, 640), stride=32, auto=False)[0]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB

    im = torch.from_numpy(im).to(device)
    return im

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def load_pt_model(path_, device):
    from models.yolo import Detect, Model
    fuse = True
    inplace = True
    model = Ensemble()
    ckpt = torch.load(path_, map_location='cpu')  # load
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
    model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode
    # Compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is Conv:
            m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {path_}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # return ensemble


class Detect(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()
        fp16 &= device.type != 'cpu'  # FP16

        model = load_pt_model(weights, device)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        y = self.model(im, augment=augment, visualize=visualize)[0]

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y


if __name__ == "__main__":
    device = torch.device('cuda:0')
    dir_path = "data/images"
    weights = "checkpoints/yolov5n.pt"
    FP16 = False
    model = Detect(weights, device=device, fp16=FP16)

    paths_list = glob.glob(dir_path + "/*.jpg")
    paths_list*=1

    time_list = []
    loop_count = 1000
    for idx in tqdm(range(loop_count)):

        start_time = time.time()
        torch_img_list = []
        orignalimgs_shapes_list = []

        for file_ in paths_list:
            im = cv2.imread(file_)
            orignalimgs_shapes_list.append(im.shape)
            torch_img = preprocess(im, device)
            torch_img_list.append(torch_img)

        im = torch.stack(torch_img_list)  # add images
        im = im.half() if FP16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        pred = model(im)

        if idx==0:
            print("Model pred shape =========== ",pred.shape)


        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
        if idx==0:
            print("Post NMS pred shape =========== ",pred[0].shape)

        for i, det in enumerate(pred):  # per image
            # gn = torch.tensor(im1_orig.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # print("torch image shape = ", torch_img_list[i].shape)
            # print("original image shape = ", orignalimgs_shapes_list[i])
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], orignalimgs_shapes_list[i]).round()

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



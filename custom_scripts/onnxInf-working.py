import onnxruntime as rt
import time
import glob
import cv2
import torch
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from tqdm import tqdm


def preprocess(im, device):
    im = letterbox(im, (640, 640), stride=32, auto=False)[0]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    im = torch.from_numpy(im).to(device)
    return im


if __name__ == "__main__":
    providers = ['CUDAExecutionProvider']
    model = rt.InferenceSession("checkpoints/yolov5n_fp32_dynamic.onnx", providers=providers)
    FP16 = False
    device = torch.device('cuda:0')

    batch_size = 2
    dir_path = "data/images"  # keep 1 image in the directory
    paths_list = glob.glob(dir_path + "/*.jpg")
    paths_list *= batch_size

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

        im = torch.stack(torch_img_list) # add images
        im = im.half() if FP16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = im.cpu().numpy()  # torch to numpy
        pred = model.run(['output'], {"images": im})[0]
        pred = torch.tensor(pred, device=device)


        if idx==0:
            print("Model pred shape =========== ",pred.shape)

        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
        if idx==0:
            print("Post NMS pred shape =========== ",pred[0].shape)

        for i, det in enumerate(pred):  # per image
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

        end_time = time.time() - start_time
        time_list.append(end_time)

    print(f"Frame Processing Time: {sum(time_list)/loop_count:.3f}")



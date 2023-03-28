import cv2
import torch
from tqdm import tqdm
import time


# Model
model = torch.hub.load('','custom', './checkpoints/yolov5n_fp32_dynamic.onnx', source='local')

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.max_det = 1000  # maximum number of detections per image


time_list = []
loop_count = 1000
batch_size = 1
for idx in tqdm(range(loop_count)):
    # Inference
    start_time = time.time()

    im1 = cv2.imread('data/images/zidane.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    imgs = [im1] * batch_size  # batch of images
    results = model(imgs, size=640)  # includes NMS

    # Results
    if idx == 0:
        results.print()
    # results.save()  # or .show()

    for *xyxy, conf, cls in reversed(results.pred[0]):
        if idx==0:
            print("XYXY = ", xyxy)
            print("conf = ", conf)
            print("cls = ", cls)

    end_time = time.time() - start_time
    time_list.append(end_time)

print(f"Frame Processing Time: {sum(time_list)/loop_count:.3f}")
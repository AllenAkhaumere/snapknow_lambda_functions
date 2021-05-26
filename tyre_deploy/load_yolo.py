# PyTorch Hub
import torch
import yolov5

model_path = "/home/allen/Dataset/tyre_damaged.pt"
# Model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # custom model
model = yolov5.load(model_path=model_path)
# model.iou = 0.4
model.conf = 0.25

#Images
img = '/home/allen/Dataset/Damage Segmentation2/test/IMG_0495.JPG'
img2 = '/home/allen/Dataset/IMG_0555.jpg'
img3 = '/home/allen/Dataset/Tire-pavement-copy.jpg'

# Inference
pred = model(img3)
#nc = pred.shape[2] - 5  # number of classes
#xc = prediction[..., 4] > conf_thres  # candidates
num_of_box = len(pred.xyxy[0])
if num_of_box >= 1:
    print("Send our image")
else:
    print("Send an ok message")

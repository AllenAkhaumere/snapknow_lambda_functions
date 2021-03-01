
from torchvision import models
import json
import urllib
import torch
from PIL import Image
from torchvision import transforms
import logging
import torch.nn.functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)


image_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet18()
model.load_state_dict(torch.load('/home/allen/SnapKnow Deploy/data/resnet18-5c106cde.pth'))
model.eval()



global labels_map
json_file = open('/home/allen/SnapKnow Deploy/data/index_to_name.json')
json_str = json_file.read()
labels_map = json.loads(json_str)

def transform_image(image): 
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image_transforms(image)

def lambda_handler(url):
    image = Image.open(urllib.request.urlopen(url))
    image = transform_image(image)
    image = image.view(-1, 3, 224, 224)
    prediction = F.softmax(model(image)[0])
    topk_probs, topk_idxs = torch.topk(prediction, k=3)
    result = []
    for i in range(len(topk_idxs)):
        pred = {"name": labels_map[str(topk_idxs[i].item())][1],
             "score": topk_probs[i].item()}
        result.append(pred)

    return {'Response': result}

url = 'https://raw.githubusercontent.com/shivajid/Yolo/master/data/eagle.jpg'
print(lambda_handler(url=url))



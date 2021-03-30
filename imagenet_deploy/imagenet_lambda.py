
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

#initialized resnet18 model achitecture
model = models.resnet18()
#load the pretrained weight from directory
model.load_state_dict(torch.load('/home/allen/Documents/GitHub/snapknow_lambda_functions/assets/resnet18-5c106cde.pth'))
#Put model on evaluation mode for inference
model.eval()



global labels_map
#load the file from our directory
json_file = open('/home/allen/SnapKnow Deploy/data/index_to_name.json')
#read the content of the file
json_str = json_file.read()
#load the content in a json
labels_map = json.loads(json_str)

def transform_image(image): 
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image_transforms(image)

def lambda_handler(url):
    image = Image.open(urllib.request.urlopen(url))
    image = transform_image(image)
    image = image.view(-1, 3, 224, 224)
    prediction = F.softmax(input=model(image)[0], dim=0)
    topk_probs, topk_idxs = torch.topk(prediction, k=3)
    result = []
    for i in range(len(topk_idxs)):
        pred = {"name": labels_map[str(topk_idxs[i].item())][1],
             "score": topk_probs[i].item()}
        result.append(pred)

    return {'Response': result}


url1 = 'https://raw.githubusercontent.com/kimx3314/Stanford-Cars-Dataset-Vehicle-Recognition/master/additional_images/AM%20General%20Hummer%20SUV/AM%20General%20Hummer%20SUV.jpg'
url2 = 'https://raw.githubusercontent.com/simoninithomas/Dog-Breed-Classifier/master/images/American_water_spaniel_00648.jpg'
url3 = 'https://raw.githubusercontent.com/simoninithomas/Dog-Breed-Classifier/master/images/Curly-coated_retriever_03896.jpg'
url4 = 'https://raw.githubusercontent.com/simoninithomas/Dog-Breed-Classifier/master/images/Labrador_retriever_06455.jpg'
print(lambda_handler(url=url1))
print(lambda_handler(url=url2))
print(lambda_handler(url=url3))
print(lambda_handler(url=url4))



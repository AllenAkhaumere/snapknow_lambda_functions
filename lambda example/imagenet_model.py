"""Flower Model On Lambda"""

try:
    import unzip_requirements
except ImportError:
    pass
import io
import os
import time
import json

import boto3
import requests
import torch
from PIL import Image
from torchvision import transforms

s3_resource = boto3.resource('s3')
####################################
#Fruit Image Transform             #
####################################
fruit_transforms = transforms.Compose([
    transforms.Resize(200),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

####################################
#Imagenet Image Transform          #
####################################
imagenet_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

####################################
#Flower Image Transform            #
####################################
flower_transforms = transforms.Compose([
    transforms.Resize(200),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def download_image(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            f = io.BytesIO(r.content)
            img = Image.open(f)
            return img
        else:
            return None
    except:
        return None


def download_model(bucket='', key=''):
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.exists(location):
        s3_resource.Object(bucket, key).download_file(location)
    return location

def download_imagenet_to_name_json(bucket='', key=''):
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.exists(location):
        s3_resource.Object(bucket, key).download_file(location)
    return location



def get_prediction(model_path, image, imagenet_to_name_json):
    imagenet_class_index = json.load(open(imagenet_to_name_json))
    model = torch.jit.load(model_path)
    image = imagenet_transforms(image).unsqueeze(0)
    outputs = model(image)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

def classify_image(model_path, img):
    model = torch.jit.load(model_path)
    img = imagenet_transforms(img).unsqueeze(0)
    cl = model(img).argmax().item()
    return cl


def lambda_handler(event, context):
    # download json file
    imagenet_json = download_imagenet_to_name_json(bucket='snapknow_bucket', key='models/index_to_name.json')
    # download model
    model_path = download_model(
        bucket='snapknow_bucket', key='models/imagenet_densenet_model.pt')
    # download image
    img = download_image(event['url'])
    # get prediction
    if img:
        pred = get_prediction(model_path=model_path, image=img, imagenet_to_name_json=imagenet_json)
        return pred
    else:
        return {
            'statusCode': 404,
            'pred is': None
        }

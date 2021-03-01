import json
import logging
import os
import torch
import urllib
from PIL import Image
from torchvision import transforms
from torchvision import models
from fastai.vision import models
import torch.nn as nn

mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
image_transform = transforms.Compose([
            transforms.Resize(size=200),
            transforms.CenterCrop(size=200),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std_dev)
        ])


def tyre_model(model_dir):
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 15),
        nn.Dropout(0.2),
        nn.LogSoftmax(dim=1))

    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    model.eval()
    return model

def prediction(model, image, transform, topk=4):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
    classes = {0: 'bad tyre', 1: 'fairly used tyre',
               2: 'new tyre'}

    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        predictions = torch.exp(output)
        topk, topclass = predictions.topk(topk, dim=1)
        result = []
        for i in range(3):
            pred = {'name': classes[topclass.cpu().numpy()[0][i]], 'score': f'{topk.cpu().numpy()[0][i]}'}
            result.append(pred)
    return result

def inference_handler(event, context):
    tyre_recog = tyre_model(model_dir="/home/allen/SnapKnow Deploy/data/tyre_best_model.pt")
    url = event["body"]["url"]
    image = Image.open(urllib.request.urlopen(url))
    tyre_response = prediction(tyre_recog, image, image_transform, topk=4)

    return {'Response': tyre_response}

def tyre_test(url):
    tyre_recog = tyre_model(model_dir="/home/allen/SnapKnow Deploy/data/tyre_best_model.pt")
    image = Image.open(urllib.request.urlopen(url))
    tyre_response = prediction(tyre_recog, image, image_transform, topk=4)

    return {'Response': tyre_response}

#https://raw.githubusercontent.com/AllenAkhaumere/snapknow_sagemaker_deployment/master/data/tyre_best_model.pt
#https://raw.githubusercontent.com/AllenAkhaumere/snapknow_sagemaker_deployment/master/data/shadrach-warid--gqb3xbGa5Y-unsplash.jpg

print(tyre_test(url="https://raw.githubusercontent.com/AllenAkhaumere/randon_test_data/main/tyres/shadrach-warid--gqb3xbGa5Y-unsplash.jpg"))
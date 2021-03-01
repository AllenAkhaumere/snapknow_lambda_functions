import os
import numpy as np
from torchvision import transforms, datasets
from torchvision import models
from torch import nn
import random
import torch
import os
from PIL import Image
import json
import logging
import urllib
import io
import numpy as np


logger = logging.getLogger()
logger.setLevel(logging.INFO)

mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([transforms.Resize(200),
                                      transforms.CenterCrop(200),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std_dev)])

imagenet_transform = transforms.Compose([
    transforms.Resize(200),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std_dev
    )
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load class_to_name json file
def load_json(json_file):
    with open(json_file, 'r') as f:
        flower_to_name = json.load(f)
        return flower_to_name


def load_txt(file):
    with open(file) as f:
         classes = [line.strip() for line in f.readlines()]
    return classes

def load_index(path):
    json_file = open(path)
    json_str = json_file.read()
    labels = json.loads(json_str)
    return labels



def imagenet_model(model_dir):
    model = models.resnet18()
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model


def fruit_model(model_dir):
    model = models.resnet50(pretrained=False)
    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

    model.class_to_idx = checkpoint['class_to_idx']
    num_classes = checkpoint['output_size']
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def flower_model(model_dir):
    model = models.resnet50(pretrained=False)
    checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))

    model.class_to_idx = checkpoint['class_to_idx']
    num_classes = checkpoint['output_size']
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def prediction(model, image_path, index_to_name, transforms, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.to(device)
    model = model.eval()

    img = Image.open(image_path)
    img = img.convert('RGB')
    transformed_img = transforms(img)
    image = transformed_img
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        predictions = model(image)

    # Convert softmax output to probabilities
    probabilities = torch.softmax(predictions, dim=1)
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = torch.topk(probabilities, dim=1, k=topk)
    # Convert to lists
    top_probabilities, top_indices = top_probabilities.to('cpu').numpy(), top_indices.to('cpu').numpy()
    top_probabilities, top_indices = top_probabilities[0].tolist(), top_indices[0].tolist()
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    # print(idx_to_class)
    top_classes = [idx_to_class[index] for index in top_indices]
    # Convert from the class integer encoding to actual flower names
    object_names = [index_to_name[i] for i in top_classes]
    # map both object_name list and top_prob list into a single dictionary
    dict_zip = dict(zip(object_names, top_probabilities))
    return dict_zip

MAX_RETRY = 5
def get_image(url, timeout=10):
    for tries in range(MAX_RETRY):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as image:
                return image
        except Exception as e:
            logging.warning(str(e) + ',_url:{0}'.format(url))
            if tries < (MAX_RETRY - 1):
                continue
            else:
                print('Has tried {0} times to access url {1}, all failed!'.format(MAX_RETRY, url))
                return None

def predict_imagenet(model, image_path, index_to_name, device, topk=2):
    data = {}
    image_url = get_image(image_path)
    print(image_url)
    image = Image.open(io.BytesIO(image_url))
    img = Image.open(image)
    img = img.convert('RGB')
    image = imagenet_transform(img).unsqueeze(0)
    model.to(device)
    output = model(image)
    prediction = torch.softmax(output)
    topk_probs, topk_idxs = torch.topk(prediction, k=topk)
    data["predictions"] = []

    for i in range(len(topk_idxs)):
        r = {"label": index_to_name[str(topk_idxs[i].item())][0],
             "probability": topk_probs[i].item()}
        data["predictions"].append(r)
    return json.dumps(data)

def max_dict(dictionary):
    return max(dictionary, key=dictionary.get)

class_to_name_dict_fruit = load_json('/home/allen/SnapKnow Deploy/data/fruit_to_name.json')
class_to_name_dict_flower = load_json('/home/allen/SnapKnow Deploy/data/flower_to_name.json')
#class_to_name_imagenet = load_txt('/home/allen/SnapKnow Deploy/data/imagenet.json')

fruit_recog = fruit_model(model_dir='/home/allen/SnapKnow Deploy/data/fruit_best_model.pt')
flower_recog = flower_model(model_dir='/home/allen/SnapKnow Deploy/data/flower_best_model.pth')
#image_recog = imagenet_model(model_dir='/home/allen/SnapKnow Deploy/data/resnet18-5c106cde.pth')

image_dir = '/home/allen/SnapKnow Deploy/data/A_sunflower.jpg'
image_dir2 = '/home/allen/SnapKnow Deploy/data/apple.jpg'
#image_result = predict_imagenet(image_recog, image_dir, class_to_name_imagenet, device,topk=2)
fruit_result = prediction(fruit_recog, image_dir2, class_to_name_dict_fruit, test_transforms, device, topk=1)
flower_result = prediction(flower_recog, image_dir2, class_to_name_dict_flower, test_transforms, device, topk=1)
#merge fruit_result and flower_result dictionary together
all_results = {**fruit_result, **flower_result}
#print(fruit_result)
print(max_dict(all_results))
#print(image_result)

try:
    import unzip_requirements
except ImportError:
    pass

import io
from torchvision import models
from torch import nn
import os
from PIL import Image
import json
import PIL
import boto3
import requests
import torch
from PIL import Image
from torchvision import transforms
import time
import logging
from os import path

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_resource = boto3.resource('s3')
mybucket = 'sagemaker-us-east-1-50325481'

image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Load class_to_name json file
def load_json(json_file):
    with open(json_file, 'r') as f:
        index_to_name = json.load(f)
        return index_to_name


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


def get_model():
    if path.exists("/tmp/fruit_best_model.pt"):
        file_path = os.path.join('/tmp/', 'fruit_best_model.pt')
    else:
        strKey = 'fruit_best_model.pt'
        strFile = '/tmp/fruit_best_model.pt'
        downloadFromS3(mybucket, strKey, strFile)
        file_path = os.path.join('/tmp/', 'fruit_best_model.pt')
    return file_path


def get_json():
    if path.exists("/tmp/fruit_to_name.json"):
        file_path = os.path.join('/tmp/', 'fruit_to_name.json')
    else:
        strKey = 'fruit_to_name.json'
        strFile = '/tmp/fruit_to_name.json'
        downloadFromS3(mybucket, strKey, strFile)
        file_path = os.path.join('/tmp/', 'fruit_to_name.json')
    return file_path


# download files from S3
def downloadFromS3(strBucket, strKey, strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)


def download_model(bucket='', key=''):
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.exists(location):
        s3_resource.Object(bucket, key).download_file(location)
    return location


def download_index_to_name_json(bucket='', key=''):
    location = f'/tmp/{os.path.basename(key)}'
    if not os.path.exists(location):
        s3_resource.Object(bucket, key).download_file(location)
    return location


def input_fn(request_body):
    """Pre-processes the input data from JSON to PyTorch Tensor.
    Parameters
    ----------
    request_body: dict, required
        The request body submitted by the client. Expect an entry 'url' containing a URL of an image to classify.
    Returns
    ------
    PyTorch Tensor object: Tensor

    """
    logger.info("Getting input URL to a image Tensor object")
    if isinstance(request_body, str):
        request_body = json.loads(request_body)
    img_request = requests.get(request_body['url'], stream=True)
    img = PIL.Image.open(io.BytesIO(img_request.content))
    image_tensor = image_transforms(img).unsqueeze(0)
    return image_tensor


def model_fn(model_dir):
    with open(model_dir, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=False)
    # checkpoint = torch.jit.load(model_dir, map_location=torch.device('cpu'))
    model.class_to_idx = checkpoint['class_to_idx']
    num_classes = checkpoint['output_size']
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


def prediction(model, image_path, index_to_name, transforms, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = model.eval()
    img = Image.open(image_path)
    img = img.convert('RGB')
    transformed_img = transforms(img)
    image = transformed_img
    image = image.unsqueeze(0)
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


def inference_handler(event, context):
    index_to_name = get_json()
    index_to_name = load_json(index_to_name)
    # download model
    model_path = get_model()
    # download image
    # input_image = input_fn(event['body'])
    input_image = input_fn(event['body'])
    # return predictions as response
    model = model_fn(model_dir=model_path)
    response = prediction(model, input_image, index_to_name, image_transforms, topk=4)
    return {
        'statusCode': 200,
        'Body': json.dumps(response)
    }

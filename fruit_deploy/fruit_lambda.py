from torchvision import models
from torch import nn
import json
import torch
from PIL import Image
from torchvision import transforms
import urllib
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

image_transforms = transforms.Compose([
    transforms.Resize(200),
    transforms.CenterCrop(200),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Load class_to_name json file
def load_json(json_file):
    with open(json_file, 'r') as f:
        index_to_name = json.load(f)
        return index_to_name


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
    model = model.eval()
    return model


def prediction(model, image, index_to_name, transforms, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = image.convert('RGB')
    image = transforms(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image)
    # Convert softmax output to probabilities
    probabilities = torch.softmax(predictions, dim=1)
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_prob, top_indices = torch.topk(probabilities, k=topk)
    # Convert to lists
    top_indices =  top_indices.to('cpu').numpy()
    top_indices = top_indices[0].tolist()
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    # print(idx_to_class)
    top_classes = [idx_to_class[index] for index in top_indices]
    # Convert from the class into inference_handler encoding to actual flower names
    result = []
    for i in range(topk):
        pred = {'name': index_to_name[top_classes[i]], 'score': f'{top_prob.cpu().numpy()[0][i]}'}
        result.append(pred)
    return result


def inference_handler(event, context):
    fruit_index_to_name = load_json('/mnt/access/model_store/fruit_to_name.json')
    fruit_recog = fruit_model(model_dir='/mnt/access/model_store/fruit_best_model.pth')

    url = event["body"]["url"]
    image = Image.open(urllib.request.urlopen(url))

    fruit_response = prediction(fruit_recog, image, fruit_index_to_name, image_transforms, topk=4)

    return {'Response': fruit_response}

def test_fruit(url):
    fruit_index_to_name = load_json('/home/allen/SnapKnow Deploy/data/fruit_to_name.json')
    fruit_recog = fruit_model(model_dir='/home/allen/SnapKnow Deploy/data/fruit_best_model.pth')

    image = Image.open(urllib.request.urlopen(url))

    fruit_response = prediction(fruit_recog, image, fruit_index_to_name, image_transforms, topk=4)

    return {'Response': fruit_response}

print(test_fruit(url="https://raw.githubusercontent.com/AllenAkhaumere/randon_test_data/main/fruits/mango2.jpeg"))
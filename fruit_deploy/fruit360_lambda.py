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
    transforms.Resize(100),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Load class_to_name json file
def load_json(json_file):
    with open(json_file, 'r') as f:
        index_to_name = json.load(f)
        return index_to_name


def fruit360_model(model_dir):
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
    top_prob, top_class = torch.topk(probabilities, k=topk)

    
    classes = {0: "Apple Braeburn", 1: "Apple Crimson Snow", 2: "Apple Golden", 3: "Apple Golden type 2",
               4: "Apple Golden type 3", 5: "Apple Granny Smith", 6: "Apple Pink Lady", 7: "Apple Red",
               8: "Apple Red type 2", 9: "Apple Red type 3", 10: "Apple Red Delicious",
               11: "Apple Red Yellow", 12: "Apple Red Yellow type 2", 13: "Apricot", 14: "Avocado", 15: "Avocado ripe",
               16: "Banana", 17: "Banana Lady Finger", 18: "Banana Red", 19: "Beetroot", 20: "Blueberry",
               21: "Cactus fruit", 22: "Cantaloupe", 23: "Cantaloupe type 2", 24: "Carambula", 25: "Cauliflower",
               26: "Cherry", 27: "Cherry type 2", 28: "Cherry Rainier", 29: "Cherry Wax Black", 30: "Cherry Wax Red",
               31: "Cherry Wax Yellow", 32: "Chestnut", 33: "Clementine", 34: "Cocos", 35: "Corn", 36: "Corn Husk",
               37: "Cucumber Ripe", 38: "Cucumber Ripe type 2", 39: "Dates", 40: "Eggplant",
               41: "Fig", 42: "Ginger Root", 43: "Granadilla", 44: "Grape Blue", 45: "Grape Pink", 46: "Grape White",
               47: "Grape White type 2", 48: "Grape White type 3", 49: "Grape White type 4", 50: "Grapefruit Pink",
               51: "Grapefruit White", 52: "Guava", 53: "Hazelnut", 54: "Huckleberry", 55: "Kaki", 56: "Kiwi",
               57: "Kohlrabi", 58: "Kumquats", 59: "Lemon", 60: "Lemon Meyer",
               61: "Limes", 62: "Lychee", 63: "Mandarine", 64: "Mango", 65: "Mango Red", 66: "Mangostan",
               67: "Maracuja", 68: "Melon Piel de Sapo", 69: "Mulberry", 70: "Nectarine",
               71: "Nectarine Flat", "72": "Nut Forest", 73: "Nut Pecan", 74: "Onion Red", 75: "Onion Red Peeled",
               76: "Onion White", 77: "Orange", 78: "Papaya", 79: "Passion Fruit", 80: "Peach",
               81: "Peach type 2", 82: "Peach Flat", 83: "Pear", 84: "Pear type 2", 85: "Pear Abate",
               86: "Pear Forelle", 87: "Pear Kaiser", 88: "Pear Monster", 89: "Pear Red", 90: "Pear Stone",
               91: "Pear Williams", 92: "Pepino", 93: "Pepper Green", 94: "Pepper Orange", 95: "Pepper Red",
               96: "Pepper Yellow", 97: "Physalis", 98: "Physalis with Husk", 99: "Pineapple", 100: "Pineapple Mini",
               101: "Pitahaya Red", 102: "Plum", 103: "Plum type 2", 104: "Plum type 3", 105: "Pomegranate",
               106: "Pomelo Sweetie", 107: "Potato Red", 108: "Potato Red Washed", 109: "Potato Sweet",
               110: "Potato White",
               111: "Quince", 112: "Rambutan", 113: "Raspberry", 114: "Redcurrant", 115: "Salak", 116: "Strawberry",
               117: "Strawberry Wedge", 118: "Tamarillo", 119: "Tangelo", 120: "Tomato type 1",
               121: "Tomato type 2", 122: "Tomato type 3", 123: "Tomato type 4", 124: "Tomato Cherry Red",
               125: "Tomato Heart", 126: "Tomato Maroon", 127: "Tomato Yellow", 128: "Tomato not Ripened",
               129: "Walnut", 130: "Watermelon", }

    result = []
    for i in range(topk):
        pred = {'name': classes[top_class.cpu().numpy()[0][i]], 'score': f'{top_prob.cpu().numpy()[0][i]}'}
        result.append(pred)
    return result


def inference_handler(event, context):
    fruit360_index_to_name = load_json('/mnt/access/model_store/fruit360_name_index.json')
    fruit360_recog = fruit360_model(model_dir='/mnt/access/model_store/best_fruit_model360.pt')

    url = event["body"]["url"]
    image = Image.open(urllib.request.urlopen(url))

    fruit_response = prediction(fruit360_recog, image, fruit360_index_to_name, image_transforms, topk=4)

    return {'Response': fruit_response}


def fruit360_test(url):
    fruit360_index_to_name = load_json('/home/allen/lambda_snapknow/assets/fruit360_name_index.json')
    fruit360_recog = fruit360_model(model_dir="/home/allen/SnapKnow Deploy/fruit_file/best_fruit_model360.pt")
    image = Image.open(urllib.request.urlopen(url))
    fruit_response = prediction(fruit360_recog, image, fruit360_index_to_name, image_transforms, topk=4)

    return {'Response': fruit_response}


print(print(fruit360_test(
    url="https://raw.githubusercontent.com/AllenAkhaumere/randon_test_data/main/fruits/mango2.jpeg")))

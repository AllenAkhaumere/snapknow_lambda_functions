from io import BytesIO
from PIL import Image
import base64
import torch



def test_yolo(url):
    model_path = "/home/allen/Dataset/tyre_damaged.pt"
    #torch.hub.set_dir("/home/allen/aws")
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # custom model
    model = torch.hub._load_local('/home/allen/aws/ultralytics_yolov5_master/', model='custom', path=model_path)
    model.conf = 0.4
    image = url
    #image = Image.open(urllib.request.urlopen(url))
    results = model(image, augment=True, size=640)
    num_of_damages_detected = len(results.xyxy[0])
    
    if num_of_damages_detected >= 1:
        results.render()
        for img in results.imgs:
            buffered = BytesIO()
            image_array = Image.fromarray(img)
            image_array.save(buffered, format="JPEG")
            image_to_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_64_decode = base64.b64decode(image_to_base64)
            image_result = open('damaged_tyre_decoder.jpeg',
                                'wb')  # create a writable image and write the decoding result
            image_result.write(image_64_decode)
        return {'Response': image_to_base64}
    return {'The tyre is OK'}


url = 'https://raw.githubusercontent.com/shivajid/Yolo/master/data/eagle.jpg'
img = '/home/allen/Dataset/Damage Segmentation2/test/IMG_0495.JPG'
img2 = '/home/allen/Dataset/Tire-pavement-copy.jpg'
print(test_yolo(url=img))
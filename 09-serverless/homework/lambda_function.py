import numpy as np
import tensorflow.lite as tflite

from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

interpreter = tflite.Interpreter(model_path="dino-vs-dragon-v2.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):
    
    image = download_image(url=url)
    resized_image = prepare_image(img=image, target_size=(150, 150))
    x = np.array(resized_image)
    x = np.float32(x/255)
    X = np.array([x])
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_predictions = preds[0].tolist()
    
    return float_predictions[0]

def lambda_handler(event, context):
    
    url = event["url"]
    result = predict(url=url)
    return result
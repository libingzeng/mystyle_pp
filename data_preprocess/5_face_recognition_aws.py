import boto3
import json
import io
import cv2
import numpy as np
import os

import pdb

celebrity_name = 'denzel_washington'#'taylor_swift'#'Dwayne_Johnson'#'emma_watson'#'leonardo_dicaprio'#'lindsay_lohan'
image_path = 'celebrities1/{}/images'.format(celebrity_name)
attr_path = 'celebrities1/{}/images_attributes/'.format(celebrity_name)
os.makedirs(attr_path, exist_ok=True)

attr_dict = {}

img_name_list = sorted(os.listdir(image_path))

client=boto3.client('rekognition')

for img_name in img_name_list:
    print(img_name)
    if 'jpeg' not in img_name and  'jpg' not in img_name and 'png' not in img_name:
        continue
    img_path = os.path.join(image_path, img_name)
    
    with open(img_path, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])
        face_details = response['FaceDetails']
    attr_dict[img_name] = face_details

with open(os.path.join(attr_path, '{}_aws_dict.json'.format(celebrity_name)), "w") as f:
    json.dump(attr_dict, f)

with open(os.path.join(attr_path, '{}_aws_dict.json'.format(celebrity_name)), "r") as f:
    attr_dict_l = json.load(f)

# pdb.set_trace()

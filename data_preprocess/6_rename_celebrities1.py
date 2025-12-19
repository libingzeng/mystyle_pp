import json
import io
import cv2
import numpy as np
import os

import pdb

celebrity_name = 'joe_biden'#'taylor_swift'#'Dwayne_Johnson'#'emma_watson'#'leonardo_dicaprio'#'lindsay_lohan'
image_path = 'celebrities1/{}/images_train'.format(celebrity_name)
image_renamed_path = 'celebrities1/{}/images_train_renamed'.format(celebrity_name)
attr_path = 'celebrities1/{}/images_train_attributes/'.format(celebrity_name)
os.makedirs(image_renamed_path, exist_ok=True)
os.makedirs(attr_path, exist_ok=True)

with open(os.path.join(attr_path, '{}_aws_dict.json'.format(celebrity_name)), "r") as f:
    attr_aws_dict = json.load(f)
# with open(os.path.join(attr_path, '{}_age_dict.json'.format(celebrity_name)), "r") as f:
#     attr_age_dict = json.load(f)

for img_name in attr_aws_dict.keys():
    # if img_name not in attr_age_dict.keys():
    #     continue
    # age = 'Age' + str(int(attr_age_dict[img_name] / 2)).zfill(3)
    
    img_aws = attr_aws_dict[img_name]
    yk = str(int(img_aws[0]['Pose']['Yaw'] / 5)).zfill(3)
    pk = str(int(img_aws[0]['Pose']['Pitch'] / 5)).zfill(3)
    yaw = 'VY' + str(int(yk))
    pitch = 'VP' + str(int(pk))
    
    # Smile, MouthOpen
    esm = 'S' + str(int(img_aws[0]['Smile']['Value'])) + str(int(img_aws[0]['Smile']['Confidence'] / 20)) + \
        'M' + str(int(img_aws[0]['MouthOpen']['Value'])) + str(int(img_aws[0]['MouthOpen']['Confidence'] / 20))
    exp = 'Exp' + esm

    # EyesOpen, 
    eye = 'Eye' + str(int(img_aws[0]['EyesOpen']['Value'])) + str(int(img_aws[0]['EyesOpen']['Confidence'] / 20))
   
    img_name_new = celebrity_name.lower() + '_' + yaw + '_' + pitch + '_' + exp  + '_' + img_name.split('_')[-2]  + '_' + img_name.split('_')[-1]


    
    # age_eye_name = age + '_' + eye
    # img_name_new =img_name.replace('align1500', age_eye_name)
    # img_name_new = celebrity_name.lower() + '_' + yaw + '_' + pitch + '_' + exp  + '_' + img_name.split('_')[-1].split('.')[0] + '.jpg'

    
    # pdb.set_trace()

    img_path = os.path.join(image_path, img_name)
    img_renamed_path = os.path.join(image_renamed_path, img_name_new)
    
    cmd = 'cp -r {} {}'.format(img_path, img_renamed_path)
    os.system(cmd)

    print(img_name)
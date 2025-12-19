import os
import sys
import pdb
import random
random.seed(10)

celebrities_dict = {0:'barack_obama', 1:'Dwayne_Johnson', 2:'joe_biden', 3:'oprah_winfrey', \
    4:'scarlett_johansson', 5:'michelle_obama', 6:'taylor_swift', \
    7:'Emma_Watson', 8:'Leonardo_Dicaprio', 9:'lindsay_lohan', 10:'denzel_washington'}
c_id = 0

# for i in range(7):
# c_id = i
all_list_path = 'celebrities1/{}/images_renamed'.format(celebrities_dict[c_id])
train_list_path = 'celebrities1/{}/images_train'.format(celebrities_dict[c_id])
test_list_path = 'celebrities1/{}/images_test'.format(celebrities_dict[c_id])
os.makedirs(train_list_path, exist_ok = True)
os.makedirs(test_list_path, exist_ok = True)
train_ratio = 0.9
all_list = os.listdir(all_list_path)
for img in all_list:
    img_path = os.path.join(all_list_path, img)
    random_num = random.uniform(0, 1)
    if random_num < train_ratio:
        cmd_str = 'cp {} {}'.format(img_path, train_list_path)
    else:
        cmd_str = 'cp {} {}'.format(img_path, test_list_path)

    os.system(cmd_str)
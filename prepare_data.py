# Copyright 2017 Sunkari Preetham Paul. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################################
"""
//RESIZING IMAGES IN BULK FOR HOG DESCRIPTOR :
Used for preparing new data
Resizes all the images into 32X32 sized images
New data is created in the folder 'new_train_data'
Ensure that images to be resized are placed in the folder 'raw_train_data'
"""
################################################################################################
import os,sys
from PIL import Image

cwd = os.getcwd()
train_path = os.path.join(cwd,'raw_train_data')
dest_path = os.path.join(cwd,'new_train_data')
new_dim = [32,32]


def create_image_data():
    if not os.path.exists(train_path):
            os.makedirs(train_path)
    if not os.path.exists(dest_path):
            os.makedirs(dest_path)

    class_list = []

    class_list.extend(os.listdir(train_path))

    for folder,val in enumerate(class_list):
        class_path = os.path.join(train_path,val)
        new_class_path = os.path.join(dest_path,val)

        if not os.path.exists(new_class_path):
                os.makedirs(new_class_path)

        image_list = os.listdir(class_path)

        for i in image_list:
            img_path = os.path.join(class_path,i)
            new = os.path.join(new_class_path,i)
            if not os.path.exists(new):
                print('Re-saving... '+str(i))
                image = Image.open(img_path)
                image = image.resize(new_dim)
                image = image.save(new)
            else:
                print('Found... '+str(i))
            

if __name__ == '__main__' :
	create_image_data()
	print('\nRe-saving Complete....')

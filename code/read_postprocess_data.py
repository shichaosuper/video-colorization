import os
import random
import time
import numpy as np
from PIL import Image
import json
import yaml



dict = {}
train_val_dict = {}
test_dict = {}

def eachFile0(filepath):
    global dict;
    train_data = []
    val_data = []
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        sub_Dir_path = filepath + allDir + "/"
        subDir_ =  os.listdir(sub_Dir_path)
        if(allDir in train_val_dict):
          tmp_data = []
          for jpg_ in subDir_:
              if jpg_[-3:] == "npy":
                tmp_data += [sub_Dir_path + jpg_]
          train_data += [tmp_data]
        else:
          tmp_data = []
          for jpg_ in subDir_:
              if jpg_[-3:] == "npy":
                tmp_data += [sub_Dir_path + jpg_]
          val_data += [tmp_data]
    dict['train_data'] = list(train_data)
    dict['val_data'] = list(val_data)
    print("collecting finished!")
        
def eachFile1(filepath):
    global dict;
    val_data = []
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        sub_Dir_path = filepath + allDir + "/"
        subDir_ =  os.listdir(sub_Dir_path)
        tmp_data = []
        for jpg_ in subDir_:
            #im = Image.open(sub_Dir_path + jpg_)
            #im = np.array(im)
            tmp_data += [sub_Dir_path + jpg_]
            #time.sleep(6)
        val_data += [tmp_data]
    dict['annotation'] = list(val_data)
    
    print("collecting finished!")
    
    
def read_data():
    output = open("train_data.json", 'w')
    data_path_train_data = "./processed_data/OSVOS/"
    print("building index from " + data_path_train_data)
    eachFile0(data_path_train_data)
    
    data_path_train_data = "./data/Annotations/"
    print("building index from " + data_path_train_data)
    eachFile1(data_path_train_data)
    json.dump(dict, output)
    
    
def get_video(video, part, step, img_size1, img_size2):
    data, ann = [], []
    for k in range(part, part + step):

        im = Image.open("./data/train_data/" + video[k][23:-7]+"jpg")
        im = im.resize((100, 100),Image.ANTIALIAS)
        im = np.array(im)

        data += [im]
        
        im = Image.open("./data/train_data/" + video[k][23:-7]+"jpg")
        im = im.resize((100, 100),Image.ANTIALIAS)
        im = im.convert('L') 
        im = np.array(im)

        ann += [im]
    return data, ann
Iter1 = 0
Iter2 = 0
def get_data_divided():
    global train_val_dict
    global test_dict
    f = open('./processed_data/db_info.yml')  
    x = yaml.load(f)
    for pr in x['sequences']:
      if (pr['set'] == 'training'):
        train_val_dict[pr['name']] = pr['num_frames']
      if (pr['eval_t'] == True):
        test_dict[pr['name']] = pr['num_frames']
      
    
def initial_get_batch():
    global dict, Iter
    Iter = 10
    input = open("train_data.json")
    dict = json.load(input)
    
def get_batch(batch_size, step, img_size1, img_size2, data_type = 1):
    #[batch_size, step, img_size, img_size, channel]
    global Iter1
    global Iter2
    max_iter_1 = len(dict['train_data'])
    max_iter_2 = len(dict['val_data'])
    ans1, ans2 = [], []

    if(data_type == 1):
        for k in range(batch_size):
            Iter1 %= max_iter_1
            video = dict['train_data'][Iter1]
            part = random.randint(0, len(video) / step - 1) 
            sub_video_data, sub_video_ann = get_video(video, part, step, img_size1, img_size2)
            ans1 += [sub_video_data]
            ans2 += [sub_video_ann]
            Iter1 += 1
        ans1 = np.array(ans1)
        ans2 = np.array(ans2)
    else:
        for k in range(batch_size):
            Iter2 %= max_iter_2
            video = dict['val_data'][Iter2]
            part = random.randint(0, len(video) / step - 1) 
            sub_video_data, sub_video_ann = get_video(video, part, step, img_size1, img_size2)
            ans1 += [sub_video_data]
            ans2 += [sub_video_ann]
            Iter2 += 1
        ans1 = np.array(ans1)
        ans2 = np.array(ans2)
    return ans1, ans2
'''get_data_divided()
read_data()
initial_get_batch()
ans1,ans2 = get_batch(1,1,224,1)
print(ans1)'''
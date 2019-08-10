import matplotlib.pyplot as plt
import copy
import matplotlib.image as mpimg
import numpy as np
import os
direction_path=r'UCF-101/ucfTrainTestlist/'
file_path=r'UCF-101/'
image_path=r'UCFImage/'
dest_path=r'./UCFImage/'
action={}
action_label={}
action_name={}
x_dataset=[]
y_dataset=[]
now_start=0
item=[]
with open(direction_path+'classInd.txt') as f:
    content = f.readlines()
    content = [x.strip('\r\n') for x in content]
f.close()
i=0
for line in content:
    label,action = line.split(' ')
    print(label,action)
    if action not in action_label.keys():
        action_label[action]=label
    action_name[i]=action
    i=i+1
print(type(action_name))
for typenum in range(len(action_name)):
    if typenum>=2:
        break
    print('in')
    print(image_path+action_name[typenum])
    path_list= os.listdir(image_path+action_name[typenum])
    path_list.sort()
    print(path_list)
    for files in path_list:
        print(image_path+action_name[typenum]+'/'+files)
        try:
            #IC = mpimg.imread(image_path+action_name[typenum]+'/'+files)
            I=image_path+action_name[typenum]+'/'+files
            print('typenum:',typenum)
            x_dataset.append(I)
            y_dataset.append(np.eye(10)[typenum])
            print((np.eye(10)[typenum]))
        except KeyboardInterrupt:
            exit()
        except OSError:
            os.remove(image_path+action_name[typenum]+'/'+files)
        except :
            continue
x_dataset=np.array(x_dataset)
y_dataset=np.array(y_dataset)
permutation=np.random.permutation(x_dataset.shape[0])
x_dataset=x_dataset[permutation]
y_dataset=y_dataset[permutation, :]

def Normalize(data):
    data_r=data/255
    data_r.astype(np.float32)
    return data_r

def get_next_batch(batch_size):
    global x_dataset
    global y_dataset
    global now_start
    now_start+=batch_size
    x_batch=[]
    if (now_start+batch_size)>=x_dataset.shape[0]:
        print('shuffling!-------------------------------------------------------------')
        permutation=np.random.permutation(x_dataset.shape[0])
        x_dataset=x_dataset[permutation]
        y_dataset=y_dataset[permutation, :]
        now_start=0
    x_name=x_dataset[now_start:now_start+batch_size]
    for name in x_name:
        x_batch.append(mpimg.imread(name))
    x_batch=np.array(x_batch)
    y_batch=y_dataset[now_start:now_start+batch_size,:]
    x_batch=Normalize(x_batch)
    return x_batch,y_batch


import numpy as np
import cv2
import os
direction_path=r'UCF-101/ucfTrainTestlist/'
file_path=r'UCF-101/'
dest_path=r'./UCFImage/'
action={}
action_label={}
action_name={}
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
# label types has been finished until here
print(type(action_name))
for typenum in action_name:
    os.mkdir(dest_path+action_name[typenum])
    for root, dirs, files in os.walk(file_path+action_name[typenum], topdown=False):
        k=0
        for name in files:
            print(os.path.join(root, name))

            vc = cv2.VideoCapture(os.path.join(root, name))
            #vc = cv2.VideoCapture(r'UCF-101/Archery/v_Archery_g01_c02.avi')
            n = 1  

            rval, frame = vc.read()

            timeF = 10  

            i = 0
            while rval: 
                rval, frame = vc.read()
                if (n % timeF == 0):  
                    i += 1
                    cv2.imwrite(dest_path+action_name[typenum]+r'/'+str(k)+'_'+str(i)+'.jpg', frame)
                    print(frame)
                    print('dest:',dest_path+action_name[typenum]+r'/'+str(k)+'_'+str(i)+'.jpg')
                n = n + 1
                cv2.waitKey(1)
            vc.release()
            k+=1


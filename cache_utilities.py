import numpy as np
import math
import time
W=20
H=20
bound_x=int(240/W)
bound_y=int(320/H)
nspr_v=np.zeros([bound_x,bound_y,bound_x,bound_y])
class PICT():
    def __init__(self,content,num_x,num_y,width,height,Mx=0,My=0):
        self.content=content
        self.num_x=num_x
        self.num_y=num_y
        self.Mx=Mx
        self.My=My
        if width*(num_x+1)<=240:
            self.coor_x=W*num_x
        if width*(num_y+1)<=320:
            self.coor_y=H*num_y
        self.width=width
        self.height=height
    def printPict(self):
        print('num_x:',self.num_x,'\tnum_y:',self.num_y,'\twidth:',self.width,'\theight:',self.height)

def NSPR(pict1,pict2):
    sums=0
    if pict1.width!=pict2.width:
        return 0
    if pict1.height!=pict2.height:
        return 0
    if pict1.height!=H:
        return 0
    if pict1.width!=W:
        return 0
    ref_data=pict1.content[int(pict1.width*pict1.num_x):int(pict1.width*(pict1.num_x+1)),int(pict1.height*pict1.num_y):int(pict1.height*(pict1.num_y+1))]
    target_data=pict2.content[int(pict2.width*pict2.num_x):int(pict2.width*(pict2.num_x+1)),int(pict2.height*pict2.num_y):int(pict2.height*(pict2.num_y+1))]
    diff = ref_data - target_data
    diff=diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    if rmse==0:
        return 10000
    return 20*math.log10(1.0/rmse)

def Diamond_Search(pict1,previous_image):
#never think about corner, it would be added later
    x=pict1.num_x
    y=pict1.num_y
    Big_max=0
    Big_num=0
    Big_scale=[[x,y]]
    Big_score=[]
    if y-2>=0:
        Big_scale.append([x,y-2])
    if y-1>=0 and x-1>=1:
        Big_scale.append([x-1,y-1])
    if y-1>=0 and x+1<bound_x:
        Big_scale.append([x+1,y-1])
    if x-2>=0:
        Big_scale.append([x-2,y])
    if y+1<bound_y and x-1>=0:
        Big_scale.append([x-1,y+1])
    if y-1>=0 and x+1<bound_x:
        Big_scale.append([x+1,y-1])
    if y+2<bound_y:
        Big_scale.append([x,y+2])
    if x+2<bound_x:
        Big_scale.append([x+2,y])
    for i in range(len(Big_scale)):
        target=PICT(previous_image,Big_scale[i][0],Big_scale[i][1],W,H)
        score=NSPR(pict1,target)
        nspr_v[pict1.num_x,pict1.num_y,target.num_x,target.num_y]=score
        Big_score.append(score)
    Big_num=np.argmax(Big_score)
    x_s=Big_scale[Big_num][0]
    y_s=Big_scale[Big_num][1]
    Small_num=0
    Small_max=0
    Small_scale=[[x_s,y_s]]
    if y_s-1>=0:
        Small_scale.append([x_s,y_s-1])
    if y_s+1<bound_y:
        Small_scale.append([x_s,y_s+1])
    if x_s+1<bound_x:
        Small_scale.append([x_s+1,y_s])
    if x_s-1>=0:
        Small_scale.append([x_s-1,y_s])
    Small_score=[]
    for i in range(len(Small_scale)):
        target=PICT(previous_image,Small_scale[i][0],Small_scale[i][1],W,H)
        score=NSPR(pict1,target)
        if Small_max<score:
            Small_max=score
            Small_num=i
        Small_score.append(score)
    return Small_scale[Small_num][0],Small_scale[Small_num][1]

def density(List,left,top,right,down):
    cardinal=(right-left+1)*(down-top+1)
    if cardinal==0:
        return 0
    effective=0
    for item in List:
        item_right=item.num_x+int(item.width/W)-1
        item_down=item.num_y+int(item.height/H)-1
        if item.num_x>=left and item.num_y>=top and item_right<=right and item_down<=down:
            effective+=1
    return (effective/cardinal)


def ComputeOverlay(picture1,picture2):
    tao=15
    total_num=0
    sum_x=0
    sum_y=0
    result=[]
    o_time=time.time()
    
    for i in range(bound_x):
        if i%2:
            for j in range(bound_y):
                if j%2==0:
                    pict1=PICT(picture1,i,j,W,H)
                    x,y=Diamond_Search(pict1,picture2)
                    pict2=PICT(picture2,x,y,W,H)
                    if NSPR(pict1,pict2)>tao :
                        total_num+=1
                        sum_x+=(pict1.num_x-pict2.num_x)
                        sum_y+=(pict1.num_y-pict2.num_y)
    #step 3
    if total_num!=0:
        Mx=int(sum_x/total_num)
        My=int(sum_y/total_num)
    else:
        return np.array([0,0,0,0,0,0])
    #step into the fantastic great algorithm, the cube merge algorithm! time complexity n^2.....well it seems that it is not great but useful...
    next_cube=PICT(picture1,0,0,W,H)
    count=0
    effective_blocks=0
    s_time=time.time()
    for i in range(bound_x):
        if i%2:
            for j in range(bound_y):
                if j%2==0 and nspr_v[i,j,(i+Mx)%bound_x,(j+My)%bound_y]>tao:
                    pict1=PICT(picture1,i,j,W,H,Mx,My)
                    result.append(pict1)
                    effective_blocks+=1
    '''
    while next_cube!=None:
        pict_s=next_cube
        pict_s.Mx=Mx
        pict_s.My=My
        if pict_s.width==W and pict_s.height==H:
            count+=1
        next_cube=PICT(picture1,pict_s.num_x+pict_s.width/W,pict_s.num_y-1+pict_s.height/H,W,H)
        if pict_s.num_x+(pict_s.width/W)>=bound_x:
            next_cube=PICT(picture1,0,pict_s.num_y+(pict_s.height/H),W,H)
            if pict_s.num_y+(pict_s.height/H)>=bound_y:
                next_cube=None

        if pict_s.num_x+Mx>=0 and pict_s.num_x+Mx<=bound_x and pict_s.num_y+My>=0 and pict_s.num_y+My<=bound_y:
            pict_t=PICT(picture2,pict_s.num_x+Mx,pict_s.num_y+My,W,H)
            if NSPR(pict_s,pict_t)>tao:
                if pict_s.width==W and pict_s.height==H:
                    result.append(pict_s)
                    effective_blocks+=1
    print('effective:',len(result))
    '''
    e_time=time.time()
    dens=0
    rate=0.20
    [left,top]=[0,0]
    [right,down]=[bound_x-1,bound_y-1]
    direction=0
    print('initial time:',1000*(s_time-o_time))
    print('extra time:',1000*(e_time-s_time))
    while(dens<rate):
        if left<=bound_x-1:
            left+=1
        else:
            break
        dens=density(result,left,top,right,down)
    if(dens>=rate):
        return np.array([left,top,(right-left+1)*W,(down-top+1)*H,Mx,My])
    [left,top]=[0,0]
    [right,down]=[bound_x-1,bound_y-1]
    while(dens<rate):
        if top<=bound_y-1:
            top+=1
        else:
            break
        dens=density(result,left,top,right,down)
    if(dens>=rate):
        return np.array([left,top,(right-left+1)*W,(down-top+1)*H,Mx,My])
    [left,top]=[0,0]
    [right,down]=[bound_x-1,bound_y-1]
    while(dens<rate):
        if right>=1:
            right-=1
        else:
            break
        dens=density(result,left,top,right,down)
    if(dens>=rate):
        return np.array([left,top,(right-left+1)*W,(down-top+1)*H,Mx,My])
    [left,top]=[0,0]
    [right,down]=[bound_x-1,bound_y-1]
    while(dens<rate):
        if down>=1:
            down-=1
        else:
            break
        dens=density(result,left,top,right,down)

    return np.array([left,top,(right-left+1)*H,(down-top+1)*W,Mx,My])

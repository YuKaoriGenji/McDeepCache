import tensorflow as tf
import math
W=20
H=20
bound_x=int(240/W)
bound_y=int(320/H)
NUMX=0
NUMY=1
WIDTH=2
HEIGHT=3
MX=4
MY=5
#0       1       2       3       4       5
#num_x   num_y   width   height  Mx      My 
def NSPR(pict1,pict2,block1,block2):
    sums=tf.constant(0)
    devide=block1[WIDTH]*block2[HEIGHT]
    ref_data=pict1[block1[WIDTH]*block1[NUMX]:block1[WIDTH]*(block1[NUMX]+1),block1[HEIGHT]*block1[NUMY]:block1[HEIGHT]*(block1[NUMY]+1)]
    target_data=pict2[block2[WIDTH]*block2[NUMX]:block2[WIDTH]*(block2[NUMX]+1),block2[HEIGHT]*block2[NUMY]:block2[HEIGHT]*(block2[NUMY]+1)]
    diff = ref_data - target_data
    diff=tf.reshape(diff,[-1])
    rmse=tf.sqrt(tf.reduce_mean(tf.cast(tf.square(diff),'float')))
    print('mark:',tf.log(tf.cast(10,'float')))
    nspr=tf.cond(tf.equal(rmse,0),lambda: tf.cast(10000,'float'),lambda: 20*(tf.log(1.0/rmse)/tf.log(tf.cast(10,'float'))))
    nspr=tf.cond(tf.equal(block1[HEIGHT],block2[HEIGHT]) & tf.equal(block2[WIDTH],block2[WIDTH]) ,lambda: nspr,lambda: tf.cast(0,'float'))
    return nspr
def Diamond_Search(pict,block1,previous_image):
    #x=tf.cast(block1[NUMX],'float')
    #y=tf.cast(block1[NUMY],'float')
    x=block1[NUMX]
    y=block1[NUMY]
    Big_scale=tf.convert_to_tensor([[x,y],[x,y-2],[x-1,y-1],[x+1,y-1],[x-2,y],[x-1,y+1],[x+1,y-1],[x,y+1],[x+2,y]])
    Big_scale=tf.stack([x,y])
    Big_scale=tf.expand_dims(Big_scale,0)
    tmp=tf.stack([x,y-2])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond(y-2>=0,lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x-1,y-1])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond((y-1>=0) & (x-1>=0),lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x+1,y-1])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond((y-1>=0) & (x+1<bound_x),lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x-2,y])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond(x-2>=0,lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x-1,y+1])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond((x-1>=0) & (y+1<bound_y),lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x,y+2])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond(y+2<=bound_y,lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x+2,y])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond(x+2>=bound_x,lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    tmp=tf.stack([x+1,y+1])
    tmp=tf.expand_dims(tmp,0)
    Big_scale=tf.cond((y+1>=bound_y) & (x+1>=bound_x),lambda: tf.concat([Big_scale,tmp],0),lambda: tf.concat([Big_scale,[[100,100]]],0))
    print('Big_scale:',Big_scale)
    score0=tf.cond(tf.not_equal(Big_scale[0,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[0,0],Big_scale[0,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score1=tf.cond(tf.not_equal(Big_scale[1,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[1,0],Big_scale[1,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score2=tf.cond(tf.not_equal(Big_scale[2,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[2,0],Big_scale[2,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score3=tf.cond(tf.not_equal(Big_scale[3,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[3,0],Big_scale[3,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score4=tf.cond(tf.not_equal(Big_scale[4,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[4,0],Big_scale[4,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score5=tf.cond(tf.not_equal(Big_scale[5,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[5,0],Big_scale[5,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score6=tf.cond(tf.not_equal(Big_scale[6,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[6,0],Big_scale[6,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score7=tf.cond(tf.not_equal(Big_scale[7,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[7,0],Big_scale[7,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score8=tf.cond(tf.not_equal(Big_scale[8,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[8,0],Big_scale[8,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score=tf.convert_to_tensor([score0,score1,score2,score3,score4,score5,score6,score7,score0])
    #return score
    Big_num=tf.argmax(score)
    score0=tf.cond(tf.not_equal(Big_scale[0,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Big_scale[0,0],Big_scale[0,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    x_s=Big_scale[Big_num,0]
    y_s=Big_scale[Big_num,1]
    Small_scale=tf.stack([x_s,y_s])
    Small_scale=tf.expand_dims(Small_scale,0)
    tmp=tf.stack([x_s,y_s-1])
    tmp=tf.expand_dims(tmp,0)
    Small_scale=tf.cond(y_s-1>=0,lambda: tf.concat([Small_scale,tmp],0),lambda: tf.concat([Small_scale,[[100,100]]],0))
    tmp=tf.stack([x_s,y_s+1])
    tmp=tf.expand_dims(tmp,0)
    Small_scale=tf.cond(y_s+1<bound_y,lambda: tf.concat([Small_scale,tmp],0),lambda: tf.concat([Small_scale,[[100,100]]],0))
    tmp=tf.stack([x_s+1,y_s])
    tmp=tf.expand_dims(tmp,0)
    Small_scale=tf.cond(x_s+1<bound_x,lambda: tf.concat([Small_scale,tmp],0),lambda: tf.concat([Small_scale,[[100,100]]],0))
    tmp=tf.stack([x_s-1,y_s])
    tmp=tf.expand_dims(tmp,0)
    Small_scale=tf.cond(x_s-1>=0,lambda: tf.concat([Small_scale,tmp],0),lambda: tf.concat([Small_scale,[[100,100]]],0))
    score0=tf.cond(tf.not_equal(Small_scale[0,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Small_scale[0,0],Small_scale[0,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score1=tf.cond(tf.not_equal(Small_scale[1,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Small_scale[1,0],Small_scale[1,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score2=tf.cond(tf.not_equal(Small_scale[2,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Small_scale[2,0],Small_scale[2,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score3=tf.cond(tf.not_equal(Small_scale[3,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Small_scale[3,0],Small_scale[3,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score4=tf.cond(tf.not_equal(Small_scale[4,0],100),lambda: NSPR(pict,previous_image,block1,tf.convert_to_tensor([Small_scale[4,0],Small_scale[4,1],W,H,0,0])),lambda: tf.convert_to_tensor(0,'float'))
    score_s=tf.convert_to_tensor([score0,score1,score2,score3,score4])
    Small_num=tf.argmax(score)
    return Small_scale[Small_num,0],Small_scale[Small_num,1]

def ComputeOverlay(picture1,picture2):
    block_all=np.zeros([12,16,6],dtype=np.int)
    tao=tf.cast(15,'float')
    total_num=tf.convert_to_tensor(0)
    sum_x=tf.convert_to_tensor(0)
    sum_y=tf.convert_to_tensor(0)
    i=tf.convert_to_tensor(0)
    for i in range(12):
        for j in range(16):
            block_all[i,j]=np.array([i,j,20,20,0,0])

    print(block_all)

    def cond1(i,pict1,pict2,total_num,sum_x,sum_y,tao):
        return i<12
    def body1(i,pict1,pict2,total_num,sum_x,sum_y,tao):
        def cond2(i,j,pict1,pict2,total_num,sum_x,sum_y,tao):
            return j<16
        def body2(i,j,pict1,pict2,total_num,sum_x,sum_y,tao):
            block1=tf.convert_to_tensor([i,j,20,20,0,0])
            x,y=Diamond_Search(pict1,block1,pict2)
            block2=tf.convert_to_tensor([x,y,20,20,0,0])
            total_num=tf.cond((NSPR(pict1,pict2,block1,block2)>tao),lambda:total_num+1,lambda:total_num)
            sum_x=tf.cond((NSPR(pict1,pict2,block1,block2)>tao),lambda:sum_x+(block1[NUMX]-block2[NUMX]),lambda:sum_x)
            sum_y=tf.cond((NSPR(pict1,pict2,block1,block2)>tao),lambda:sum_y+(block1[NUMY]-block2[NUMY]),lambda:sum_y)
            j+=1
            return i,j,pict1,pict2,total_num,sum_x,sum_y,tao
        j=tf.convert_to_tensor(0)
        i,j,pict1,pict2,total_num,sum_x,sum_y,tao=tf.while_loop(cond2,body2,[i,j,pict1,pict2,total_num,sum_x,sum_y,tao])
        i+=1
        return i,pict1,pict2,total_num,sum_x,sum_y,tao
    i,picture1,picture2,total_num,sum_x,sum_y,tao=tf.while_loop(cond1,body1,[i,picture1,picture2,total_num,sum_x,sum_y,tao])
    Mx=sum_x/total_num
    print('Mx:',Mx)
    Mx=tf.cast((sum_x/total_num),'int32')
    My=tf.cast(sum_y/total_num,'int32')
    next_cube=tf.convert_to_tensor([0,0,20,20,0,0])
    def cond3(next_cube,picture1,Mx,My,k):
        return tf.not_equal(next_cube[0],100)
    def body3(next_cube,picture1,Mx,My,k):
        pict_s=next_cube
        next_cube=tf.convert_to_tensor([tf.cast(pict_s[NUMX]+(pict_s[WIDTH]/20),'int'),tf.cast(pict_s[NUMY]-1+(pict_s[HEIGHT]/20),'int'),20,20,0,0])
        next_cube=tf.cond(((pict_s[NUMX]+(pict_s[WIDTH]/20))>=tf.cast(bound_x),'int'),tf.convert_to_tensor([0,tf.cast(pict_s[NUMY]+(pict_s[HEIGHT]/20),'int'),20,20,0,0]),next_cube)
        next_cube=tf.cond((tf.cast(pict_s[NUMY]+(pict_s[HEIGHT]/20),'int')>=tf.cast(bound_y,'int')) &(((pict_s[NUMX]+(pict_s[WIDTH]/20))>=tf.cast(bound_x),'int')) , [100,100,100,100,100,100],next_cube)
    return sum_y

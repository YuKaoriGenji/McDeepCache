import tensorflow as tf
import numpy as np
import TestData as ID
#import ImageData as ID
import cache_utilities as cu
import os
import time
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from ctypes import *
import ctypes
KEEPPRO=1
x_dataset=ID.x_dataset
y_dataset=ID.y_dataset
batch_size=1
now_start=0
learning_rate=1e-5
W=20
H=20
layer1_num=512
class StructPointer(ctypes.Structure):
    _fields_=[("numx",ctypes.c_int),("numy",ctypes.c_int),("width",ctypes.c_int),("height",ctypes.c_int),("Mx",ctypes.c_int),("My",ctypes.c_int)]

lib=ctypes.cdll.LoadLibrary("./cache_uti.so")
print('lib:',lib)
def compute(picture1,picture2):
    tmp=np.asarray(picture1)
    data1=tmp.ctypes.data_as(ctypes.c_char_p)
    tmp=np.asarray(picture2)
    data2=tmp.ctypes.data_as(ctypes.c_char_p)
    lib.ComputeOverlay.restype=ctypes.POINTER(StructPointer)
    res=lib.ComputeOverlay(data1,data2)
    result=np.array([res.contents.numx,res.contents.numy,res.contents.width,res.contents.height,res.contents.Mx,res.contents.My]) 
    #print('result:',result)
    return result

def maxPoolLayer(x,kHeight,kWidth,strideX,strideY,name,padding="SAME"):
    print(name,'out----------------------',tf.nn.max_pool(x,ksize=[1,kHeight,kWidth,1],strides=[1,strideX,strideY,1],padding=padding,name=name))
    return tf.nn.max_pool(x,ksize=[1,kHeight,kWidth,1],strides=[1,strideX,strideY,1],padding=padding,name=name)

def dropout(x,keepPro,name=None):
    #return tf.nn.dropout(x,keepPro,name)
    return x

def LRN(x,R,alpha,beta,name=None,bias=1.0):
    return tf.nn.local_response_normalization(x,depth_radius=R,alpha=alpha,beta=beta,bias=bias,name=name)

def fcLayer(x,inputD,outputD,reluFlag,name):
    with tf.variable_scope(name) as scope:
        w=tf.get_variable("w",shape=[inputD,outputD],dtype="float")
        b=tf.get_variable("b",[outputD],dtype="float")
        out=tf.nn.xw_plus_b(x,w,b,name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def conv_c(a,b,List,strideX,strideY,padding,previous_feature):
    batch_size=a.get_shape()[0]    # width
    width=int(a.get_shape()[1])    # width
    print('width:++++++++++++++++++++++++++++++++++++++\n',width)
    height=int(a.get_shape()[2])   # height
    kernel_x=int(b.get_shape()[1])
    kernel_y=int(b.get_shape()[0])
    channels=int(b.get_shape()[2])
    out_channels=int(b.get_shape()[3])
    #Lis_len=int(List.get_shape()[0])
    pict_n=List[0]      # the blank we operate 
    #List.remove(pict_n) # delete the blank of the list
    # condition
    # all in picture
    #List=List[1:Lis_len-1]
    print('x_coor:',List[0],'y_coor:',List[1])
    print('b:',b)
    # split the graph and compute the convolution
    graph_A=tf.slice(a,[0,0,0,0],[-1,width,List[1]*H+kernel_y,channels])
    conv_A=tf.nn.conv2d(graph_A,b,strides=[1,strideY,strideX,1],padding=padding)
    print('conv_A____________________________________________________\n',conv_A)
    graph_B=tf.slice(a,[0,0,List[1]*W+4,0],[-1,List[0]*H+kernel_x,List[3]-8,channels])
    conv_B=tf.nn.conv2d(graph_B,b,strides=[1,strideY,strideX,1],padding=padding)
    print('graph_B____________________________________________________\n',graph_B)
    print('conv_B____________________________________________________\n',conv_B)
    #graph_C=tf.slice(a,[0,List[0]*W+width-kernel_x,List[1]*H+4,0],[-1,width-(List[0]*W+List[2])+kernel_x,List[3]-8,channels])
    graph_C=tf.slice(a,[0,List[0]*W+List[2]-kernel_x,List[1]*H+4,0],[-1,-1,List[3]-8,channels])
    conv_C=tf.nn.conv2d(graph_C,b,strides=[1,strideY,strideX,1],padding=padding)
    print('graph_C____________________________________________________\n',graph_C)
    print('conv_C____________________________________________________\n',conv_C)
    #graph_D=tf.slice(a,[0,0,List[1]*W+List[3]-kernel_y,0],[-1,width,height-List[3]-List[1]*H+kernel_y,channels])
    graph_D=tf.slice(a,[0,0,List[1]*H+List[3]-kernel_y,0],[-1,width,-1,channels])
    conv_D=tf.nn.conv2d(graph_D,b,strides=[1,strideY,strideX,1],padding=padding)
    print('conv_D____________________________________________________\n',conv_D)
    print('previous____________________________________________________\n',previous_feature)
    conv_Center=tf.slice(previous_feature,[0,tf.cast((List[0]*W+List[4]*W)/strideX+1,tf.int32),tf.cast((List[1]*H+List[5]*H)/strideY+1,tf.int32),0],[-1,tf.cast(1+(List[2]-kernel_x)/strideX-2,tf.int32),tf.cast(1+(List[3]-kernel_y)/strideY-2,tf.int32),out_channels])
    #conv_Center=tf.slice(previous_feature,[0,tf.cast((List[0]*W+List[4]*W)/strideX+1,tf.int32),tf.cast((List[1]*H+List[5]*H)/strideY+1,tf.int32),0],[-1,-1,tf.cast(1+(List[3]-kernel_y)/strideY-2,tf.int32),out_channels])
    print('conv_Center____________________________________________________\n',conv_Center)
    middle=tf.concat([conv_B,conv_Center,conv_C],axis=1)
    result=tf.concat([conv_A,middle,conv_D],axis=2)
    return result

def conv_d(a,b,List,strideX,strideY,padding,previous_feature):
    print("in conv_d==========================================================================================")
    batch_size=a.get_shape()[0]    # width
    width=int(a.get_shape()[1])    # width
    print('width:++++++++++++++++++++++++++++++++++++++\n',width)
    height=int(a.get_shape()[2])   # height
    kernel_x=int(b.get_shape()[1])
    kernel_y=int(b.get_shape()[0])
    channels=int(b.get_shape()[2])
    out_channels=int(b.get_shape()[3])
    #Lis_len=int(List.get_shape()[0])
    pict_n=List[0]      # the blank we operate
    #List.remove(pict_n) # delete the blank of the list
    # condition
    # all in picture
    #List=List[1:Lis_len-1]
    print('x_coor:',List[0],'y_coor:',List[1])
    print('b:',b)
    # split the graph and compute the convolution
    graph_B=tf.slice(a,[0,List[2]-kernel_x,0,0],[-1,240-List[2]+kernel_x,-1,channels])
    conv_B=tf.nn.conv2d(graph_B,b,strides=[1,strideY,strideX,1],padding=padding)
    print('graph_B____________________________________________________\n',graph_B)
    print('conv_B____________________________________________________\n',conv_B)
    print('previous____________________________________________________\n',previous_feature)
    conv_Center=tf.slice(previous_feature,[0,0,0,0],[-1,tf.cast(1+(List[2]-kernel_x)/strideX-2,tf.int32),-1,out_channels])
    #conv_Center=tf.slice(previous_feature,[0,tf.cast((List[0]*W+List[4]*W)/strideX+1,tf.int32),tf.cast((List[1]*H+List[5]*H)/strideY+1,tf.int32),0],[-1,-1,tf.cast(1+(List[3]-kernel_y)/strideY-2,tf.int32),out_channels])
    print('conv_Center____________________________________________________\n',conv_Center)
    result=tf.concat([conv_Center,conv_B],axis=1)
    return result


def convLayer_c(x,kHeight,kWidth,strideX,strideY,featureNum,name,padding="SAME",groups=1,mode=0,previous_feature=None,Blank=None):
    channel=int(x.get_shape()[-1])
    conv=lambda a,b:tf.nn.conv2d(a,b,strides=[1,strideY,strideX,1],padding=padding)
    with tf.variable_scope(name) as scope:
        w= tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b= tf.get_variable("b",shape=[featureNum])
        mergeFeatureMap=tf.cond(tf.equal(mode,1),lambda: conv_d(x,w,Blank,4,4,padding,previous_feature),lambda: conv(x,w))
        #mergeFeatureMap=conv_c(x,w,tf.constant([0,0,200,200,1,1]),4,4,padding,previous_feature)

        out=tf.nn.bias_add(mergeFeatureMap,b)
        print(name,'x----------------------',x.get_shape())
        print(name,'w----------------------',w.get_shape())
        print(name,'shape----------------------',mergeFeatureMap.get_shape())
        print(name,'out----------------------',out)
        #return tf.nn.relu(tf.reshape(out,mergeFeatureMap.get_shape()),name=scope.name)
        return tf.nn.relu(out,name=scope.name)

def convLayer_cc(x,kHeight,kWidth,strideX,strideY,featureNum,name,padding="SAME",groups=1,mode=0,previous_feature=None,Blank=None):
    channel=int(x.get_shape()[-1])
    conv=lambda a,b:tf.nn.conv2d(a,b,strides=[1,strideY,strideX,1],padding=padding,name="CONVCC_in")
    with tf.variable_scope(name) as scope:
        w= tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b= tf.get_variable("b",shape=[featureNum])
        logit_feature=tf.less(tf.ones([1,57,77,512]),tf.zeros([1,57,77,512]))
        mergeFeatureMap=tf.where(logit_feature,conv(x,w),previous_feature)
        #mergeFeatureMap=conv_c(x,w,tf.constant([0,0,200,200,1,1]),4,4,padding,previous_feature)

        out=tf.nn.bias_add(mergeFeatureMap,b)
        print(name,'x----------------------',x.get_shape())
        print(name,'w----------------------',w.get_shape())
        print(name,'shape----------------------',mergeFeatureMap.get_shape())
        print(name,'out----------------------',out)
        #return tf.nn.relu(tf.reshape(out,mergeFeatureMap.get_shape()),name=scope.name)
        return tf.nn.relu(out,name=scope.name)
    '''
    
def convLayer_c2(x,kHeight,kWidth,strideX,strideY,featureNum,name,padding="SAME",groups=1,mode=0,previous_feature=None,Blank=None):
    channel=int(x.get_shape()[-1])
    conv=lambda a,b:tf.nn.conv2d(a,b,strides=[1,strideY,strideX,1],padding=padding)
    with tf.variable_scope(name) as scope:
        w= tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b= tf.get_variable("b",shape=[featureNum])
        mergeFeatureMap=conv_c2(x,w,Blank,4,4,padding,previous_feature)
        #mergeFeatureMap=conv_c(x,w,tf.constant([0,0,200,200,1,1]),4,4,padding,previous_feature)

        out=tf.nn.bias_add(mergeFeatureMap,b)
        print(name,'x----------------------',x.get_shape())
        print(name,'w----------------------',w.get_shape())
        print(name,'shape----------------------',mergeFeatureMap.get_shape())
        print(name,'out----------------------',out)
        #return tf.nn.relu(tf.reshape(out,mergeFeatureMap.get_shape()),name=scope.name)
        return tf.nn.relu(out,name=scope.name)
    '''

def convLayer(x,kHeight,kWidth,strideX,strideY,featureNum,name,padding="SAME",groups=1,mode=0,previous_feature=None,Blank=None):
    channel=int(x.get_shape()[-1])
    conv=lambda a,b:tf.nn.conv2d(a,b,strides=[1,strideY,strideX,1],padding=padding)
    with tf.variable_scope(name) as scope:
        w= tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b= tf.get_variable("b",shape=[featureNum])
        mergeFeatureMap=conv(x,w)

        out=tf.nn.bias_add(mergeFeatureMap,b)
        print(name,'x----------------------',x.get_shape())
        print(name,'w----------------------',w.get_shape())
        print(name,'shape----------------------',mergeFeatureMap.get_shape())
        print(name,'out----------------------',out)
        #return tf.nn.relu(tf.reshape(out,mergeFeatureMap.get_shape()),name=scope.name)
        return tf.nn.relu(out,name=scope.name)
"""build model"""
#x,yy=ID.get_next_batch(100)
X=tf.placeholder("float",[None,240,320,3])
Y=tf.placeholder("float",[None,10])
template=tf.placeholder("float",[240,320,3])
cache_1=tf.placeholder("float",[None,57,77,layer1_num])
keep_prob=tf.placeholder(tf.float32)
mode=tf.placeholder(tf.int32)
blank=tf.placeholder('int32',[6])
#blank=tf.cast(tf.py_func(cu.ComputeOverlay,[X[0],template],tf.int64),tf.int32)

blank1=tf.cast(tf.py_func(compute,[X[0],template],tf.int64),tf.int32)
print('blank----------------------------\n',blank.get_shape())
#conv0=convLayer(X, 16,16, 4, 4, 412, "conv0", "VALID",Blank=blank,mode=mode,previous_feature=cache_1)
conv1 = convLayer_c(X, 16,16, 4, 4, layer1_num, "conv1", "VALID",Blank=blank,mode=mode,previous_feature=cache_1)
print('conv1_+_+_+_+_+_+_+_+++_+_+_+_+_+_+_+:',conv1)
cache_conv=conv1
pool1 = maxPoolLayer(conv1, 3, 3, 2, 2, "pool1", "VALID")
print('pool1:',pool1)
conv2 = convLayer(pool1,6,6, 1, 1, 256, "conv2")
pool2 = maxPoolLayer(conv2, 3, 3, 2, 2, "pool2", "VALID")
print('conv222',conv2)
conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

conv4 = convLayer(conv3, 4,3, 1, 1, 384, "conv4")

conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5")

pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

fcIn = tf.reshape(pool5, [-1, 256 *6 * 8])

fc1 = fcLayer(fcIn, 256 * 6*8, 4096, True, "fc6")

dropout1 = dropout(fc1, keep_prob)


fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")

dropout2 = dropout(fc2, keep_prob)


w3=tf.get_variable("w3",shape=[4096,10],dtype="float")
b3=tf.get_variable("b3",[10],dtype="float")

result=tf.nn.softmax(tf.matmul(dropout2, w3) + b3)


#fc3 = fcLayer(dropout2, 4096, 10, True, "fc8")
#result=tf.nn.softmax(fc3)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=result))
train_step=tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
grad_w3,grad_b3=tf.gradients(xs=[w3,b3],ys=cross_entropy)
correct_prediction=tf.equal(tf.argmax(result,1),tf.argmax(Y,1))
pred=tf.argmax(result,1)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
init=tf.global_variables_initializer()

writer=tf.summary.FileWriter('./path/cachelogs',tf.get_default_graph())

saver=tf.train.Saver()
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    mnist_profiler = model_analyzer.Profiler(graph=sess.graph)
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(init)

    saver.restore(sess,'tmp/model.ckpt')
    '''
    loss_sum1=0
    loss_sum2=1000
    for i in range(40000):
        batch_x,batch_y=ID.get_next_batch(batch_size)
        sess.run(train_step,feed_dict={X:batch_x,Y:batch_y,keep_prob:1})
        loss=sess.run(cross_entropy,feed_dict={X:batch_x,Y:batch_y,keep_prob:1})
        loss_sum1+=loss
        if i%50==0:
            train_loss=sess.run(cross_entropy,feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0})
            train_accuracy=accuracy.eval(feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0})
            print ("step %d, training loss %g, accuracy %g "%(i,train_loss,train_accuracy))
            print ('gradient\nw3:\n',sess.run(grad_w3,feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0}),'\nb3:\n',sess.run(grad_b3,feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0}))
            saver.save(sess,'tmp/model.ckpt')
            if loss_sum1<=loss_sum2:
                learning_rate=learning_rate/2314             template_x=batch_x[0]

                #train_step=tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
            loss_sum2=loss_sum1
            loss_sum1=0
    '''
    blank_x=np.array([0,0,W,H,0,0])
    cache1=np.random.rand(1,57,77, layer1_num)
    cache2=np.random.rand(1,28,38, 256)
    batch_size=1
    template_x=None
    template_y=None
    theshow=0
    for i in range(20):
        if i%10==0:
            ID.now_start=12000
            time_start=time.time()
            batch_x,batch_y=ID.get_next_batch(batch_size)
            template_x=batch_x[0]
            #template_y=batch_y
            example=batch_x.reshape([240,320,3])
            #res=sess.run([cache_conv,pred],feed_dict={X:batch_x,Y:batch_y,keep_prob:1,mode:0,cache_1:cache1,template:template_x,blank:[0,0,20,20,0,0]},options=run_options, run_metadata=run_metadata)
            res=sess.run([cache_conv,pred],feed_dict={X:batch_x,Y:batch_y,keep_prob:1,mode:0,cache_1:cache1,template:template_x,blank:[0,0,20,20,0,0]},options=run_options, run_metadata=run_metadata)
            mnist_profiler.add_step(step=i, run_meta=run_metadata)
            predict=res[1]
            cache1=res[0]
            time_end=time.time()
            print(predict)
            print('[original*]time cost conv:',time_end-time_start)
        else:
            ID.now_start=12001
            time_start=time.time()
            batch_x,batch_y=ID.get_next_batch(batch_size)
            batch_test=np.zeros([batch_size,240,320,3])
            item1=cu.ComputeOverlay(batch_x[0],example) 
            item=compute(batch_x[0],template_x)
            print('item:',item)
            print('item1:',item1)
            print('compute:',compute(batch_x[0],template_x))
            convvvv,blank_x,res=sess.run([conv1,blank,pred],feed_dict={X:batch_test,Y:batch_y,keep_prob:1,mode:1,cache_1:cache1,template:template_x,blank:[0,0,240,320,0,0]},options=run_options, run_metadata=run_metadata)
            writer.add_run_metadata(run_metadata, 'step %d' % i)
            mnist_profiler.add_step(step=i, run_meta=run_metadata)
            predict=res
            print(predict)
            print('blank:',blank_x)
            time_end=time.time()
            print('[changed*]time cost conv:',time_end-time_start)

    print(theshow)
    profile_graph_opts_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
    #profile_graph_opts_builder.with_timeline_output(timeline_file='./tmp_tf/time_profiler.json')

    #profile_graph_opts_builder.with_timeline_output(timeline_file='./tmp_tf/accumulate_profiler.json')
    profile_graph_opts_builder.with_timeline_output(timeline_file='./tmp_tf/test_profiler.json')
    #profile_graph_opts_builder.with_timeline_output(timeline_file='./tmp_tf/origi_profiler.json')
    mnist_profiler.profile_graph(profile_graph_opts_builder.build())
    writer.close()

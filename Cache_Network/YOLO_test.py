import numpy as np
import cv2
import tensorflow as tf
import time
import sys
import os
import pdb
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
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

class YOLO_TF:
	fromfile = None
	tofile_img = 'test/output.jpg'
	tofile_txt = 'test/output.txt'
	imshow = False
	filewrite_img = False
	filewrite_txt = False
	disp_console = True
	weights_file = 'weights/YOLO_tiny.ckpt'
	alpha = 0.1
	threshold = 0.2
	iou_threshold = 0.5
	num_class = 20
	num_box = 2
	grid_size = 7
	classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

	w_img = 640
	h_img = 480

	def __init__(self,argvs = []):
		self.detected = 0
		self.overall_pics = 0
		self.argv_parser(argvs)
		self.build_networks()
		if self.fromfile is not None: self.detect_from_file(self.fromfile)
		print(self.fromfolder)
		print('init------------------------------------------------------------------------------')
		if self.fromfolder is not None:
			filename_list = os.listdir(self.fromfolder)
			for filename in filename_list:
				print("Pics number:",self.overall_pics)
				self.overall_pics+=1
				self.detect_from_file(self.fromfolder+"/"+filename)
			print("Accuracy:", self.detected/self.overall_pics)

	def argv_parser(self,argvs):
		for i in range(1,len(argvs),2):
			if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
			if argvs[i] == '-fromfolder' : 
				self.fromfolder = argvs[i+1]
			else:
				self.fromfolder = None
			if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
			if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
			if argvs[i] == '-imshow' :
				if argvs[i+1] == '1' :self.imshow = True
				else : self.imshow = False
			if argvs[i] == '-disp_console' :
				if argvs[i+1] == '1' :self.disp_console = True
				else : self.disp_console = False
				
	def build_networks(self):
		if self.disp_console : print ("Building YOLO_tiny graph...")
		self.x = tf.placeholder('float32',[None,448,448,3])
		self.conv_1 = self.conv_layer(1,self.x,16,3,1)
		self.pool_2 = self.pooling_layer(2,self.conv_1,2,2)
		self.conv_3 = self.conv_layer(3,self.pool_2,32,3,1)
		self.pool_4 = self.pooling_layer(4,self.conv_3,2,2)
		self.conv_5 = self.conv_layer(5,self.pool_4,64,3,1)
		self.pool_6 = self.pooling_layer(6,self.conv_5,2,2)
		self.conv_7 = self.conv_layer(7,self.pool_6,128,3,1)
		self.pool_8 = self.pooling_layer(8,self.conv_7,2,2)
		self.conv_9 = self.conv_layer(9,self.pool_8,256,3,1)
		self.pool_10 = self.pooling_layer(10,self.conv_9,2,2)
		self.conv_11 = self.conv_layer(11,self.pool_10,512,3,1)
		self.pool_12 = self.pooling_layer(12,self.conv_11,2,2)
		self.conv_13 = self.conv_layer(13,self.pool_12,1024,3,1)
		self.conv_14 = self.conv_layer(14,self.conv_13,1024,3,1)
		self.conv_15 = self.conv_layer(15,self.conv_14,1024,3,1)
		self.fc_16 = self.fc_layer(16,self.conv_15,256,flat=True,linear=False)
		self.fc_17 = self.fc_layer(17,self.fc_16,4096,flat=False,linear=False)
		#skip dropout_18
		self.fc_19 = self.fc_layer(19,self.fc_17,1470,flat=False,linear=True)
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		self.saver = tf.train.Saver()
		self.writer=tf.summary.FileWriter('./cachelogs',tf.get_default_graph())
		self.profiler=model_analyzer.Profiler(graph=self.sess.graph)
		self.run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		self.run_metadata = tf.RunMetadata()
		#self.saver.restore(self.sess,self.weights_file)
		if self.disp_console : print( "Loading complete!" + '\n')

	def conv_layer(self,idx,inputs,filters,size,stride):
		print("in conv------------------------------------------------------------------")
		channels = inputs.get_shape()[3]
		weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=[filters]))

		pad_size = size//2
		pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
		inputs_pad = tf.pad(inputs,pad_mat)

		conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	
		conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')	
		if self.disp_console : print ('    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))
		return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

	def pooling_layer(self,idx,inputs,size,stride):
		if self.disp_console : print ('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))
		return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

	def fc_layer(self,idx,inputs,hiddens,flat = False,linear = False):
		input_shape = inputs.get_shape().as_list()		
		if flat:
			dim = input_shape[1]*input_shape[2]*input_shape[3]
			inputs_transposed = tf.transpose(inputs,(0,3,1,2))
			inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
		else:
			dim = input_shape[1]
			inputs_processed = inputs
		weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))	
		if self.disp_console : print ('    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))	)
		if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
		ip = tf.add(tf.matmul(inputs_processed,weight),biases)
		return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

	def detect_from_cvmat(self,img):
		s = time.time()
		i=123
		self.h_img,self.w_img,_ = img.shape
		img_resized = cv2.resize(img, (448, 448))
		img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
		img_resized_np = np.asarray( img_RGB )
		inputs = np.zeros((1,448,448,3),dtype='float32')
		inputs[0] = (img_resized_np/255.0)*2.0-1.0
		in_dict = {self.x: inputs}
		print("detect frome cvmat------=======================================================================")
		net_output = self.sess.run(self.fc_19,feed_dict=in_dict,options=self.run_options,run_metadata=self.run_metadata)
		self.writer.add_run_metadata(self.run_metadata,"12345")
		self.profiler.add_step(step=i,run_meta=self.run_metadata)
		profile_graph_opts_builder = option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
		profile_graph_opts_builder.with_timeline_output(timeline_file='./tmp_tf/Yolo_profiler.json')
		self.profiler.profile_graph(profile_graph_opts_builder.build())
		self.writer.close()
		self.result = self.interpret_output(net_output[0])
		self.show_results(img,self.result)
		strtime = str(time.time()-s)
		if self.disp_console : print( 'Elapsed time : ' + strtime + ' secs' + '\n')

	def detect_from_file(self,filename):
		if self.disp_console : print( 'Detect from ' + filename)
		img = cv2.imread(filename)
		#img = misc.imread(filename)
		self.detect_from_cvmat(img)

	def detect_from_crop_sample(self):
		self.w_img = 640
		self.h_img = 420
		f = np.array(open('person_crop.txt','r').readlines(),dtype='float32')
		inputs = np.zeros((1,448,448,3),dtype='float32')
		for c in range(3):
			for y in range(448):
				for x in range(448):
					inputs[0,y,x,c] = f[c*448*448+y*448+x]

		in_dict = {self.x: inputs}
		net_output = self.sess.run(self.fc_19,feed_dict=in_dict)
		self.boxes, self.probs = self.interpret_output(net_output[0])
		img = cv2.imread('person.jpg')
		self.show_results(self.boxes,img)

	def interpret_output(self,output):
		probs = np.zeros((7,7,2,20))
		class_probs = np.reshape(output[0:980],(7,7,20))
		scales = np.reshape(output[980:1078],(7,7,2))
		boxes = np.reshape(output[1078:],(7,7,2,4))
		offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

		boxes[:,:,:,0] += offset
		boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
		boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
		boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
		boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
		
		boxes[:,:,:,0] *= self.w_img
		boxes[:,:,:,1] *= self.h_img
		boxes[:,:,:,2] *= self.w_img
		boxes[:,:,:,3] *= self.h_img

		for i in range(2):
			for j in range(20):
				probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

		filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs)
		boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]
		
		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0 : continue
			for j in range(i+1,len(boxes_filtered)):
				if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
					probs_filtered[j] = 0.0
		
		filter_iou = np.array(probs_filtered>0.0,dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]

		result = []
		for i in range(len(boxes_filtered)):
			result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

		return result

	def show_results(self,img,results):
		img_cp = img.copy()
		if self.filewrite_txt :
			ftxt = open(self.tofile_txt,'w')
		class_results_set = set()
		for i in range(len(results)):
			x = int(results[i][1])
			y = int(results[i][2])
			w = int(results[i][3])//2
			h = int(results[i][4])//2
			class_results_set.add(results[i][0])
			if self.disp_console : print ('    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5]))
			if self.filewrite_img or self.imshow:
				cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
				cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
				cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
			if self.filewrite_txt :				
				ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
		if "person" in class_results_set:
			self.detected+=1
			# new_img_path=self.fromfolder[:-14]+"test7/selected_ImageNet_person/"+str(self.detected)+"_white_margin_orgin_pic.jpg"
			# cv2.imwrite(new_img_path,img_cp)
		if self.filewrite_img : 
			if self.disp_console : print( '    image file writed : ' + self.tofile_img)
			is_saved = cv2.imwrite(self.tofile_img,img_cp)
			if is_saved == True:
				print("Saved under:",self.tofile_img)
			else:
				print("Saving error!s")
		if self.imshow :
			cv2.imshow('YOLO_tiny detection',img_cp)
			cv2.waitKey(1)
		if self.filewrite_txt : 
			if self.disp_console : print( '    txt file writed : ' + self.tofile_txt)
			ftxt.close()
		self.writer.close()

	def iou(self,box1,box2):
		tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
		lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
		if tb < 0 or lr < 0 : intersection = 0
		else : intersection =  tb*lr
		return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

	def training(self): #TODO add training function!
		return None

	
			

def main(argvs):
	yolo = YOLO_TF(argvs)
	cv2.waitKey(1000)


if __name__=='__main__':	
	main(sys.argv)

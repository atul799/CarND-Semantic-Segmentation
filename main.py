#%%

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
#to estimate performance use time module
import time

import csv

#import cv2 # ami doen't have cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip #carnd ami doesn't have moviepy pip install moviepy
#%%

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


#%%
###GLOBAL VARS (move to __main__ section###################
num_classes = 2
batch_size = 8
#epochs =1
#epochs=10
#epochs = 50
#epochs =60
epochs=100

#175 epochs in paper
#epochs=175
################################
#%%
class ProcessImage:
    def __init__(self, sess, logits, keep_prob, image_pl, image_shape):
        self.sess = sess
        self.logits = logits
        self.keep_prob = keep_prob
        self.image_pl = image_pl
        self.image_shape = image_shape
    def __call__(self, image):
        print("IM SHAPE",image.shape)
        im_out=helper.pipeline(self.sess, self.logits, self.keep_prob, self.image_pl, image, self.image_shape)
        #return image
        return im_out




#%% 

#mean iou function
def mean_iou(ground_truth, prediction, num_classes):
    # TODO: Use `tf.metrics.mean_iou` to compute the mean IoU.
    #iou, iou_op = None
    iou, iou_op =tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    return iou, iou_op



#%%
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #load vgg model file
    #tf.saved_model.loader.load(sess,[vgg_tag],vgg_tag)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    #create graph instance
    graph=tf.get_default_graph()
    
    #print("Trainable variables in VGG")
    print(tf.trainable_variables())
    
    #collect layers
    
    #input layer
    l1=graph.get_tensor_by_name(vgg_input_tensor_name)
    
    #keep prob layer (dropout)
    keep=graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    
    #layer 3
    l3=graph.get_tensor_by_name(vgg_layer3_out_tensor_name) 
    
    #layer 4
    l4=graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    
    #output layer 7
    l7=graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    #print stats on layers
    print("Layer 1 Shape",l1.get_shape())
    #print("Layer keep_prob Shape",keep.get_shape())
    #print("Stats on layer 3")
    #tf.Print(l3, [tf.shape(l3)])
    print("Layer 3 Shape",l3.get_shape())
    
    #print("Stats on layer 4")
    #tf.Print(l4, [tf.shape(l4)])
    print("Layer 4 Shape",l4.get_shape())
    #print("Stats on layer 7")
    #tf.Print(l7, [tf.shape(l7)])
    print("Layer 7 Shape",l7.get_shape())
    
    return l1, keep, l3, l4, l7

#move to __main__ section or run()
#from project_tests.py wirh decorator test_safe

tests.test_load_vgg(load_vgg, tf)
print("load vgg test passed")
#%%


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    #regularize weight and initialize with trunc_normal
    w_reg = tf.contrib.layers.l2_regularizer(1e-3)
    w_init = tf.truncated_normal_initializer(stddev=0.1)
    
    
    #1x1 convolution on layer 7 (output) of vgg
    # kernel size 1, stride 1 l2 regularization 
    conv7_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                 num_classes,
                                 1,
                                 strides=(1,1),
                                 padding='same',
                                 kernel_regularizer=w_reg,
                                 kernel_initializer=w_init,
                                 name='conv7_1x1')
    
    #print("Stats on FCN8")
    #tf.Print(conv7_1x1, [tf.shape(conv7_1x1)])
    print("Layer con7_1x1 Shape",conv7_1x1.get_shape())
    
    
    #transpose convolution-->upsample by 2 (kernel size 4x4)
    output_7 = tf.layers.conv2d_transpose(conv7_1x1,
                                          num_classes,
                                          4,
                                          strides=(2,2),
                                          padding='same',
                                          kernel_regularizer=w_reg,
                                          kernel_initializer=w_init,
                                          name='decoder_L1_transpose')
    
    #tf.Print(output_7, [tf.shape(output_7)])
    print("Layer output_7 Shape",output_7.get_shape())
    #1x1 on layer 4
    
    conv4_1x1 = tf.layers.conv2d(vgg_layer4_out,
                num_classes,
                1,
                strides=(1,1),
                padding="same",
                kernel_regularizer=w_reg,
                kernel_initializer=w_init,
                name='conv4_1x1')
    
    
    #tf.Print(vgg_layer4_out, [tf.shape(conv4_1x1)])
    print("Layer conv4_1x1 Shape",conv4_1x1.get_shape())
    
    
    #add layer 4 of vgg(encoder) to layer 7 of decoder (skip connection-->Resnet)
    input_4 = tf.add(output_7, 
                     conv4_1x1,
                     name='decoder_L2_skip')
    #tf.Print(input_4, [tf.shape(input_4)])
    print("Layer input_4 Shape",input_4.get_shape())
    
    #upsample by 2 (kernel 4x4)
    output_4 = tf.layers.conv2d_transpose(input_4,
                                          num_classes,
                                          4,
                                          strides=(2,2),
                                          padding="same",
                                          kernel_regularizer=w_reg,
                                          kernel_initializer=w_init,
                                          name='decoder_L3_transpose')
    
    print("Layer output_4 Shape",output_4.get_shape())
    #1x1 on layer 3
    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out,
                             num_classes,
                             1,
                             strides=(1,1),
                             padding="same",
                             kernel_regularizer=w_reg,
                             kernel_initializer=w_init,
                             name='conv3_1x1')
    #tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)])
    print("Layer conv3_1x1 Shape",conv3_1x1.get_shape())
    
    ##add layer 3 of vgg(encoder) to layer 4 of decoder (skip connection-->Resnet)
    input_3 = tf.add(output_4, 
                     conv3_1x1,
                     name='decoder_L4_skip')
    print("Layer input_3 Shape",input_3.get_shape())
    ## upsample by 8
    output_3 = tf.layers.conv2d_transpose(input_3,
                                         num_classes,
                                         16,
                                         strides=(8,8),
                                         padding="same",
                                         kernel_regularizer=w_reg,
                                         kernel_initializer=w_init,
                                         name='decoder_L5_transpose')

    print("Layer output_3 Shape",output_3.get_shape())

    #tf.Print(output_3, [tf.shape(output_3)])
    
    return output_3


#move to run() or __main__
tests.test_layers(layers)
print("layers test passed")
#%%
def optimize(nn_last_layer, correct_label, learning_rate, num_classes,iou_f=False):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :param iou_f, is a flag to turn on/off iou metric
    :return: Tuple of (logits, train_op, cross_entropy_loss) or 
    :(logits, train_op, cross_entropy_loss,iou,iou_op) if iou_f is True
    """
    # Implement function
    
    #print("IOU_F",iou_f)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    if iou_f:
        prediction = tf.argmax(nn_last_layer, axis=3)
        #road
        #ground_truth = correct_label[:,:,:,1]
        #not road
        ground_truth = correct_label[:,:,:,0]
        #iou, iou_op = mean_iou(ground_truth, prediction, num_classes)
        iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
        return logits, train_op, cross_entropy_loss,iou,iou_op
    else:
        
        return logits, train_op, cross_entropy_loss
    


##move to run() or __main__
tests.test_optimize(optimize)
print("optimize test passed")
#%%

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,iou=None,iou_op=None,save_trg=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param iou is iou accuracy
    :param iou_op is iou operation
    :param save_trg is an object of tf saver method
    """
    # TODO: Implement function
    
    #epoch training start time
    ep_start_time=0
    #epoch training end time
    ep_end_time=0
    
    
    #list of loss values over all the batches
    loss_list=[]
    
    #list of accuracy per batch
    acc_list=[]
    
    #iou over batches in an epoch
    mean_iou=[]
    
    #loss over batches in an epoch
    mean_loss=[]
    
    #learning rate
    lr= 1e-4 #according to paper
    #keep prob
    #kb=0.5
    kb=0.75
    #iterate over nr of epochs
    for epoch in range(epochs):
        #start timer
        ep_start_time=time.time()
        
        #total loss this epoch
        ep_loss=0
        
        #iou accuracy this epoch
        ep_acc=0
        
        #num of image per epoch counter, used for avg_iou acc per epoch
        image_count=0
        
        #batch counter
        batch_ctr=0
        
        #get batch
        #print("BATCH_SIZE",batch_size)
        for image,label in get_batches_fn(batch_size):
            #run tf session on optimizer and cross entropy
            #to get acc and loss
            
            #capture batch processing time
            batch_start_time=time.time()
            
            #just a test
            loss=-10
            #run_b=True
            #while(run_b):
            ll, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={
                               input_image:image,
                               correct_label:label,
                               keep_prob:kb,
                               learning_rate:lr
                               })
            #
            #append this loss to loss_list        
            loss_list.append(loss)
            ep_loss += loss
            
            loss_calc_end_time=time.time()
            #print("Loss data")
            #print("Epoch: {},Batch: {}, Loss:{}, Loss end time:{} ".format(epoch,batch_ctr,loss,loss_calc_end_time-batch_start_time))
            
            #######################
            acc=-10;
            ##IOU
            if iou is not None and iou_op is not None:
                sess.run(iou_op,
                        feed_dict={
                        input_image:image,
                        correct_label:label,
                        keep_prob:kb,
                        learning_rate:lr}
                        )
                #run iou accuracy
                acc=sess.run(iou)
                acc_list.append(acc)
                #print("IoU =", acc)
                ep_acc += acc #* len(image)
            image_count += len(image)
            
            iou_calc_end_time=time.time()
            #print("Accuracy data")
            print("Epoch: {}, Batch: {} , Loss: {},Accuracy: {}, Batch time {}".format(epoch,batch_ctr,loss,acc,iou_calc_end_time-batch_start_time))
            
            #print("")
            batch_ctr +=1
        
         
        #avg_acc= ep_acc/batch_ctr       
        #avg_loss=ep_loss/batch_ctr
        ep_end_time=time.time()
        
        
        #capture iou for the epoch
        mean_iou.append(ep_acc/image_count)
        
        mean_loss.append(ep_loss/image_count)
            
        print("Epoch: {} took {} time".format(epoch,ep_end_time-ep_start_time))
        print("Mean LOSS: {}, Mean IoU: {}".format(ep_loss/batch_ctr,ep_acc/batch_ctr))
        
        #save at each epoch
        #if save_trg is not None:
        #    print("Saving model at Epoch:{}")
        #    #save the model at last epoch
        #    if epoch==epochs:
        #        save_trg.save(sess, './models/saved_model')
    
        
            
    
    print("Training complete!") 
    
            
    #save trained data at end of training        
    if save_trg is not None:
        print("Saving model at Epoch:{}".format(epochs))
        save_trg.save(sess, './models/saved_model')
        
        
    return loss_list,acc_list,mean_loss,mean_iou
#move to run() or __main__      
tests.test_train_nn(train_nn)
print("train_nn test passed")
#%%
def run():
    #num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    
    
    #correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
    correct_label  = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    #keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    
    
    
    tests.test_for_kitti_dataset(data_dir)
    

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    
    
    # Freezing Graphs
    #TensorFlow configuration object. 
    config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'

    # JIT level, this can be set to ON_1 or ON_2
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    
    

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    

    #with tf.Session(config=config) as sess:
    with tf.Session() as sess:    
        
        #save_trg=None
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #not done as publication mentioned augmentation didn't help much
        
        # TODO: Build NN using load_vgg, layers, and optimize function
        
        # use load_vgg function to return input_image,keep_prob,layer3/4/7 
        #from pretrained vgg architecture
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        
        
        
        #FCN decoder network of skip and upsampling
        fcn_output = layers(layer3, layer4, layer7, num_classes)
        
  
        #Set cross_entropy loss,logits and optimizer calculation expression using tf functions
        #optimize(nn_last_layer, correct_label, learning_rate, num_classes,iou_f=False):
        #logits, train_operation, cross_entropy_loss,iou,iou_op
        iou_f=True
        #iou_f=False
        iou=None
        iou_op=None
        if iou_f:
            logits, train_op, cross_entropy_loss,iou,iou_op = optimize(fcn_output, correct_label, learning_rate, num_classes,iou_f)
        
        else:
            logits, train_op, cross_entropy_loss = optimize(fcn_output, correct_label, learning_rate, num_classes,iou_f)
            
        #logits--> shape nrpixel_pixel x class
        #train_op --> Adamoptimizer with learning rate minimization function
        # cross_entropy_loss --> softmax_cross_entropy_loss mean on logits vs correct labels
        
        #intialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        save_trg = tf.train.Saver(max_to_keep=5)
        
        # TODO: Train NN using the train_nn function
        
        #train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
        #     correct_label, keep_prob, learning_rate,iou=None,iou_op=None,save_trg=None):
        loss_list,acc_list,mean_loss_list,mean_iou_list=train_nn(sess, epochs, batch_size, get_batches_fn, 
             train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,iou,iou_op,save_trg)
        
        ##SAVE BATCH acc/loss data
        print("batch loss_list length",len(loss_list))
        batch_file_acc_loss='batch_trg_file.txt'
        
        with open(batch_file_acc_loss,'w') as resf:
            bb="Batch,Loss,Accuracy\n"
            
            resf.write(bb)
            for i in range(len(loss_list)):
                aa=[str(i),str(loss_list[i]),str(acc_list[i])]
                aa_str=",".join(aa)
                aa_str += "\n"
                resf.write(aa_str)
        
        ##SAVE epoch mean loss/iou
        #print("epoch loss_list length",len(mean_loss_list))
        epoch_file_acc_loss='epoch_trg_file.txt'
        
        with open(epoch_file_acc_loss,'w') as resfe:
            bb="Epoch,Mean Loss,Mean Accuracy\n"
            
            resfe.write(bb)
            for i in range(len(mean_loss_list)):
                aa=[str(i),str(mean_loss_list[i]),str(mean_iou_list[i])]
                aa_str=",".join(aa)
                aa_str += "\n"
                resfe.write(aa_str)
        

        # TODO: Save inference data using helper.save_inference_samples
        img_labeling=True
        #img_labeling=False
        if(img_labeling):
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video                
        #video_labeling=False        
        video_labeling=True
        
        if (video_labeling):
           print("Starting Video Pipeline")
           vid1 = './driving.mp4'
           voutput1='./driving_annotated.mp4' 
           if os.path.isfile(voutput1):
               os.remove(voutput1) 
           video_clip = VideoFileClip(vid1) #.subclip(0,2)
           ##pipeline(sess, logits, keep_prob, image_pl, image_file, image_shape)
           processed_video = video_clip.fl_image(lambda image: helper.pipeline(image,sess, logits, keep_prob, input_image, image_shape))
           ##lambda image: change_image(image, myparam)
           processed_video.write_videofile(voutput1, audio=False)  
           #processed_video = video_clip.fl_image(ProcessImage(sess, logits, keep_prob, input_image, image_shape)) 
           #processed_video.write_videofile(voutput1, audio=False) 
#%%
if __name__ == '__main__':
    
    run()

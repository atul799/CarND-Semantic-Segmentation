#%%

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

#to estimate performance use time module
import time


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
epochs =1
#epochs=10
#epochs = 50


################################
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
    tf.Print(l3, [tf.shape(l3)])
    tf.Print(l4, [tf.shape(l4)])
    tf.Print(l7, [tf.shape(l7)])
    
    return l1, keep, l3, l4, l7

#move to __main__ section or run()
#from project_tests.py wirh decorator test_safe
tests.test_load_vgg(load_vgg, tf)

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
                                 kernel_regularizer=w_reg, kernel_initializer=w_init)
    
    tf.Print(conv7_1x1, [tf.shape(conv7_1x1)])
    #transpose convolution-->upsample by 2 (kernel size 4x4)
    output_7 = tf.layers.conv2d_transpose(conv7_1x1,
                                          num_classes,
                                          4,
                                          strides=(2,2),
                                          padding='same',
                                          kernel_regularizer=w_reg, kernel_initializer=w_init)
    
    tf.Print(output_7, [tf.shape(output_7)])
    
    #1x1 on layer 4
    
    conv4_1x1 = tf.layers.conv2d(vgg_layer4_out,
                num_classes,
                1,
                strides=(1,1),
                padding="same",
                kernel_regularizer=w_reg, kernel_initializer=w_init)
    
    
    tf.Print(vgg_layer4_out, [tf.shape(conv4_1x1)])
    
    #add layer 4 of vgg(encoder) to layer 7 of decoder (skip connection-->Resnet)
    input_4 = tf.add(output_7, conv4_1x1)
    tf.Print(input_4, [tf.shape(input_4)])
    
    
    #upsample by 2 (kernel 4x4)
    output_4 = tf.layers.conv2d_transpose(input_4,
                                          num_classes,
                                          4,
                                          strides=(2,2),
                                          padding="same",
                                          kernel_regularizer=w_reg, kernel_initializer=w_init)
    
    
    #1x1 on layer 3
    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out,
                             num_classes,
                             1,
                             strides=(1,1),
                             padding="same",
                             kernel_regularizer=w_reg, kernel_initializer=w_init)
    tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)])
    
    
    ##add layer 3 of vgg(encoder) to layer 4 of decoder (skip connection-->Resnet)
    input_3 = tf.add(output_4, conv3_1x1)
    
    ## upsample by 8
    output_3 = tf.layers.conv2d_transpose(input_3,
                                         num_classes,
                                         16,
                                         strides=(8,8),
                                         padding="same",
                                         kernel_regularizer=w_reg, kernel_initializer=w_init)



    tf.Print(output_3, [tf.shape(output_3)])
    
    return output_3


#move to run() or __main__
tests.test_layers(layers)

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
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_operation = optimizer.minimize(cross_entropy_loss)
    
    if iou_f:
        prediction = tf.argmax(nn_last_layer, axis=3)
        ground_truth = correct_label[:,:,:,0]
        iou, iou_op = mean_iou(ground_truth, prediction, num_classes)
        
        return logits, train_operation, cross_entropy_loss,iou,iou_op
    else:
        return logits, train_operation, cross_entropy_loss
    


##move to run() or __main__
tests.test_optimize(optimize)
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
    :param save_trg is an object of model saving method
    """
    # TODO: Implement function
    
    #epoch training start time
    ep_start_time=0
    #epoch training end time
    ep_end_time=0
    
    
    #list of loss values over all the epochs
    loss_list=[]
    
    #learning rate
    lr= 1e-4 #according to paper
    #keep prob
    kb=0.5
    
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
        
        #get batch
        for image,label in get_batches_fn(batch_size):
            #run tf session on optimizer and cross entropy
            #to get acc and loss
            _, loss = sess.run([train_op, cross_entropy_loss],
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
            
            acc=-10;
            ##IOU
            if iou is not None and iou_op is not None:
                sess.run(iou_op,feed_dict={
                        input_image:image,
                        correct_label:label,
                        keep_prob:kb,
                        learning_rate:lr}
                        )
                #run iou accuracy
                acc=sess.run(iou)
                
                print("Mean IoU =", acc)
                ep_acc += acc * len(image)
            image_count += len(image)
            print("Epoch: {}, Loss: {} , Accuracy: {}".format(epoch,loss,acc))
            
        
        
        avg_acc = ep_acc / image_count  
        #Epoch end time
        ep_end_time=time.time()
        print("Epoch: {} took {} time".format(epoch,ep_end_time-ep_start_time))
        
            
        print("EPOCH {} ...".format(epoch+1))
        print("Loss {}..".format(loss))
    
        
            
    
    print("Training complete!") 
    if save_trg is not None:
        save_trg.save(sess, './models/saved_model')

#move to run() or __main__      
tests.test_train_nn(train_nn)

#%%
def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
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

    with tf.Session(config=config) as sess:
        
        #intialize variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        
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
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        
        
        
        #FCN decoder network of skip and upsampling
        fcn_output = layers(layer3, layer4, layer7, num_classes)
        
        #Set cross_entropy loss,logits and optimizer calculation expression using tf functions

        logits, train_opt, cross_entropy_loss = optimize(fcn_output, correct_label, learning_rate, num_classes)
        
        #logits--> shape nrpixel_pixel x class
        #train_op --> Adamoptimizer with learning rate minimization function
        # cross_entropy_loss --> softmax_cross_entropy_loss mean on logits vs correct labels
        
        
        
        # TODO: Train NN using the train_nn function
        
        train_nn(sess, epochs, batch_size, get_batches_fn, 
             train_opt, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

#%%
if __name__ == '__main__':
    save_trg = tf.train.Saver()
    run()

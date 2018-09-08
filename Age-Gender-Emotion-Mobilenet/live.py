import visualization_utils
import cv2
import tensorflow as tf
import mobilenet_v2
import mobilenet
import numpy as np
import conv_blocks as op
import tensorflow.contrib.slim as slim
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.reset_default_graph()
sess = tf.InteractiveSession()


X = tf.placeholder(tf.float32,[None,None,None])
images = tf.expand_dims(X,axis=0)
images = tf.image.resize_images(images,[224,224])
images = tf.image.grayscale_to_rgb(images)
images = images / 128. - 1
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
    logits, endpoints = mobilenet_v2.mobilenet(images)
logits = tf.nn.relu6(logits)
emotion_logits = slim.fully_connected(logits, 7, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      weights_regularizer=slim.l2_regularizer(1e-5),
                                      scope='emo/emotion_1', reuse=False)

with tf.variable_scope("age"):
    age1=op.expanded_conv(endpoints['layer_16'],160,stride=1)
    age2=op.expanded_conv(age1,320,stride=1)
    age3=mobilenet.global_pool(op.expanded_conv(age2,1280,stride=1,kernel_size=[1, 1]))
    age_logits=slim.conv2d(age3,101, [1, 1],activation_fn=None,normalizer_fn=None,
                biases_initializer=tf.zeros_initializer(),scope='Conv2d_1c_1x1')
    age_logits = tf.squeeze(age_logits, [1, 2])

with tf.variable_scope("gender"):
    gender1=op.expanded_conv(endpoints['layer_16'],160,stride=1)
    gender2=op.expanded_conv(gender1,320,stride=1)
    gender3=mobilenet.global_pool(op.expanded_conv(gender2,1280,stride=1,kernel_size=[1, 1]))
    gender_logits=slim.conv2d(gender3,2, [1, 1],activation_fn=None,normalizer_fn=None,
                biases_initializer=tf.zeros_initializer(),scope='Conv2d_1c_1x1')
    gender_logits = tf.squeeze(gender_logits, [1, 2])

age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, "checkpoints/combine-checkpoint-mobilenet.ckpt")

detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
min_size = (20,20)
haar_scale = 1.1
min_neighbors = 3
haar_flags = 0
label_emotions = ['anger','contempt','disgust','fear','happy','sadness','surprise']
label_genders = ['female','male']

while True:
    last_time = time.time()
    ret, img = cap.read()
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = detector.detectMultiScale(input_img, haar_scale, min_neighbors, haar_flags, min_size)
    for (x,y,w,h) in detected:
        emotion_, gender_, age_ = sess.run([emotion_logits,gender_logits,age],feed_dict={X:np.expand_dims(input_img[y:y+h,x:x+w],2)})
        visualization_utils.draw_bounding_box_on_image_array(img,y,x,y+h,x+w,'YellowGreen',display_str_list=['face',
        '',label_emotions[np.argmax(emotion_[0])],'',
        label_genders[np.argmax(gender_[0])],'',
        '%f years old'%(age_[0]-9)],use_normalized_coordinates=False)
    cv2.putText(img,'%.1f FPS'%(1/(time.time() - last_time)), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    cv2.imshow("result", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

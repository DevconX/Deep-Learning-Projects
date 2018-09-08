import tensorflow as tf
import mobilenet_v2
import mobilenet
import numpy as np
import conv_blocks as op
import tensorflow.contrib.slim as slim

model_version = 1
export_model_dir = "./serving/versions"

tf.reset_default_graph()
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32,[None,None,None])
images = tf.expand_dims(X,axis=0)
images = tf.image.resize_images(images,[224,224])
images = tf.image.rgb_to_grayscale(images)
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

import os
export_path_base = export_model_dir
export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

tensor_input = tf.saved_model.utils.build_tensor_info(X)
tensor_gender_output = tf.saved_model.utils.build_tensor_info(gender_logits)
tensor_age_output = tf.saved_model.utils.build_tensor_info(age)
tensor_emotion_output = tf.saved_model.utils.build_tensor_info(emotion_logits)

prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'image': tensor_input},
                outputs={'gender': tensor_gender_output,
                        'age':tensor_age_output,
                        'emotion':tensor_emotion_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict_classes':prediction_signature,})

builder.save(as_text=True)

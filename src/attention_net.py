from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet import ResNet101
from keras import callbacks
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Add, ReLU
from keras.optimizers import Adam, SGD
import numpy as np
import os
import cv2
import itertools
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

epochs = 100
steps_per_epoch = 1500
model_dir = '/root/share/imagenet/models/18.10'


def load_attention(dataset_dir):

    im_names = os.listdir(os.path.join(dataset_dir, 'target2014'))
    im_names = itertools.cycle(im_names)
    # Add images
    for a in im_names:
        image_path = os.path.join(dataset_dir, 'train2014', a[:-4]+'.jpg')
        target_path = os.path.join(dataset_dir, 'target2014', a)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        image = np.expand_dims(image, axis=0)
        target = np.load(target_path)
        target = np.transpose(target, [1, 2, 0])
        target = np.expand_dims(target, axis=0)
        yield image, target


# create the base pre-trained model
base_model = ResNet101(input_shape=(512, 512, 3), weights='imagenet', include_top=False)
for layer in base_model.layers:
    print(layer.name)
outputs = [base_model.get_layer('conv5_block3_out').output, base_model.get_layer('conv4_block23_out').output,
           base_model.get_layer('conv3_block4_out').output, base_model.get_layer('conv2_block3_out').output]
for o in outputs:
    print(o.shape)
# add a global spatial average pooling layer
x = base_model.output
x = Concatenate()([x, outputs[0]])
print('looool', x.shape)
x = Conv2DTranspose(512, (1, 1), strides=(2, 2))(x)
x = Concatenate()([x, outputs[1]])
print(x.shape)
x = Conv2DTranspose(256, (1, 1), strides=(2, 2))(x)
x = Concatenate()([x, outputs[2]])
print(x.shape)
x = Conv2DTranspose(128, (1, 1), strides=(2, 2))(x)
x = Concatenate()([x, outputs[3]])
print(x.shape)
x = Conv2DTranspose(64, (1, 1), strides=(2, 2))(x)
print(x.shape)
# predictions = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
predictions = Conv2D(5, (3, 3), activation='relu', padding='same')(x)
print('predictions', predictions.shape)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:99]:
   layer.trainable = False
for layer in model.layers[99:]:
   layer.trainable = True

callbacks = [
    callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True, write_images=False),
    callbacks.ModelCheckpoint(model_dir+"/last.h5", verbose=0, save_weights_only=True, period=1),
]

# def my_loss(y_true, y_pred):
#    loss=K.mean(K.sum(K.square(y_true-y_pred)))
#    return loss

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')
# model.load_weights(model_dir+'/last.h5')
# train the model on the new data for a few epochs
model.fit(load_attention('/root/share/imagenet/'), epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)




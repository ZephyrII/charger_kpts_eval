from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet import ResNet101
from keras import callbacks
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.optimizers import Adam
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

batch_size = 1
epochs = 1
steps_per_epoch = 1000
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
predictions = Conv2D(5, (3, 3), activation='relu', padding='same')(x)
print('predictions', predictions.shape)


gene = load_attention('/root/share/imagenet/')
for img, tar in gene:

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_dir+'/last.h5')
    out = model.predict(img)
    attention = out
    attention = (attention + abs(np.min(attention))) / (abs(np.min(attention)) + abs(np.max(attention)))
    # attention = np.squeeze(attention.astype(np.uint8))
    print(np.max(attention))
    print(np.min(attention))
    print(attention.shape)
    attention = np.transpose(np.squeeze(attention), [2, 0, 1])
    for i in range(5):
        cv2.imshow('att', attention[i])
        cv2.imshow('img', img[0])
        k = cv2.waitKey(0)
        if k==ord('q'):
            exit(0)
    # print(out.shape)



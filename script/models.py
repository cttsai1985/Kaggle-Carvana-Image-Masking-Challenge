"""
modified by cttsai (Chia-Ta Tsai), @Sep 2017
for Kaggle Carvana Image Masking Challenge
main body refactered and seperated from /model/u_net.py in 
https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge
for more flexible unet structures
"""

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers import Activation, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.optimizers import RMSprop
from keras.regularizers import l2

from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff



#pieces
def relu(x): return Activation('relu')(x)
def bn(x): return BatchNormalization(axis=-1)(x)
def relu_bn(x): return relu(bn(x))

def pooling(x, k=2, stride=2): return MaxPooling2D((k, k), strides=(stride, stride))(x)
def concat(down, up, k=2): return concatenate([down, UpSampling2D((k, k))(up)], axis=-1)
def reverse(a): return list(reversed(a))

#conv_blocks
def conv(x, nf, k, stride=1, wd=0): 
    return Conv2D(filters=nf, kernel_size=(k, k), 
                  strides=(stride, stride), padding='same', 
                  kernel_initializer='he_uniform', kernel_regularizer=l2(wd))(x)

def conv_relu_bn(x, nf, k=3, wd=0, stride=1): 
    #bn - relu - conv
    #return conv(relu_bn(x), nf=nf, k=k, stride=stride, wd=wd)
    #conv - bn - relu
    return relu_bn(conv(x, nf=nf, k=k, stride=stride, wd=wd))
    

def create_unet_4down(input_shape=(80, 80, 3), num_classes=1, f=25):
#2 ** 4 = 16
#w0240, h0160 = 2 ** 4 * (5 *  1) * (3, 2)  = (15, 10)@center 1/8
#w0720, h0480 = 2 ** 4 * (3 *  5) * (3, 2)  = (45, 30)@center 3/8
#w1200, h0800 = 2 ** 4 * (5 ** 2) * (3, 2)  = (75, 50)@center 5/8
#w1680, h1120 = 2 ** 4 * (5 *  7) * (3, 2) = (105, 70)@center 7/8
    
    inputs = Input(shape=input_shape)


    d1 = conv_relu_bn(inputs, f, k=5, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    p = pooling(d1, k=2, stride=2)

    f*=2 
    d2 = conv_relu_bn(p, f, k=5, wd=0, stride=1)
    d2 = conv_relu_bn(d2, f, k=3, wd=0, stride=1)
    d2 = conv_relu_bn(d2, f, k=3, wd=0, stride=1)
    p = pooling(d2, k=2, stride=2)

    f*=2 
    d3 = conv_relu_bn(p, f, k=5, wd=0, stride=1)
    d3 = conv_relu_bn(d3, f, k=3, wd=0, stride=1)
    d3 = conv_relu_bn(d3, f, k=3, wd=0, stride=1)
    p = pooling(d3, k=2, stride=2)

    f*=2 
    d4 = conv_relu_bn(p, f, k=5, wd=0, stride=1)
    d4 = conv_relu_bn(d4, f, k=3, wd=0, stride=1)
    d4 = conv_relu_bn(d4, f, k=3, wd=0, stride=1)
    p = pooling(d4, k=2, stride=2)


    f*=2
    c = conv_relu_bn(p, f, k=1, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=5, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=5, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=1, wd=0, stride=1)

    f//=2 
    u4 = concat(down=d4, up=c, k=2)
    u4 = conv_relu_bn(u4, f, k=7, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=5, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=1, wd=0, stride=1)

    f//=2 
    u3 = concat(down=d3, up=u4, k=2)
    u3 = conv_relu_bn(u3, f, k=7, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=5, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=3, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=1, wd=0, stride=1)

    f//=2 
    u2 = concat(down=d2, up=u3, k=2)
    u2 = conv_relu_bn(u2, f, k=7, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=5, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=3, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=1, wd=0, stride=1)
    
    f//2
    u1 = concat(down=d1, up=u2, k=2)
    u2 = conv_relu_bn(u2, f, k=7, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=5, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=3, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=1, wd=0, stride=1)

    classify = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(u1)
    #classify = SeparableConv2D(num_classes, (7, 7), padding='same', activation='sigmoid')(u0)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    #model.summary()

    return model


def create_unet_5down(input_shape=(160, 160, 3), num_classes=1, f=25):
#2 ** 5 = 32
#w0480, h0320 = 2 ** 5 * (5 * 1) * (3, 2) = (15, 10)@center
#w1440, h0960 = 2 ** 5 * (3 * 5) * (3, 2) = (45, 30)@center  
#3,2,2,2,2,3(d14),c4,2,2,2,2,2,2(u14) 
    
    inputs = Input(shape=input_shape)


    d1 = conv_relu_bn(inputs, f, k=5, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=5, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    p = pooling(d1, k=2, stride=2)

    f*=2 
    d2 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d2 = conv_relu_bn(d2, f, k=3, wd=0, stride=1)
    p = pooling(d2, k=2, stride=2)

    f*=2 
    d3 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d3 = conv_relu_bn(d3, f, k=3, wd=0, stride=1)
    p = pooling(d3, k=2, stride=2)

    f*=2 
    d4 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d4 = conv_relu_bn(d4, f, k=3, wd=0, stride=1)
    p = pooling(d4, k=2, stride=2)
    
    f*=2 
    d5 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d5 = conv_relu_bn(d5, f, k=3, wd=0, stride=1)
    p = pooling(d5, k=2, stride=2)

    f*=2
    c = conv_relu_bn(p, f, k=1, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=1, wd=0, stride=1)

    f//2
    u5 = concat(down=d5, up=c, k=2)
    u5 = conv_relu_bn(u5, f, k=7, wd=0, stride=1)
    u5 = conv_relu_bn(u5, f, k=5, wd=0, stride=1)
    u5 = conv_relu_bn(u5, f, k=3, wd=0, stride=1)
    u5 = conv_relu_bn(u5, f, k=1, wd=0, stride=1)

    f//=2 
    u4 = concat(down=d4, up=u5, k=2)
    u4 = conv_relu_bn(u4, f, k=7, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=5, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=1, wd=0, stride=1)

    f//=2 
    u3 = concat(down=d3, up=u4, k=2)
    u3 = conv_relu_bn(u3, f, k=5, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=3, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=1, wd=0, stride=1)

    f//=2 
    u2 = concat(down=d2, up=u3, k=2)
    u2 = conv_relu_bn(u2, f, k=5, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=3, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=1, wd=0, stride=1)
    
    f//2
    u1 = concat(down=d1, up=u2, k=2)
    u1 = conv_relu_bn(u1, f, k=5, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=3, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=1, wd=0, stride=1)

    classify = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(u1)
    #classify = SeparableConv2D(num_classes, (7, 7), padding='same', activation='sigmoid')(u0)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    #model.summary()

    return model


def create_unet_6down(input_shape=(320, 320, 3), num_classes=1, f=25):
#2 ** 6 = 64
#w0960, h0640 = 2 ** 6 * (5 * 1) * (3, 2) = (30, 20)@center
    
    inputs = Input(shape=input_shape)

    d1 = conv_relu_bn(inputs, f, k=3, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    p = pooling(d1, k=2, stride=2)

    f*=2 
    d2 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d2 = conv_relu_bn(d2, f, k=3, wd=0, stride=1)
    p = pooling(d2, k=2, stride=2)

    f*=2 
    d3 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d3 = conv_relu_bn(d3, f, k=3, wd=0, stride=1)
    p = pooling(d3, k=2, stride=2)

    f*=2 
    d4 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d4 = conv_relu_bn(d4, f, k=3, wd=0, stride=1)
    p = pooling(d4, k=2, stride=2)

    f*=2 
    d5 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d5 = conv_relu_bn(d5, f, k=3, wd=0, stride=1)
    p = pooling(d5, k=2, stride=2)
    
    f*=2 
    d6 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d6 = conv_relu_bn(d6, f, k=3, wd=0, stride=1)
    p = pooling(d6, k=2, stride=2)

    f*=2
    c = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)

    f//=2 
    u6 = concat(down=d6, up=c, k=2)
    u6 = conv_relu_bn(u6, f, k=3, wd=0, stride=1)
    u6 = conv_relu_bn(u6, f, k=3, wd=0, stride=1)
    #u6 = conv_relu_bn(u6, f, k=3, wd=0, stride=1)

    f//2
    u5 = concat(down=d5, up=u6, k=2)
    u5 = conv_relu_bn(u5, f, k=3, wd=0, stride=1)
    u5 = conv_relu_bn(u5, f, k=3, wd=0, stride=1)
    #u5 = conv_relu_bn(u5, f, k=3, wd=0, stride=1)

    f//=2 
    u4 = concat(down=d4, up=u5, k=2)
    u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)
    #u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)

    f//=2 
    u3 = concat(down=d3, up=u4, k=2)
    u3 = conv_relu_bn(u3, f, k=3, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=3, wd=0, stride=1)
    #u3 = conv_relu_bn(u3, f, k=3, wd=0, stride=1)

    f//=2 
    u2 = concat(down=d2, up=u3, k=2)
    u2 = conv_relu_bn(u2, f, k=3, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=3, wd=0, stride=1)
    #u2 = conv_relu_bn(u2, f, k=3, wd=0, stride=1)
    
    f//2
    u1 = concat(down=d1, up=u2, k=2)
    u1 = conv_relu_bn(u1, f, k=3, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=3, wd=0, stride=1)
    #u1 = conv_relu_bn(u1, f, k=3, wd=0, stride=1)

    #classify = Conv2D(num_classes, (7, 7), padding='same', activation='sigmoid')(u1)
    classify = SeparableConv2D(num_classes, (3, 3), padding='same', activation='sigmoid')(u1)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    #model.summary()

    return model


def create_unet_7down(input_shape=(320, 320, 3), num_classes=1, f=8):
    
    inputs = Input(shape=input_shape)

    d1 = conv_relu_bn(inputs, f, k=5, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=5, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    d1 = conv_relu_bn(d1, f, k=3, wd=0, stride=1)
    p = pooling(d1, k=2, stride=2)

    f*=2 
    d2 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d2 = conv_relu_bn(d2, f, k=3, wd=0, stride=1)
    p = pooling(d2, k=2, stride=2)

    f*=2 
    d3 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d3 = conv_relu_bn(d3, f, k=3, wd=0, stride=1)
    p = pooling(d3, k=2, stride=2)

    f*=2 
    d4 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d4 = conv_relu_bn(d4, f, k=3, wd=0, stride=1)
    p = pooling(d4, k=2, stride=2)

    f*=2 
    d5 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d5 = conv_relu_bn(d5, f, k=3, wd=0, stride=1)
    p = pooling(d5, k=2, stride=2)
    
    f*=2 
    d6 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d6 = conv_relu_bn(d6, f, k=3, wd=0, stride=1)
    p = pooling(d6, k=2, stride=2)

    f*=2 
    d7 = conv_relu_bn(p, f, k=3, wd=0, stride=1)
    d7 = conv_relu_bn(d7, f, k=3, wd=0, stride=1)
    p = pooling(d7, k=2, stride=2)

    f*=2
    c = conv_relu_bn(p, f, k=1, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=3, wd=0, stride=1)
    c = conv_relu_bn(c, f, k=1, wd=0, stride=1)

    f//=2 
    u7 = concat(down=d7, up=c, k=2)
    u7 = conv_relu_bn(u7, f, k=7, wd=0, stride=1)
    u7 = conv_relu_bn(u7, f, k=5, wd=0, stride=1)
    u7 = conv_relu_bn(u7, f, k=3, wd=0, stride=1)
    u7 = conv_relu_bn(u7, f, k=1, wd=0, stride=1)

    f//=2 
    u6 = concat(down=d6, up=u7, k=2)
    u6 = conv_relu_bn(u6, f, k=7, wd=0, stride=1)
    u6 = conv_relu_bn(u6, f, k=5, wd=0, stride=1)
    u6 = conv_relu_bn(u6, f, k=3, wd=0, stride=1)
    u6 = conv_relu_bn(u6, f, k=1, wd=0, stride=1)

    f//2
    u5 = concat(down=d5, up=u6, k=2)
    u5 = conv_relu_bn(u5, f, k=5, wd=0, stride=1)
    u5 = conv_relu_bn(u5, f, k=3, wd=0, stride=1)
    u5 = conv_relu_bn(u5, f, k=3, wd=0, stride=1)

    f//=2 
    u4 = concat(down=d4, up=u5, k=2)
    u4 = conv_relu_bn(u4, f, k=5, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)
    u4 = conv_relu_bn(u4, f, k=3, wd=0, stride=1)

    f//=2 
    u3 = concat(down=d3, up=u4, k=2)
    u3 = conv_relu_bn(u3, f, k=5, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=3, wd=0, stride=1)
    u3 = conv_relu_bn(u3, f, k=1, wd=0, stride=1)

    f//=2 
    u2 = concat(down=d2, up=u3, k=2)
    u2 = conv_relu_bn(u2, f, k=5, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=3, wd=0, stride=1)
    u2 = conv_relu_bn(u2, f, k=1, wd=0, stride=1)
    
    f//2
    u1 = concat(down=d1, up=u2, k=2)
    u1 = conv_relu_bn(u1, f, k=5, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=3, wd=0, stride=1)
    u1 = conv_relu_bn(u1, f, k=1, wd=0, stride=1)

    classify = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(u1)
    #classify = SeparableConv2D(num_classes, (3, 3), padding='same', activation='sigmoid')(u1)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    #model.summary()

    return model

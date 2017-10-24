"""
modified by cttsai (Chia-Ta Tsai), @Sep 2017
main body refactered and re-organized from train.py and test_submit.py in 
https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge

key changes:
1) refactored,
2) flexible u_net structures,
3) stacking from low resolutions,
4) ensemble,
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#
from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

import augmentations as aug
import params_stacked as params #global variables
from utilities import run_length_encode, split_by_car_model    

###############################################################################
def read_img(img_id, width, height, path='.', augment={}):
    
    img = cv2.imread('{}/train/{}.jpg'.format(path, img_id))
    #img = cv2.resize(img, (width, height))
    mask = cv2.imread('{}/train_masks/{}_mask.png'.format(path, img_id), cv2.IMREAD_GRAYSCALE)
    #mask = cv2.resize(mask, (width, height))
    
    #augmentations        
    if augment.get('HueSatVal', False):
        img = aug.HueSaturationValue(img, u=0.5, v=np.random.random())

    if augment.get('Constrast', False):
        img = aug.Contrast(img, u=0.5, v=np.random.random())
        
    if augment.get('Shift', False):
        img, mask = aug.ShiftScaleRotate(img, mask, u=0.5, v=np.random.random())
        
    if augment.get('Flip', False):
        img, mask = aug.HorizontalFlip(img, mask, u=0.5, v=np.random.random())
    
    img = cv2.resize(img, (width, height))
    mask = cv2.resize(mask, (width, height))    
    mask = np.expand_dims(mask, axis=2)

    return img, mask


def setup_model(model, learning_rate=0.005, file_weights='', opt_resume=False):
    
    if os.path.isfile(file_weights) and opt_resume:
        file_weights = '../weights/' + file_weights
        try:
            model.load_weights(file_weights)
        except:
            pass
    else:
        model.load_weights('../weights/init_weights_w{0:04d}h{1:04d}.hdf5'.format(params.input_width, params.input_height))

    #
    tmstmp = '{}'.format(time.strftime("%Y-%m-%d-%H-%M"))
    file_weights = '../weights/weights_w{0:04d}h{1:04d}_{2}.hdf5'.format(input_width, input_height, tmstmp)

    #obj = dice_loss
    obj = bce_dice_loss #default 
    #obj = weighted_dice_loss
    #obj = weighted_bce_dice_loss
    #weighted_dice_coeff
    model.compile(optimizer=RMSprop(lr=learning_rate), loss=obj, metrics=[dice_coeff])
    #model.compile(optimizer=Adam(lr=learning_rate), loss=obj, metrics=[dice_coeff])
    ################################
    #call backs
    earlystop = EarlyStopping(monitor='val_dice_coeff', patience=8, verbose=1, min_delta=1e-4, mode='max')
    reduce_lr_coeff = ReduceLROnPlateau(monitor='val_dice_coeff', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='max')
    #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, epsilon=1e-4, mode='auto')
    model_chk = ModelCheckpoint(monitor='val_dice_coeff', filepath=file_weights, save_best_only=True, save_weights_only=True, mode='max')
    
    #callbacks = [earlystop, reduce_lr_coeff, reduce_lr_loss, model_chk, TensorBoard(log_dir='../logs')]
    callbacks = [earlystop, reduce_lr_coeff, model_chk, TensorBoard(log_dir='../logs')]
    ##########
    return model, callbacks, file_weights


def data_generator(ids_split, width, height, path='../input', augment={}):
    while True:
        for start in range(0, len(ids_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_split))
            ids_batch = ids_split[start:end]
            for img_id in ids_batch.values:
                img, mask = read_img(img_id, width=width, height=height, path=path, augment=augment)
                x_batch.append(img)
                y_batch.append(mask)
                
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            
            yield x_batch, y_batch


def predict_img(model, ids, batch_size=4, opt_test=True, opt_tta=True, opt_overwrite=True, w=0.5):
    
    width = params.input_width
    height = params.input_height
    threshold = params.threshold
    orig_width = params.orig_width
    orig_height = params.orig_height
         
    if opt_test:
        read_path = '../input/test'        
    else:
        read_path = '../input/train'
    
    save_path = '../w{0:04d}h{1:04d}'.format(width, height)
    
    #save submission    
    tmstmp = '{}'.format(time.strftime("%Y-%m-%d-%H-%M"))
    names = ['{0}.jpg'.format(i) for i in ids]
    rles = []

    print('Predicting on {} samples with batch_size = {}...'.format(len(ids), batch_size))
    for start in tqdm(range(0, len(ids), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(ids))
        
        ids_batch = ids[start:end]
        
        #read in and resize
        for i in ids_batch.values:
            img = cv2.imread('{0}/{1}.jpg'.format(read_path, i))
            img = cv2.resize(img, (width, height))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255.
        
        #prediction
        preds = model.predict_on_batch(x_batch)
        #and test time augmentations
        #input of tfs: (samples, input_size_h, input_size_w, channels)
        if opt_tta:
            x_batch_flip = np.flip(x_batch, axis=2)
            preds += np.flip(model.predict_on_batch(x_batch_flip), axis=2)
            preds *= 0.5

        preds = np.squeeze(preds, axis=3)
        for p, i in zip(preds, ids_batch):
            f = '{0}/{1}.npy'.format(save_path, i)
            if not opt_overwrite:
                try:
                    p_npy = np.load(f)
                    p = p * w + p_npy * (1.0 - w) / 255. #load npy from 0-255 scale
                except:
                    print('load {} failed'.format(f))
            np.save(f, (255. * p).astype(np.uint8))

        if opt_test:
            for p in preds:
                prob = cv2.resize(p, (orig_width, orig_height))
                mask = prob > threshold
                rle = run_length_encode(mask)
                rles.append(rle)

    #save prediction
    if opt_test:
        submit = '../submit/subm_w{0:04d}h{1:04d}_{2}.csv.gz'.format(width, height, tmstmp)
        if opt_tta:
            submit = '../submit/subm_w{0:04d}h{1:04d}_tta_{2}.csv.gz'.format(width, height, tmstmp)
            
        print('Generating submission file...', flush=True)
        #pd.DataFrame({'img': names, 'rle_mask': rles}).to_csv(submit, index=False, compression='gzip')
        df = pd.DataFrame({'img': names, 'rle_mask': rles})
        df.to_csv(submit, index=False, compression='gzip')
        

###############################################################################
if __name__ == '__main__':

    np.random.RandomState(2017)
    seed_split = 1809
    r = 0.5 #validation
    opt_group = True
    #opt_group = False
    opt_resume = False
    #opt_resume = True
    
    #input
    input_size = params.input_size
    input_width = params.input_width
    input_height = params.input_height
    print('dim_base={}, w={}, h={}'.format(input_size, input_width, input_height))

    #for training
    opt_augments = {'HueSatVal': True, 'Shift': True, 'Flip': True, 'Constrast': False}
    
    #keras
    epochs = params.max_epochs
    batch_size = params.batch_size
    print('epochs={}, batch={}'.format(epochs, batch_size))
    
    #prediction
    opt_tta = False
    opt_tta = True


    #read train and split
    df_train = pd.read_csv('../input/train_masks.csv')
    #split approaches
    if opt_group:
        ids_train_split, ids_valid_split = split_by_car_model(df_train, ratio=r, seed_split=seed_split)
   
    else:
        ids_train = df_train['img'].apply(lambda s: s.split('.')[0])
        ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=r, random_state=seed_split)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    #read test
    df_test = pd.read_csv('../input/sample_submission.csv')
    ids_test = df_test['img'].apply(lambda s: s.split('.')[0])
    print('Test on {} samples'.format(len(ids_test)))

    
    model = params.model_factory
    model.summary()
    model.save('../weights/init_weights_w{0:04d}h{1:04d}.hdf5'.format(input_width, input_height))

    #epochs = 3 #params.max_epochs
    #ids_train_split = ids_train_split[:20]
    #ids_valid_split = ids_valid_split[:20]
    #ids_test = ids_test[:50]    
    
    model, callbacks, file_weights = setup_model(model)
    ##########    
    model.fit_generator(generator=data_generator(ids_train_split, input_width, input_height, augment=opt_augments),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=data_generator(ids_valid_split, input_width, input_height, augment={}),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))

    if os.path.isfile(file_weights):
        model.load_weights(file_weights)

    print('first half')
    predict_img(model, ids=ids_valid_split, batch_size=batch_size, opt_test=False, opt_tta=True, opt_overwrite=True)
    predict_img(model, ids=ids_test, batch_size=batch_size, opt_test=True, opt_tta=True, opt_overwrite=True) 


    #second half
    ###########################################################################
    
    model, callbacks, file_weights = setup_model(model)
    ##########
    
    model.fit_generator(generator=data_generator(ids_valid_split, input_width, input_height, augment=opt_augments),
                        steps_per_epoch=np.ceil(float(len(ids_valid_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=data_generator(ids_train_split, input_width, input_height, augment={}),
                        validation_steps=np.ceil(float(len(ids_train_split)) / float(batch_size)))

    if os.path.isfile(file_weights):
        model.load_weights(file_weights)

    print('second half')
    predict_img(model, ids=ids_train_split, batch_size=batch_size, opt_test=False, opt_tta=True, opt_overwrite=True)
    predict_img(model, ids=ids_test, batch_size=batch_size, opt_test=True, opt_tta=True, opt_overwrite=False) 


    #stacked
    ###########################################################################

#df, ids_stem = strip_id_stem(df.copy())
#ids_batch = df[df.stem.isin(ids_stem_train_split)]['id'].reset_index(drop=True)#.tolist()

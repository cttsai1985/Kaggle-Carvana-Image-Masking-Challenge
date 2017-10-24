import numpy as np
import pandas as pd
from numba import jit


#################fast rle
#from https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


#split id/img
###############################################################################
from sklearn.model_selection import train_test_split

def strip_id_stem(df):
    
    df['id'] = df['img'].apply(lambda s: s.split('.')[0])
    df['stem'] = df['img'].apply(lambda s: s.split('_')[0])

    return df, df['stem'].unique() 


def split_by_car_model(df, ratio=0.5, seed_split=1):
    
    df, ids_stem = strip_id_stem(df.copy())
    
    print('{:d} unique car models'.format(len(ids_stem)))
    ids_stem_train_split, ids_stem_valid_split = train_test_split(ids_stem, test_size=ratio, random_state=seed_split)
    ids_train_split = df[df.stem.isin(ids_stem_train_split)]['id'].reset_index(drop=True)#.tolist()
    ids_valid_split = df[df.stem.isin(ids_stem_valid_split)]['id'].reset_index(drop=True)#.tolist()

    #print(ids_train_split)
    print('Split on car models to {0:d} and {1:d}'.format(len(ids_stem_train_split), len(ids_stem_valid_split)))

    return ids_train_split, ids_valid_split

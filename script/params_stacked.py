"""
modified by cttsai (Chia-Ta Tsai), @Sep 2017
for Kaggle Carvana Image Masking Challenge
main body refactered and seperated from params.py in 
https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge
for use to setting some global parameters for stacked and ensemble models
"""
from models import create_unet_4down, create_unet_5down, create_unet_6down, create_unet_7down


def get_size(i, opt_square=False, ratio=1.5):    
    if opt_square:
        return int(i), int(i)
    else:
        return int(i * ratio), int(i)


input_size = 160
input_size = 320
#input_size = 640
#input_size = 800
#input_size = 960
#input_size = 1120
#stacking_size = 800
#stacking_size = 960
stacking_size = 1120
#stacking_size = 1024
##
orig_width = 1918
orig_height = 1280
threshold = 0.5


opt_square = True
opt_square = False

input_width, input_height = get_size(input_size, opt_square=False, ratio=1.5)
print('Params: ', input_size, input_width, input_height)
stacking_width, stacking_height = get_size(stacking_size, opt_square=False, ratio=1.5)


max_epochs = 75

model_dict = {}
# 160: 2 ** 4 * 5 = 80
# 320: 2 ** 5 * 5 = 160
# 480: 2 ** 4 * 3 * 5 = 240
# 640: 2 ** 6 * 5 = 320
# 800: 2 ** 4 * 5 ** 2 = 400
# 960: 2 ** 5 * 3 * 5 = 480
#1120: 2 ** 4 * 5 * 7 
if input_size % (32 * 2) == 0:
    if input_size == 320:
        model_dict['320'] = create_unet_5down(input_shape=(input_height, input_width, 3), num_classes=1, f=16)
    if input_size == 960:
        model_dict['960'] = create_unet_5down(input_shape=(input_height, input_width, 3), num_classes=1, f=9)
elif input_size % (64 * 2) == 0:
    if input_size == 640:
        model_dict['640'] = create_unet_6down(input_shape=(input_height, input_width, 3), num_classes=1, f=10)
else:
    #model_dict['160'] = create_unet_4down(input_shape=(input_height, input_width, 3), num_classes=1, f=36)
    if input_size == 480:
        model_dict['480'] = create_unet_4down(input_shape=(input_height, input_width, 3), num_classes=1, f=36)
    #model_dict['800'] = create_unet_4down(input_shape=(input_height, input_width, 3), num_classes=1, f=36)
    #model_dict['1120'] = create_unet_4down(input_shape=(input_height, input_width, 3), num_classes=1, f=36)
        
model_factory = model_dict.get('{}'.format(input_size), None)

batch_dict = {}
batch_dict['160'] = 8
batch_dict['320'] = 6
batch_dict['480'] = 3
batch_dict['640'] = 3
batch_dict['800'] = 1
batch_dict['960'] = 1
batch_dict['1120'] = 1

batch_size = batch_dict.get('{}'.format(input_size), 1)


#stacking
###############################################################################
stacking_model_dict = {}
if stacking_size % (32 * 2) == 0:
    if stacking_size  == 960:
        stacking_model_dict['960'] = create_unet_5down(input_shape=(stacking_height, stacking_width, 4), num_classes=1, f=9)
    if stacking_size  == 1024:
        stacking_model_dict['1024'] = create_unet_7down(input_shape=(stacking_height, stacking_width, 4), num_classes=1, f=8)
else:
    if stacking_size  == 1120:
        stacking_model_dict['1120'] = create_unet_4down(input_shape=(stacking_height, stacking_width, 4), num_classes=1, f=12)
    if stacking_size  == 800:
        stacking_model_dict['800'] = create_unet_4down(input_shape=(stacking_height, stacking_width, 4), num_classes=1, f=8)
    

stacking_model_factory = stacking_model_dict.get('{}'.format(stacking_size), None)

stacking_batch_dict = {}
stacking_batch_dict['800'] = 2
stacking_batch_dict['960'] = 1
stacking_batch_dict['1120'] = 1
stacking_batch_dict['1024'] = 1
stacking_batch_size = stacking_batch_dict.get('{}'.format(input_size), 1)


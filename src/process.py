import cv2
import glob
import matplotlib.image as mpimg
import time
import numpy as np

from classifier import *
from utilities import *
from data_explorer import data_importer

from train import train

## --------------------------------------------------------
## Hyperparameters
## --------------------------------------------------------
#orient=9
#pix_per_cell=8
#cell_per_block=2

# Using parameters combo provided by reviewer
orient=11
pix_per_cell=16
cell_per_block=2

#colorspace='RGB'
#colorspace='YCrCb'
colorspace='YUV'
hog_channel='ALL'
#hog_channel=0

spatial_size=(32, 32)
hist_bins=32
hist_range=(0, 256)

# Dummy variable, will be overwritten by retrieve_features
window_size = (64, 64)

#use_hog = True
# Reduce the sample size because HOG feature are slow to compute
#sample_size=500
#test_images = glob.glob('../test_images/*.jpg')

#ind = np.random.randint(0, len(test_images))
# Read in the image
#img = mpimg.imread(test_images[0])

notcars_path = '../data/non-vehicles/Extras/extra*.png'
cars_path = '../data/vehicles/KITTI_extracted/*.png'

#notcars_path = '../data/non-vehicles/**/*.png'
#cars_path = '../data/vehicles/**/*.png'



def retrieve_features(notcars_path, cars_path,
    orient,
    pix_per_cell,
    cell_per_block,
    colorspace,
    hog_channel,
    use_color,
    use_hog,
    verbose=True):

    not_cars, cars = data_importer(notcars_path, cars_path)

    # TODO: Profiler
    FE = Feature_Extractor(orient, pix_per_cell, cell_per_block)

    # Sorry, smelly code
    global window_size
    window_size = mpimg.imread(cars[0]).shape
    window_size = (window_size[0], window_size[1])

        
    if use_color:
        params = make_color_params(spatial_size, hist_bins, hist_range)
        if verbose:
            print('Using params:', params)
            print('Training image shape', window_size)

        car_features_color = FE.extract_features(cars, colorspace, params)
        notcar_features_color = FE.extract_features(not_cars, colorspace, params)

    if use_hog:
        params = make_hog_params(hog_channel)
        if verbose:
            print('Using params:', params)
            print('Training image shape', window_size)

        car_features_hog = FE.extract_features(cars, colorspace, params)
        notcar_features_hog = FE.extract_features(not_cars, colorspace, params)
        car_features_hog = np.asarray(car_features_hog)
        notcar_features_hog = np.asarray(notcar_features_hog)

    if use_color and use_hog:
        return np.hstack((car_features_hog, car_features_color)), \
            np.hstack((notcar_features_hog, notcar_features_color))
    if use_color and not use_hog:
        return car_features_color, notcar_features_color
    if use_hog and not use_color:
        return car_features_hog, notcar_features_hog

model = {}
model['orient'] = orient
model['pix_per_cell'] = pix_per_cell
model['cell_per_block'] = cell_per_block
model['colorspace'] = colorspace
model['hog_channel'] = hog_channel
model['spatial_size'] = spatial_size
model['hist_bins'] = hist_bins
model['hist_range'] = hist_range
model['window'] = window_size

car_features_color, notcar_features_color = retrieve_features(notcars_path, cars_path, 
    orient,
    pix_per_cell,
    cell_per_block,
    colorspace,
    hog_channel,
    use_color=True,
    use_hog=False)

model_color = train(car_features_color, notcar_features_color, return_model=True)

model['svc_color'] = model_color['svc']
model['scaler_color'] = model_color['scaler']

car_features_hog, notcar_features_hog = retrieve_features(notcars_path, cars_path, 
    orient,
    pix_per_cell,
    cell_per_block,
    colorspace,
    hog_channel,
    use_color=False,
    use_hog=True)

model_hog = train(car_features_hog, notcar_features_hog, return_model=True)

model['svc_hog'] = model_hog['svc']
model['scaler_hog'] = model_hog['scaler']

import pickle
with open('./svc_pickle.p', 'wb') as f:
    pickle.dump(model, f)

if __name__ == "__main__":
    pass

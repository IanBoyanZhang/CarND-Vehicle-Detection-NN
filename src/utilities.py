import cv2
import numpy as np

def colorsp_conversion(image, cspace):
    colorsp_mapping = {
        'HSV': cv2.COLOR_RGB2HSV,
        'LUV': cv2.COLOR_RGB2LUV,
        'HLS': cv2.COLOR_RGB2HLS,
        'YUV': cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb
        }
    feature_image=cv2.cvtColor(image, colorsp_mapping[cspace])
    return feature_image

def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

def crop_image(image, y_range, x_range):
    """
    input: np array 
    y_range corpping only
    """
    return image[y_range[0]: y_range[1], x_range[0]: x_range[1], :]

def image_scale_to_01(image):
    """
    input: np array
    255 to 01
    """
    return image.astype(np.float32)/255

def resize_image_by_scale(image, scale):
    """
    input: np array
    2D scale
    """
    return cv2.resize(image, (np.int(image.shape[1]/scale), np.int(image.shape[0]/scale)))

def convert_color(img, conv='RGB2YCrCb'):
    color_conv_mapping = {
        'RGB2YCrCb': cv2.COLOR_RGB2YCrCb,
        'BGR2YCrCb': cv2.COLOR_BGR2YCrCb,
        'RGB2LUV': cv2.COLOR_RGB2LUV,
        'RGB2YUV': cv2.COLOR_RGB2YUV
        }
    return cv2.cvtColor(img, color_conv_mapping[conv])

def make_color_params(spatial_size=(32,32),hist_bins=32, hist_range=(0,256)):
    return {
        'spatial_size': spatial_size,
        'hist_bins': hist_bins,
        'hist_range': hist_range,
        'option': 'COLOR'
        }
 
def make_hog_params(hog_channel=0):
    """
    HOG: 0, 1, 2, ALL
    """
    return {
        'hog_channel': hog_channel,
        'option': 'HOG'
        }


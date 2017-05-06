from skimage.feature import hog
import numpy as np
import matplotlib.image as mpimg

from classifier import *
from utilities import *

class Feature_Extractor:
    def __init__(self, orient, pix_per_cell, cell_per_block):
        """
        HOG Hyperparameter
        """
        self.orient=orient
        self.pix_per_cell=pix_per_cell
        self.cell_per_block=cell_per_block

    def _get_hog_features(self, img, vis=False, feature_vec=True):
        orient=self.orient
        pix_per_cell=self.pix_per_cell
        cell_per_block=self.cell_per_block

        features_rtn = hog(img, orientations=orient,
                         pixels_per_cell=(pix_per_cell, pix_per_cell),
                         cells_per_block=(cell_per_block, cell_per_block),
                         transform_sqrt=False,
                         visualise=vis,
                         feature_vector=feature_vec)
        if vis == True:
            return features_rtn[0], features_rtn[1]
        else:
            return features_rtn

    def _get_feature_image(self, image, cspace):
        if cspace != 'RGB':
            feature_image = colorsp_conversion(image, cspace)
        else:
            feature_image = np.copy(image)
        return feature_image
    def _get_hog(self, feature_image, params, window_range=None):
        hog_channel=params['hog_channel']
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                  self._get_hog_features(
                    feature_image[:, :, channel], 
                    vis=False, 
                    feature_vec=True))
        else:
            hog_features=self._get_hog_features(feature_image[:, :, hog_channel], vis=False, feature_vec=True)

        hog_features=np.ravel(hog_features)
        return hog_features

    def _get_hog_unravel(self, feature_image, params, vis=False, feature_vec=True):
        """
        TODO: Refactor to combine with _get_hog
        """
        hog_channel=params['hog_channel']
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(
                  self._get_hog_features(
                    feature_image[:, :, channel], 
                    vis=vis, 
                    feature_vec=feature_vec))
        else:
            hog_features=self._get_hog_features(feature_image[:, :, hog_channel], vis=vis, feature_vec=feature_vec)
        return hog_features
    
    def _get_sub_hog_features(self, features_list, params, y_range, x_range):
        """
        If all channels are used features_list passed in as list
        otherwise, just np array for single channel
        """
        hog_channel=params['hog_channel']
        y_start, y_end = y_range
        x_start, x_end = x_range

        hog_list = []
        if hog_channel == 'ALL':
            for channel in range(len(features_list)):
                hog_list.append(features_list[channel][y_start:y_end, x_start:x_end].ravel())
            return np.hstack(hog_list)
        else:
            return np.hstack(features_list[y_start:y_end, x_start:x_end].ravel())


    def _get_color(self, feature_image, params, window_range=None):
        spatial_size=params['spatial_size']
        hist_range=params['hist_range']
        hist_bins=params['hist_bins']

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        return np.concatenate((spatial_features, hist_features))

    def image_iter(self, imgs, callback, params, cspace='RGB'):
        features=[]
        for img in imgs:
            image = mpimg.imread(img)
            feature_image = self._get_feature_image(image, cspace)
            _features = callback(image, params, None)
            features.append(_features)
        return features
        

    def extract_features(self, imgs, cspace, params):
        """
        COLOR: params is dict
        HOG: HOG_channel
        """
        if params['option'] == 'COLOR':
            return self.image_iter(imgs, self._get_color, params, cspace)
        elif params['option'] == 'HOG':
            return self.image_iter(imgs, self._get_hog, params, cspace)

if __name__ == "__main__":
    pass
  #import matplotlib.pyplot as plt
  # Plot the examples
  #%matplotlib inline
  #fig = plt.figure()
  #plt.subplot(121)
  #plt.imshow(img, cmap='gray')
  #plt.title('Example Car Image')
  #plt.subplot(122)
  #plt.imshow(hog_image, cmap='gray')
  #plt.title('HOG Visualization')

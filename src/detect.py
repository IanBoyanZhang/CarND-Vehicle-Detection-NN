import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from  scipy.ndimage.measurements import label

from subsample_search import find_cars, draw_boxes
from classifier import Feature_Extractor

from heat_map import *
from moviepy.editor import VideoFileClip

from collections import deque

##---------------------------------------------------------
## Hyper parameters
##---------------------------------------------------------
ystart = 400
ystop = 656
xstart = 550
xstop = 1280
scale = 1.5
heat_threshold = 8
heat_percent = 0.5
FILTER_LENGTH = 8

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
window = dist_pickle['window']
hog_channel = dist_pickle['hog_channel']

svc_hog = dist_pickle["svc_hog"]
X_scaler_hog = dist_pickle["scaler_hog"]

svc_color = dist_pickle["svc_color"]
X_scaler_color = dist_pickle["scaler_color"]


##---------------------------------------------------------
## Global
##---------------------------------------------------------
FE = Feature_Extractor(orient, pix_per_cell, cell_per_block)
##---------------------------------------------------------
## pipeline
##---------------------------------------------------------

class Tracker():
    def __init__(self, filter_length):
        self.last_frames = deque(maxlen=filter_length)
        #self.current_frame_boxes = []
        return
    def car_tracker(self, image, DEBUG_BOX=False):
        """
        Debugging option
        "USE_QUEUE"
        """
        bbox_list_color = find_cars(image, 
            ystart,
            ystop, 
            xstart,
            xstop,
            scale, 
            svc_color, 
            X_scaler_color, 
            orient, 
            pix_per_cell, 
            cell_per_block, 
            spatial_size, 
            hist_bins,
            window,
            FE,
            use_color=True,
            use_hog=False,
            hog_channel=hog_channel, DEBUG_BOX=DEBUG_BOX)

        bbox_list_hog = find_cars(image, 
            ystart,
            ystop, 
            xstart,
            xstop,
            scale, 
            svc_hog, 
            X_scaler_hog, 
            orient, 
            pix_per_cell, 
            cell_per_block, 
            spatial_size, 
            hist_bins,
            window,
            FE,
            use_color=False,
            use_hog=True,
            hog_channel=hog_channel, DEBUG_BOX=DEBUG_BOX)

        # Ensemble
        #box_list = bbox_list_color + bbox_list_hog
        #box_list = bbox_list_color
        box_list = bbox_list_hog
        if DEBUG_BOX == 'USE_QUEUE':
            box_list = self.put_to_last_frames(box_list)

        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = add_heat(heat,box_list)

        if DEBUG_BOX == 'HEAT':
            return heat

        # Dynamical applying threshold
        # Apply threshold to help remove false positives

        max_heat = np.amax(heat)
        heat_threshold = max_heat * heat_percent
        heat = apply_threshold(heat, heat_threshold)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        # Omit heat map for now
        if DEBUG_BOX == 'GRID':
            out_image = np.copy(image)
            for box in box_list:
                out_image = draw_boxes(out_image, box)
            return out_image
        return draw_img

    def put_to_last_frames(self, new_frame):
      self.last_frames.append(new_frame)
      container = []
      for ind, q_item in enumerate(self.last_frames):
        for ind_frame, box in enumerate(q_item):
          container.append(box)

      # TODO: implement IOU
      return container

tracker = Tracker(FILTER_LENGTH)

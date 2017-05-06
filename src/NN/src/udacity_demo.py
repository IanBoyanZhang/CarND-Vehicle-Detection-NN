from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

from moviepy.editor import VideoFileClip

FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string(
#    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
#tf.app.flags.DEFINE_string(
#    'input_path', './data/sample.png',
#    """Input image or video to be detected. Can process glob input such as """
#    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")


tf.app.flags.DEFINE_string(
    'mode', 'video', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
#    'input_path', './data/test2.jpg',
    'input_path', './data/p4_project_video.mp4',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")

def resize_to_squeezeDet_size(image):
    """
    For now, squeezeDet requires input image/tensor shape to be 
    (1, 375, 1242, 3)
    """
    return cv2.resize(image, (1242, 375))

def resize_to_original_size(image, target_size=(1280, 720)):
    """
    Default Udacity video size
    """
    return cv2.resize(image, target_size)

def video_demo():
  """Detect videos."""

  with tf.Graph().as_default():
    # Load model
    mc = kitti_squeezeDet_config()
    mc.PLOT_PROB_THRESH      = 0.75
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)


    def frame_detector(frame):
    # crop frames
    #frame = frame[500:-205, 239:-439, :]
      frame = resize_to_squeezeDet_size(frame)
      im_input = frame.astype(np.float32) - mc.BGR_MEANS

    # Detect
      det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.image_input:[im_input], model.keep_prob: 1.0})

        
    # Filter
      final_boxes, final_probs, final_class = model.filter_prediction(
        det_boxes[0], det_probs[0], det_class[0])

      keep_idx    = [idx for idx in range(len(final_probs)) \
                      if final_probs[idx] > mc.PLOT_PROB_THRESH]
      final_boxes = [final_boxes[idx] for idx in keep_idx]
      final_probs = [final_probs[idx] for idx in keep_idx]
      final_class = [final_class[idx] for idx in keep_idx]

    # Draw boxes
    # TODO(bichen): move this color dict to configuration file
      cls2clr = {
        'car': (255, 191, 0),
#        'cyclist': (0, 191, 255),
#        'pedestrian':(255, 0, 191)
      }
      _draw_box(
        frame, final_boxes,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(final_class, final_probs)],
        cdict=cls2clr
      )

      frame = resize_to_original_size(frame)
      return frame



    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      times = {}
      count = 0

      output=FLAGS.out_dir + 'output.mp4'
      project_video = VideoFileClip(FLAGS.input_path)
      project_clip = project_video.fl_image(frame_detector)
      project_clip.write_videofile(output, audio=False)

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()

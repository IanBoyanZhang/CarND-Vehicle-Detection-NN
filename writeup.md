[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/searching_area.png
[image3]: ./output_images/heat_map_1.png
[image4]: ./output_images/heat_map_2.png
[image5]: ./output_images/heat_map_3.png
[image6]: ./output_images/heat_map_4.png
[image7]: ./output_images/heat_map_5.png
[image8]: ./output_images/heat_map_6.png
[image9]: ./output_images/car.png
[image10]: ./output_images/color_hist.png
[image11]: ./output_images/hog0.png
[image12]: ./output_images/hog1.png
[image13]: ./output_images/hog2.png
[image14]: ./output_images/notcar.png
[image15]: ./output_images/notcar_hog0.png
[image16]: ./output_images/notcar_hog1.png
[image17]: ./output_images/notcar_hog2.png
[image18]: ./output_images/color_spatial.png


[video1]: ./project_video.mp4
[video2]: ./output.mp4
[video3]: ./video/output_squeezeDet.mp4
[video4]: ./src/NN/data/out

---

**Vehicle Detection Project**


Code organization

All source code located in src/ path

  src/

    classifier.py

    data_exploerer.py

    detect.py

    heat_map.py

    process.py

    subsample_search.py

    train.py

    utilities.py

    window.py

***Data Input***
Where _data_exploerer_ is used to import both car and non-car labeled images recursively through glob. Please see _data\_import_ function for reference

Example of vehicle and non vehicle

![car_notcar][image1]: ./output_images/car_notcar.png


***Classifier Training***

_process.py_ is used for training process control to retrieve color and/or HOG (Histogram of Oriented Gradient) features from color classifier and/or HOG classifier defined in _classifer.py_

Please see retrieve_features function for how two different classifier color and HOG are used for invoking classifer


Three channels of labeled car image under _YCrCb_ color space

![YCrCb][image9]

Comparison between single channel in car picture and hog feature space

![hog0][image11]
![hog1][image12]
![hog2][image13]

Comparison between single channel in not picture and hog feature space

![YCrCb notcar][image14]
![notcar_hog0][image15]
![notcar_hog1][image16]
![notcar_hog2][image17]

Features under color transformation space

![Color spatial][image18]

![Color hist][image10]


Linear SVM (Support Vector Machine) classifier training/fitting procedured can be found in _train.py_

Hyper-parameters can also be configured in _process.py_.

Running process.py to train SVM model
```
  python process.py
```

Training feature normalization is on line 18 of _train.py_ using  ```X_scaler = StandardScaler().fit(X)```.
_20 percent_ of training data and labels are used revserved for testing

***Feature extraction***

Feature extraction during training

![Car][image9]


```shell
  Using params: {'spatial_size': (32, 32), 'hist_range': (0, 256), 'option': 'COLOR', 'hist_bins': 32}

  Training image shape (64, 64)

  Feature vector length: 3168

  15.89 Seconds to train SVC...

  Test Accuracy of SVC =  0.9551

  My SVC predicts:  [ 1.  1.  1.  0.  0.  0.  0.  1.  0.  0.]
  For these 10 labels:  [ 1.  1.  1.  0.  0.  0.  0.  1.  0.  0.]

  0.00107 Seconds to predict 10 labels with SVC

  Using params: {'option': 'HOG', 'hog_channel': 'ALL'}

  Training image shape (64, 64)

  Feature vector length: 5292

  14.84 Seconds to train SVC...
  
  Test Accuracy of SVC =  0.9914

  My SVC predicts:  [ 1.  1.  1.  0.  0.  0.  0.  1.  0.  0.]
  For these 10 labels:  [ 1.  1.  1.  0.  0.  0.  0.  1.  0.  0.]
  0.00106 Seconds to predict 10 labels with SVC
```

Using _YCrCb_ color space with below HOG parameters on a subset of all training images consistently achieve over _99 percent_ test accuracy.

  HOG parameters

```python

  orient=9

  pix_per_cell=8

  cell_per_block=2
```

I suspect this is a sign of over fitting.


***Sliding window search***

An sample _75 percent overlap_ searching grid is used for performing _sliding window search_. ![Searching grid][image2]

More details of sliding window search implementation can be found in _subsample_search.py__find_cars_ function

Sliding window will give candidate car locations. To make the pipeline more robust. It is recommanded to do a pseudo heat map search on top of blind 
sliding window search. Heat map search accumulate candidate boxes proposed by sliding window search to assign more weights to location pixels with more
candidate boxes.

Sliding window size and overlapping may affect detection robustness. For observing experiments we have acheived, higher overlapping rate could improve signal to 
noise (SNR) ratio, so that spatial integral could accumulate more information over detection intervals. However, this significantly reduced detection/estimation speed.


***Heat Map and Filtering***
Heat map implementation can be found in _heat_map.py_.

Heat map thresholding could be seen as a low pass filter in spatial domain.

To improve the robustness of detection tracking pipeline. Dynamic heat map threshold value is used to filtering heat map generated

see line 114, 115 in detect.py

![Heat map 1][image3]
![Heat map 2][image4]
![Heat map 3][image5]
![Heat map 4][image6]
![Heat map 5][image7]
![Heat map 6][image8]


A naive frame by frame filter is also used to smooth inter frame estimation variance. Five to ten filtering length is used to reduce single frame detection false
signal effects.

Implementation of _put\_to\_last\_frames_ can be found in  _detect.py_. It is equivalent to a low pass filter that reduce noisy data that changes frequently in short term. 
However this approach may introduce long term drifing.

Other ideas possibly worth exploring includes setting up a centroid point for potential detect object. So dimension and graphic perspective information could be used to draw
proper size bounding box. Leveraing perspective information to construct feasible 3D volume or etc. Later on, Interframe analysis could also use motion estimation for prediction
if the information is avaiable.

From reading different material, it seems classical sliding window classifiers are falling out of favor comparing to new _Neural Network(NN)_ approaches, considering detection robustness and speed.

## Discussion

After initial struggle with tuning classical SVM approach, suggested by [Mez](https://mez.github.io/2017/04/21/squeezedet-deep-learning-for-object-detection/), I took a detour of exploring SqueezeNet and SqueezeDet developed by UCB.

[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and 0.5MB model size](https://arxiv.org/abs/1602.07360)

[SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving](https://arxiv.org/abs/1612.01051)

It was a great reading of both papers. I had a brief attempt of porting squeezeDet to keras based on available Keras implementation of [squeezeNet](https://github.com/rcmalli/keras-squeezenet)

After working on this project, I realized that Convolutional NN is internally doing a sliding window. Mathematically, there should be some equivalence that I need more reading to comprehend.

Modified squeezeDet file can be found under _src/NN/squeezenet.py_

It worth noting in our implementation, output bias is omitted comparing to squeezeDet creator`s tensorflow implementation

Demoing of extracting a four D tensor obtained through modified squeezeDet defined in above file can be found in _src/NN/keras_squeeze.ipynb_

However, for the time being, I wouldn`t have enough time to implement and test squeezeDet inverse transform from feature space to image space, and associated bounding box filtering. 

Original author`s SqueezeDet tensorflow lib is included under src/NN with our modified scripts.

Using original author`s tensorflow implmenetation and pretrained weight. [Github squeezeDet](https://github.com/BichenWuUCB/squeezeDet)

After installing squeezeDet, copy customized script under _src/NN/udacity\_demo.py_ to  squeezeDet src folder. The script can be used to generate output video from project video.

On line 58 of _src/NN/udacity\_demo.py_ , plot probablity threshold is set to 0.75. There are bunch of other parameters of the model worth exploring.

Object detection through squeezeDet is much faster than my naive SVM implementation. Final output generated can be found:

A more sophisticated interframe traction based on IOU can be used to improve robustness of potential pipeline.

***Video output***

![output][video2]

SqueezeDet output
![SqueezeDet output][video3]

SqueezeDet output with p4 lane finding
![SqueezeDet p4][video4]

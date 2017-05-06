import glob
#from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles
# if python version is 3.5+
# http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
def data_importer(notcar_path, car_path):
  notcars = []
  cars = []
  images = glob.iglob(notcar_path, recursive=True)
  for image in images:
    notcars.append(image)

  images = glob.iglob(car_path, recursive=True)
  for image in images:
    cars.append(image)

  return notcars, cars
    
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    return data_dict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == "__main__":
  notcars_path = '../data/non-vehicles/**/*.png'
  cars_path = '../data/vehicles/**/*.png'
  notcars, cars = data_importer(notcars_path, cars_path)
  car_ind = np.random.randint(0, len(cars))
  notcar_ind = np.random.randint(0, len(notcars))

  # Read in car / not-car images
  car_image = mpimg.imread(cars[car_ind])
  notcar_image = mpimg.imread(notcars[notcar_ind])

  data_info = data_look(cars, notcars)

  print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
  print('of size: ',data_info["image_shape"], ' and data type:',
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images

# Plot the examples
#%matplotlib inline

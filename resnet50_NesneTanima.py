from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import os
import PIL as Image
from io import BytesIO
import requests
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array

model = ResNet50(weights='imagenet')


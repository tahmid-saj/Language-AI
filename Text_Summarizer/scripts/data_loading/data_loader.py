import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow_hub as hub
import string
from tensorflow.keras.layers import TextVectorization

DATA_DIR = "PubMed_20k_RCT_numbers_replaced_with_at_sign/"

filenames = [DATA_DIR + filename for filename in os.listdir(DATA_DIR)]
print(filenames)
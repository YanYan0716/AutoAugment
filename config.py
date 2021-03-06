import tensorflow as tf
# about model
WEIGHT_DECAY = 5e-4
NUM_CLASS = 10
DROPOUT = 0.2
WIDTH = 10
DEPTH = 28

# about dataset
# TRAIN_FILE_PATH = '../input/cifar10/cifar/train.csv'
# TEST_FILE_PATH = '../input/cifar10/cifar/test.csv'
TRAIN_FILE_PATH = '/content/cifar/unlabel.csv'
TEST_FILE_PATH = '/content/cifar/test.csv'
MAGNITUDES = None
IMG_SIZE = 32
AUTO_AUGMENT = True
CUTOUT = True
BATCH_SIZE = 128

# about train
MAX_EPOCH = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EVA_EPOCH = 1
SAVE_PATH = './weights/model' # save path

# about learning rate
ETA_MAX = 0.05
ETA_MIN = 4e-4

# data type
DTYPE = tf.float32

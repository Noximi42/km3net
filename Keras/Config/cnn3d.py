import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.layers.normalization as normal
import keras.models as models
import keras.utils.np_utils as kutils
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution3D, BatchNormalization, MaxPooling3D
# needs packages: sudo pip install pydot graphviz Pydot-ng
# from keras.utils.visualize_util import plot
from numpy import genfromtxt
import multi_gpu
from multi_gpu import *
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model

from keras.layers import LSTM 

from matrixflowUtils import *

number_parallel = 1

nTargets = 16
nTargets = 6
nTargets = 4
#nTargets = 1

import keras.backend as K
import math
import tensorflow as tf

from keras import *


def defineModelYZT(numClasses):
	nbf1 = 64
	nbf2 = 64
	nbf3 = 128
	nb_conv = 3

	model = models.Sequential()
	model.add(conv.Convolution3D(nbf1, (nb_conv,nb_conv,nb_conv), activation="relu", input_shape=(numy, numz, numt, 1), padding="same"))
	model.add(conv.Convolution3D(nbf1, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
	model.add(conv.MaxPooling3D(strides=(1,1,2)))
	model.add(core.Dropout(0.1))
	model.add(conv.Convolution3D(nbf2, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
	model.add(conv.Convolution3D(nbf2, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
	model.add(conv.MaxPooling3D(strides=(2,2,2)))
	model.add(conv.Convolution3D(nbf2, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
	#model.add(normal.BatchNormalization())
	model.add(conv.Convolution3D(nbf3, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
	model.add(conv.Convolution3D(nbf3, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
	model.add(core.Dropout(0.1))
	model.add(conv.Convolution3D(nbf3, (nb_conv,nb_conv,nb_conv), activation="relu", padding="same"))
        model.add(conv.MaxPooling3D(strides=(2,2,2)))
	model.add(normal.BatchNormalization())
	model.add(core.Flatten())
	model.add(core.Dense(256, activation="relu"))
        model.add(core.Dense(16, activation="relu"))
	model.add(core.Dense(numClasses) )
	#if number_parallel > 1:
	#	model = make_parallel(model, number_parallel)
	return model
	"""
        #model.add(layers.core.Permute((4,3,2,1)))
        # model.add(layers.core.Reshape((128, 8)))
	"""


# HDF5 files (default)
numx, numy, numz, numt = 1, 11, 18, 50
testfile, testsize = 'input/numuyztShufTail54921.csv.h5', 5000
trainfiles, trainsize = ['input/numuyztShufHead270k.csv.h5'], 270000

# CSV files
trainfilesCsv, testfileCsv = ['input/numuyztShufHead270k.csv'], 'input/numuyztShufTail54921.csv'


batchsize = 32
print "Batchsize = ", batchsize

modelname = "model_yzt_fromhdf5_numuCC_regressDirAndE_epoch"
restartIndex = 0	# 4 targets, 6xhdf5, bs 32, mse, adam

if restartIndex == 0:
	model = defineModelYZT(nTargets)
else:
	model = models.load_model("models/"+modelname+str(restartIndex)+".h5")

model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=["mean_squared_error"])
#model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
#model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mean_absolute_error"])
model.summary()


printSize = 5
i = restartIndex


while True:
	# process all hdf5 files, full epoch
	for f in trainfiles:
		i += 1
		print "Training ", i , " on file " , f
		model.fit_generator( generate_batches_from_hdf5_file(f, batchsize, numx, numy, numz, numt, nTargets), steps_per_epoch=int(trainsize/batchsize), epochs=1, verbose=1, max_q_size=1)
		# store the trained model
		model.save("models/"+modelname+str(i)+".h5")
		if testfile != "" :
			results = doTheEvaluation(model, nTargets, testfile, testsize, printSize, numx, numy, numz, numt, batchsize)

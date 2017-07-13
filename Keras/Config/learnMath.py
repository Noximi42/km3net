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

nTargets = 1

import keras.backend as K
import math
import tensorflow as tf

from keras import *


def defineModelAdd():
	model = models.Sequential()
	model.add(core.Dense(128, input_shape=(3,)))
	model.add(core.Dropout(0.2))
	model.add(core.Dense(128, activation="relu"))
	model.add(core.Dropout(0.2))
        model.add(core.Dense(16, activation="relu"))
	model.add(core.Dense(nTargets) )
	#if number_parallel > 1:
	#	model = make_parallel(model, number_parallel)
	return model


def giveAddXandYBatch(batchsize):
	maxRange = 1000.0
	numOperations = 1.0
	xs = np.zeros(3*batchsize)
	xs = np.reshape(xs, (batchsize, 3))
	ys = np.zeros(batchsize)
	c = 0
	while c <= batchsize:
		x1 = np.random.rand()*maxRange#-0.5*maxRange
		x2 = np.random.rand()*maxRange#-0.5*maxRange
		op = int(np.random.rand()*numOperations)
		y = 0
		if op == 0:
			y = x1+x2
		elif op == 1:
			y = x1-x2
		elif op == 2:
			y = x1*x2
		elif op == 3:
			if x2 > x1/maxRange:
				y = x1/x2
			else:
				y = maxRange+1.0
		else:
			y = 0.0
		# print x1, " ", x2, " ", op, " ", y
		xs[c,0] = x1
		xs[c,1] = x2
		xs[c,2] = op*maxRange/(numOperations)
		ys[c] = y
		c += 1
		if c == batchsize:
			c = 0
			return (xs,ys)
	
def generateAddXandY(batchsize):
	while True:
		x, y = giveAddXandYBatch(batchSize)
		yield (x,y)

	

batchSize = 8
print "Batchsize = ", batchSize

modelname = "model_comp1000_epoch"
restartIndex = 0	# bs 256, mae, sgd
#restartIndex = 167	# bs 256, mae, sgd
#restartIndex = 210	# bs 256, mae, sgd
#restartIndex = 250	# bs 256, mae, sgd

if restartIndex == 0:
	model = defineModelAdd()
else:
	model = models.load_model("models/"+modelname+str(restartIndex)+".h5")

model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=["mean_squared_error"])
#model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
#model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
#model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mean_absolute_error"])
model.summary()

i = restartIndex
while True:
        # process all files, full epoch
	i += 1
	print "Training ", i 
	history = model.fit_generator( generator=generateAddXandY(batchSize), steps_per_epoch=20000, epochs=2, verbose=1, validation_data=generateAddXandY(batchSize), validation_steps=500)
	#history = model.fit_generator( generator=generateAddXandY(batchSize), steps_per_epoch=1000, epochs=1, verbose=1, validation_data=generateAddXandY(batchSize), validation_steps=1000)
	# store the trained model
	model.save("models/"+modelname+str(i)+".h5")

	x, y = giveAddXandYBatch(5)
	predictions = model.predict_on_batch( x )
	print x
	print y
	print np.reshape(predictions, len(predictions))





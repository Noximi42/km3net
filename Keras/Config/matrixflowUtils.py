import numpy as np
import math
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

import h5py


# sets the target vector accordingly to the mcinfo and the desired task
# mcinfo is (default until changed): event_id, particle_type, dir_x, dir_y, dir_z, energy, isCC, bjorkeny, up/down, weightW2
# index                                 0          1            2      3      4       5     6       7        8        9
def encodeTargets(mcinfo, nTargets):
	if nTargets == 16:
		# everything at once:
		temp = mcinfo
		temp[5] = np.log10(mcinfo[5])/10.0

		temp[2] = 0.5*(mcinfo[2] + 1.0)
		temp[3] = 0.5*(mcinfo[3] + 1.0)
		temp[4] = 0.5*(mcinfo[4] + 1.0)

		numPids = 9 
		pids = np.zeros(numPids)
		
		pid = mcinfo[1]
		# just hardcode the mapping
		if pid == -12:		# a nu e
			pids[1] = 1.0
		elif pid == 12:		# nu e
			pids[2] = 1.0
		elif pid == -14:	# a nu mu
			pids[3] = 1.0
		elif pid == 14:		# nu mu
			pids[4] = 1.0
		elif pid == -16:	# a nu tau
			pids[5] = 1.0
		elif pid == 16:		# nu tau
			pids[6] = 1.0
		elif pid == -13:	# a mu
			pids[7] = 1.0
		elif pid == 13:		# mu
			pids[8] = 1.0
		else:	# if it's nothing else we know: we don't know what it is ;-)
			pids[0] = 1.0
		# TODO: Probably pid and isCC work better if there are classes e.g. for numuCC and numuNC and nueCC and nueNC 
		# instead of single classes for numu and nue but a combined flag is_CC_or_NC for all flavour
		# especially for numuCC and numuNC
		
		trainY = np.concatenate([np.reshape( temp[2:9], len(temp[2:9]), 1 ), np.reshape(pids,numPids,1)])
		# 0 1 2 3 4  5  6  7          8    9   10    11   12     13    14  15
		# x y z E cc by ud unknownPid anue nue anumu numu anutau nutau amu mu
		return trainY

	elif nTargets == 6:
		# direction, energy and iscc and bjorken-y:
		temp = mcinfo
		# energy
		temp[5] = np.log10(mcinfo[5])/10.0
		# direction
		temp[2] = 0.5*(mcinfo[2] + 1.0)
		temp[3] = 0.5*(mcinfo[3] + 1.0)
		temp[4] = 0.5*(mcinfo[4] + 1.0)

		trainY = np.reshape( temp[2:8], nTargets, 1 )
		# 0 1 2 3 4  5  
		# x y z E cc by 
		return trainY

	elif nTargets == 4:
		# direction and energy:
		temp = mcinfo
		temp[5] = np.log10(mcinfo[5])/10.0

		temp[2] = 0.5*(mcinfo[2] + 1.0)
		temp[3] = 0.5*(mcinfo[3] + 1.0)
		temp[4] = 0.5*(mcinfo[4] + 1.0)

		trainY = np.reshape( temp[2:6], nTargets, 1 )
		# 0 1 2 3 
		# x y z E 
		return trainY

	elif nTargets == 1:
		# energy:
		temp = mcinfo
		temp[5] = np.log10(mcinfo[5])/10.0
		return np.reshape( temp[5:6], nTargets, 1 )

	else:
		print "Number of targets (" + str(nTargets) + ") not supported!"
		return mcinfo
		

def decodeTargets(ys, nTargets):
	if nTargets == 16:
		# everything at once:
		infoReconstructed = np.zeros(8)
		infoReconstructed[0] = ys[0]*2.0-1.0
		infoReconstructed[1] = ys[1]*2.0-1.0
		infoReconstructed[2] = ys[2]*2.0-1.0
		infoReconstructed[3] = 10.0**(10.0*ys[3])	
		if ys[4] > 0.5:
			infoReconstructed[4] = 1.0
		else:
			infoReconstructed[4] = 0.0
		infoReconstructed[5] = ys[5]
		if ys[6] > 0.5:
			infoReconstructed[6] = 1.0
		else:
			infoReconstructed[6] = 0.0

		maxPid = np.max(ys[7:16])
		if maxPid == ys[7]:
			infoReconstructed[7] = 0
		elif maxPid == ys[8]:
			infoReconstructed[7] = -12
		elif maxPid == ys[9]:
			infoReconstructed[7] = 12
		elif maxPid == ys[10]:
			infoReconstructed[7] = -14
		elif maxPid == ys[11]:
			infoReconstructed[7] = 14
		elif maxPid == ys[12]:
			infoReconstructed[7] = -16
		elif maxPid == ys[13]:
			infoReconstructed[7] = 16
		elif maxPid == ys[14]:
			infoReconstructed[7] = -13
		elif maxPid == ys[9]:
			infoReconstructed[7] = 13
		else:
			print "The maximal particle id is not in the particle ids ... strange."
			infoReconstructed[7] = 0
		
		return infoReconstructed
	else:
		print "Number of targets (" + str(nTargets) + ") not supported!"
		return mcinfo

# returns the dimensions tuple for 2,3 and 4 dimensional data
# we don't have to write separate functions with different argument lists for different dimensions but can always use numx, numy, numz, numt
def getDimensionsEncoding(batchsize, numx, numy, numz, numt):
	dimensions = (batchsize,numx,numy,numz,numt)
	if numx == 1:
		if numy == 1:
			print "2D case without dimensions x and y"
			dimensions = (batchsize,numz,numt,1)
		elif numz == 1:
			print "2D case without dimensions x and z"
			dimensions = (batchsize,numy,numt,1)
		elif numt == 1:
			print "2D case without dimensions x and t"
			dimensions = (batchsize,numy,numz,1)
		else:
			# print "3D case without dimension x"
			dimensions = (batchsize,numy,numz,numt,1)

	elif numy == 1:
		if numz == 1:
			print "2D case without dimensions y and z"
			dimensions = (batchsize,numx,numt,1)
		elif numt == 1:
			print "2D case without dimensions y and t"
			dimensions = (batchsize,numx,numz,1)
		else:
			print "3D case without dimension y"
			dimensions = (batchsize,numx,numz,numt,1)

	elif numz == 1:
		if numt == 1:
			print "2D case without dimensions z and t"
			dimensions = (batchsize,numx,numy,1)
		else:
			print "3D case without dimension z"
			dimensions = (batchsize,numx,numy,numt,1)

	elif numt == 1:
		print "3D case without dimension t"
		dimensions = (batchsize,numx,numy,numz,1)

	else:	# 4 dimensional
		# print "4D case"
		dimensions = (batchsize,numx,numy,numz,numt)

	return dimensions


# split a line into class and other mc-related info and features
def process_line(lineAsString, dimensions, nTargets):
	dimList = list(dimensions)
	dimList[0] = 1
	singleLineDimensions = tuple(dimList)
        lineAsString = lineAsString.strip('\n')
        line = np.array(lineAsString.split(','), np.float32)
        numMC = int(line[0]+0.001)        # the first number in each line states how many MC related infos are present
        trainX = line[numMC+1:].reshape(singleLineDimensions).astype(float)         # everything after the first numMC+1 entries are features

	# the first numMC+1 entries (but 0 is numMC itself) are information from MC and may not be used as input for training! 
        trainY = encodeTargets( line[1:numMC+1], nTargets)
        return (trainX, trainY)

# generator that returns arrays of batchsize events
def generate_batches_from_csv_file(filename, batchsize, numx, numy, numz, numt, nTargets):
	dimensions = getDimensionsEncoding(batchsize,numx,numy,numz,numt)
	xs = np.array( np.zeros(batchsize*numx*numy*numz*numt), ndmin=5 )
	xs = np.reshape( xs, dimensions )
	ys = np.array( np.zeros(batchsize*nTargets), ndmin=2 )
	ys = np.reshape( ys, (batchsize, nTargets) )

	while 1:
		c = 0
		f = open(filename)
		for line in f:
			# create numpy arrays of input data
                        # and labels, from each line in the file
                        x, y = process_line(line, dimensions, nTargets)
                        xs[c] = x
                        ys[c] = y
                        c = c + 1
                        if c >= batchsize:
				#np.set_printoptions(threshold=np.inf)
				#print ys
				#print xs
                                yield (xs, ys)
                                c = 0	
		f.close()

import math

# generator that returns arrays of batchsize events
# from hdf5 file
def generate_batches_from_hdf5_file(filename, batchsize, numx, numy, numz, numt, nTargets):
	dimensions = getDimensionsEncoding(batchsize,numx,numy,numz,numt)

	xs = np.array( np.zeros(batchsize*numx*numy*numz*numt), ndmin=5 )
	xs = np.reshape( xs, dimensions )
	ys = np.array( np.zeros(batchsize*nTargets), ndmin=2 )
	ys = np.reshape( ys, (batchsize, nTargets) )
	while 1:
		# Open the file
		f = h5py.File(filename, "r")
		# Check how many entries there are
		filesize = len(f['y'])
		print "filesize = ", filesize
		# count how many entries we have read
		cTotal = 0
		# as long as we haven't read all entries from the file: keep reading
		while cTotal < (filesize-batchsize):
			# start the next batch at index 0
			# create numpy arrays of input data (features)
			xs = f['x'][cTotal:cTotal+batchsize]
			xs = np.reshape(xs, dimensions).astype(float)

			# and mc info (labels)
			yVals = f['y'][cTotal:cTotal+batchsize]
			yVals = np.reshape(yVals, (batchsize, yVals.shape[1]))
			# encode the labels such that they are all within the same range (and filter the ones we don't want for now)
			c = 0
			for yVal in yVals:
				ys[c] = encodeTargets (yVal, nTargets)
				c += 1

			# we have read one batch more from this file
			cTotal += batchsize
			#np.set_printoptions(threshold=np.inf)
			#print ys
			#print xs
			yield (xs, ys)
		f.close()

class batchReaderHdf5:
	# count how many entries we have read already
	cTotal = 0
	
	# return a batch of events
        def read_batch_from_file(self, filename, batchsize, numx, numy, numz, numt, nTargets):
		dimensions = getDimensionsEncoding(batchsize,numx,numy,numz,numt)

		xs = np.array( np.zeros(batchsize*numx*numy*numz*numt), ndmin=5 )
		xs = np.reshape( xs, dimensions )
		ys = np.array( np.zeros(batchsize*nTargets), ndmin=2 )
		ys = np.reshape( ys, (batchsize, nTargets) )
		while 1:
			# Open the file
			f = h5py.File(filename, "r")
			# Check how many entries there are
			filesize = len(f['y'])
			# as long as we haven't read all entries from the file: keep reading
			while self.cTotal < filesize-batchsize:
				cTotal = self.cTotal
				# start the next batch at index 0
				c = 0
				# create numpy arrays of input data (features)
				xs = f['x'][cTotal:cTotal+batchsize]
				xs = np.reshape(xs, dimensions)

				# and mc info (labels)
				yVals = f['y'][cTotal:cTotal+batchsize]
				yVals = np.reshape(yVals, (batchsize, yVals.shape[1]))
				# encode the labels such that they are all within the same range (and filter the ones we don't want for now)
				for yVal in yVals:
					ys[c] = encodeTargets (yVal, nTargets)
					c += 1

				# we have read one batch more from this file
				self.cTotal += batchsize
				return (xs, ys)
			f.close()
			self.cTotal = 0

class batchReaderCsv:
        filename = "zzztemp"
        inFile = open(filename, 'w+')
	
        # return a batch of events
        def read_batch_from_file(self, filename, batchsize, numx, numy, numz, numt, nTargets):
		dimensions = getDimensionsEncoding(batchsize,numx,numy,numz,numt)
                xs = np.array( np.zeros(batchsize*numx*numy*numz*numt), ndmin=5 )
                xs = np.reshape( xs, dimensions )
                ys = np.array( np.zeros(batchsize*nTargets), ndmin=2 )
                ys = np.reshape( ys, (batchsize, nTargets) )
                c = 0
                if (self.filename != filename):
                        #self.inFile.close()
                        self.filename = filename
                        self.inFile = open(filename, 'r')

                while True:
                        line = self.inFile.readline()
                        if not line:
                                self.inFile.close()
                                self.inFile = open(filename, 'r')

                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x, y = process_line(line, dimensions, nTargets)
                        xs[c] = x
                        ys[c] = y
                        c += 1
                        if c >= batchsize:
                                return (xs, ys)
		print "shouldn't be reached"
                self.inFile.close()



# computes the angular difference between two 3d vectors
def angleDiff(v1, v2):
	dotp = np.dot(v1,v2)
	n1 = np.linalg.norm(v1)
	n2 = np.linalg.norm(v2)
	cosalpha = dotp/(n1*n2)
	if cosalpha <= -1.0:
		#print v1, v2
		#print dotp, " ", n1, " ", n2, " ", cosalpha
		return 180.0
	if cosalpha >= 1.0:
		#print v1, v2
		#print dotp, " ", n1, " ", n2, " ", cosalpha
		return 0.0

	alpha = np.arccos(cosalpha)
	alphaDegrees = alpha / 3.1415 * 180.0
	return alphaDegrees
      

def predictAndPrintSome(model, testFile, printSize, numx, numy, numz, numt, nTargets, doCsv = False):
	mySmallR = batchReaderHdf5()
	if doCsv == True:
		mySmallR = batchReaderCsv()
	for j in range(printSize):
		x, yTrue = mySmallR.read_batch_from_file(testFile, 1, numx, numy, numz, numt, nTargets)
		
		# reconstruct values so we can look at them
		predictions = model.predict_on_batch(x)
		consideredNClasses = predictions.shape[1]
		yPred = predictions[0][0:consideredNClasses]
		if consideredNClasses > 1:
			yTrue = np.reshape(yTrue[0:consideredNClasses], consideredNClasses)
			yPred = np.reshape(yPred[0:consideredNClasses], consideredNClasses)
		
		#yTrue = decodeTargets(y, nTargets)
		#yPred = decodeTargets(predictions[0], nTargets)
		numberForPrint = min(consideredNClasses, 12)
		np.set_printoptions(precision=2)
		print "True and predicted:"
		print yTrue[0:numberForPrint]
		print yPred[0:numberForPrint]


def doTheEvaluation(model, nTargets, testFile, testSize, printSize, numx, numy, numz, numt, batchSize, doCsv = False):
	if printSize > 0:
		predictAndPrintSome(model, testFile, printSize, numx, numy, numz, numt, nTargets, doCsv)

	accuracy = 0.0
	mean_error = 0.0
	mean_squ_error = 0.0
	energy_diffs = []
	energy_diffs_rel = []
	angle_diffs = []
	angle_diffs_he = [] # greater than 10 TeV = 10^5 GeV = 0.5 in y_ETrue = y[3]

	myR = batchReaderHdf5()
	if doCsv == True:
		myR = batchReaderCsv()

	for l in range(int(testSize/batchSize+0.5)):
		# reconstruct some values so we can look at them
		xs, ys = myR.read_batch_from_file(testFile, batchSize, numx, numy, numz, numt, nTargets)
		predictions = model.predict_on_batch( xs )
		consideredNClasses = predictions.shape[1]

		for j in range(predictions.shape[0]):
			yTrue = ys[j]
			yPred = predictions[j]
			if consideredNClasses > 1:
				#print yTrue
				yTrue = np.reshape(yTrue[0:consideredNClasses], consideredNClasses)
				yPred = np.reshape(yPred[0:consideredNClasses], consideredNClasses)

			if consideredNClasses == 1:
				diff = abs(yTrue-yPred)
				if diff < 0.1:
					accuracy += 1.0
				mean_error += diff
				mean_squ_error += diff*diff

			if consideredNClasses > 1:
				for k in range(consideredNClasses):
					diff = abs(yTrue[k]-yPred[k])
					if diff < 0.1:
						accuracy += 1.0
					mean_error += diff
					mean_squ_error += diff*diff

			if consideredNClasses >= 4:
				energy_diffs.append( abs(yTrue[3]-yPred[3])/yTrue[3] )
				energy_diffs_rel.append( np.log10(   (10.0**(10.0*yPred[3])) / (10.0**(10.0*yTrue[3]))   ) )
				angle_diffs.append(angleDiff(yTrue[0:3], yPred[0:3]))
				if yTrue[3] > 0.5:	# if energy above 10TeV
					angle_diffs_he.append( angleDiff(yTrue[0:3], yPred[0:3]) )

	results = []
	nEventsTested = len( energy_diffs )
	nLabelsTested = nEventsTested * consideredNClasses
	results.append(["Number tested", nEventsTested])

	accuracy /= nLabelsTested
	results.append(["Accuracy", accuracy])
	mean_error /= nLabelsTested
	results.append(["Mean error", mean_error])
	mean_squ_error /= nLabelsTested
	results.append(["Mean squared error", mean_squ_error])
	mean_angle_diff = sum(angle_diffs)/nEventsTested
	results.append(["Mean angular error", mean_angle_diff])
	angle_diffs.sort()
	median_angle_diff = angle_diffs[int(0.5*nEventsTested+0.5)]
	results.append(["Median angular error", median_angle_diff])
	print ["Median angular error", median_angle_diff]
	angle_quantile_16 = angle_diffs[int(0.16*nEventsTested+0.5)]
	results.append(["Lower 16 angular error", angle_quantile_16])
	angle_quantile_84 = angle_diffs[int(0.84*nEventsTested+0.5)]
	results.append(["Upper 84 angular error", angle_quantile_84])
	angle_quantile_02 = angle_diffs[int(0.02*nEventsTested+0.5)]
	results.append(["Lower 02 angular error", angle_quantile_02])
	angle_quantile_98 = angle_diffs[int(0.98*nEventsTested+0.5)]
	results.append(["Upper 98 angular error", angle_quantile_98])

	mean_energy_diff = sum(energy_diffs)/nEventsTested
	results.append(["Mean error energy", mean_energy_diff])
	energy_diffs.sort()
	median_energy_diff = energy_diffs[int(0.5*nEventsTested+0.5)]
	results.append(["Median error energy", median_energy_diff])
	energy_quantile_16 = energy_diffs[int(0.16*nEventsTested+0.5)]
	results.append(["Lower 16 error energy", energy_quantile_16])
	energy_quantile_84 = energy_diffs[int(0.84*nEventsTested+0.5)]
	results.append(["Upper 84 error energy", energy_quantile_84])
	energy_quantile_02 = energy_diffs[int(0.02*nEventsTested+0.5)]
	results.append(["Lower 02 error energy", energy_quantile_02])
	energy_quantile_98 = energy_diffs[int(0.98*nEventsTested+0.5)]
	results.append(["Upper 98 error energy", energy_quantile_98])

	mean_energy_diff_rel = sum(energy_diffs_rel)/nEventsTested
	results.append(["Mean error energy relative", mean_energy_diff_rel])
	energy_diffs_rel.sort()
	median_energy_diff_rel = energy_diffs_rel[int(0.5*nEventsTested+0.5)]
	results.append(["Median error energy relative", median_energy_diff_rel])
	energy_quantile_rel_16 = energy_diffs_rel[int(0.16*nEventsTested+0.5)]
	results.append(["Lower 16 error energy relative", energy_quantile_rel_16])
	energy_quantile_rel_84 = energy_diffs_rel[int(0.84*nEventsTested+0.5)]
	results.append(["Upper 84 error energy relative", energy_quantile_rel_84])
	energy_quantile_rel_02 = energy_diffs_rel[int(0.02*nEventsTested+0.5)]
	results.append(["Lower 02 error energy relative", energy_quantile_rel_02])
	energy_quantile_rel_98 = energy_diffs_rel[int(0.98*nEventsTested+0.5)]
	results.append(["Upper 98 error energy relative", energy_quantile_rel_98])
	
	variance = 0.0
	for energy_diff in energy_diffs_rel:
		variance += (mean_energy_diff_rel - energy_diff)*(mean_energy_diff_rel - energy_diff)
	variance /= nEventsTested
	results.append(["Variance energy relative", variance])
	results.append(["Sigma energy relative", math.sqrt(variance)])

	mean_angle_diff_he = 0.0
	median_angle_diff_he = 0.0
	nEventsHE = len(angle_diffs_he)
	results.append(["Number tested highE", nEventsHE])
	if (nEventsHE > 0):
		mean_angle_diff_he = sum(angle_diffs_he)/nEventsHE
		results.append(["Mean error angular highE", mean_angle_diff_he])
		angle_diffs_he.sort()
		median_angle_diff_he = angle_diffs_he[int(0.5*nEventsHE+0.5)]
		results.append(["Median error angular highE", median_angle_diff_he])

	print ["Mean error", mean_error]
	return results


import sys
import os
import random as rd
import math



filenames = []
for root, dirs, files in os.walk(r'.'):
	for file in files:
		if file.endswith('.csv'):
			filenames.append(file)


oFile = "./numuUpDown/TrainData.txt"
oFileTest = "./numuUpDown/TestData.txt"


f = open(oFile, 'w')
f.write('')
f.close
f = open(oFileTest , 'w')
f.write('')
f.close
with open(r'./numuxyzt.csv') as data:
	f = open(oFile, 'a')
	fTest = open(oFileTest, 'a')

	i = 1
	up = 0
	down = 0
		
	print("start!")
	for line in data:
		#print(line)
		values = line.split(',')

		z = values[5]
		
		if float(z) > 0:
			z = "1 0"
			up = up + 1
		else:
			z = "0 1"
			down = down + 1
		

		del values[:11]
		if rd.random() < 0.1:
			fTest.write('|labels ' + z + ' |features ' + ' '.join(values))
		else:
			f.write('|labels ' + z + ' |features ' + ' '.join(values))
		
		i += 1
		if i % 1000 == 0:
			print(i / 324921.0 * 100)
		#print(i)

	f.close()
	print("finished!")
	print("Up going: " + str(up) + "\tDown going: " + str(down))

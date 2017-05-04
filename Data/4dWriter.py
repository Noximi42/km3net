import sys
import os
import random as rd
import math



filenames = []
for root, dirs, files in os.walk(r'.'):
	for file in files:
		if file.endswith('.csv'):
			filenames.append(file)


oFile = "./TrainDataReg.txt"
oFileTest = "./TestDataReg.txt"


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
	
	print("start!")
	for line in data:
		#print(line)
		values = line.split(',')

		x = values[3]
		y = values[4]
		z = values[5]
		E = math.log10(float(values[6])) / 10
		

		del values[:9]
		if rd.random() < 0.1:
			fTest.write('|labels ' + x + ' ' + ' ' + y + ' ' + z + ' ' + str(E) + ' |features ' + ' '.join(values))
		else:
			f.write('|labels ' + x + ' ' + ' ' + y + ' ' + z + ' ' + str(E) + ' |features ' + ' '.join(values))
		
		i += 1
		if i % 1000 == 0:
			print(i / 324921.0 * 100)
		#print(i)

	f.close()
	print("finished!")

import pickle as pkl

import os
import sys

platform = 'github/'

for filename in os.listdir(platform):
	with open(platform+filename, 'rb') as f:
		measurement = pkl.load(f)
		print(filename)
		print(measurement)
		
		
for filename in os.listdir(platform):
	with open(platform+filename, 'rb') as f:
		measurement = pkl.load(f)
		print(filename)
		print(type(measurement))
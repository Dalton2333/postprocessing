""" 
Created on 02/11/2012

@author: Daniel Stojanov

Calculated using the algorithm described in:
J. Grandy. Efficient computation of volume
of hexahedral cells. Lawrence Livermore National Laboratory,
October 1997. UCRL-ID-128886.

Tested on my Core is-2600 CPU @ 3.4GHz
just under 1,000,000 calculations/minute
"""

import numpy as np

def hex_volume(x):
	""" x is a list that contains the coordinates
	
	of each of the 8 verticies of the hexahedron
	"""
	dot1 = [None]*3
	dot1[0] = (x[6]-x[1]) + (x[7]-x[0])
	dot1[1] = x[6] - x[3]
	dot1[2] = x[2] - x[0]
	
	dot2 = [None]*3
	dot2[0] = x[7] - x[0]
	dot2[1] = (x[6]-x[3]) + (x[5]-x[0])
	dot2[2] = x[6] - x[4]
	
	dot3 = [None]*3
	dot3[0] = x[6] - x[1]
	dot3[1] = x[5] - x[0]
	dot3[2] = (x[6]-x[4]) + (x[2]-x[0])
	
	# Merge the vectors into a square matrix
	dot1 = np.column_stack((dot1[0], dot1[1], dot1[2]))
	dot2 = np.column_stack((dot2[0], dot2[1], dot2[2]))
	dot3 = np.column_stack((dot3[0], dot3[1], dot3[2]))
	
	volume = np.linalg.det(dot1) + \
		np.linalg.det(dot2) + \
		np.linalg.det(dot3)

	volume /= 12
	return volume
	
# Testing code_pack for this module
if __name__ == "__main__":
	x = range(8)
	# Uncomment this block to test perfomance
	# to create 1,000,000 random hexahedra (that probably
	# don't actually represent any meaningful geometry since
	# vector points are random) and calculate the algorithm
	#"""
	import time
	startTime = time.time()
	
	# repeat 1 million times
	for j in range(1000000):
		# Create 8 random vector positions
		for i in range(8):
			x[i] = np.array([
				np.random.random(),
				np.random.random(),
				np.random.random()
			])
			
		# Repeat every 100,000 times
		if j % 100000 == 0:
			print(hex_volume(x))
	
	print("time to perform 1 million volume calcualtions:")
	print(time.time() - startTime)
	#"""
	
	# Test arbitrary hex geometries
	
	x[0] = np.array([0,0,0])
	x[1] = np.array([2,0,0])
	x[2] = np.array([2,2,0])
	x[3] = np.array([0,2,0])
	
	x[4] = np.array([0,0,2])
	x[5] = np.array([2,0,2])
	x[6] = np.array([2,2,2])
	x[7] = np.array([0,2,2])

	print(hex_volume(x))

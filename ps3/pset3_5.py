import matplotlib 
from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
import numpy as np


 A = imread('peppers-small.tiff')
 plt.imshow(A)
 plt.show()

 R = A[:,:,0]
 G = A[:,:,1]
 B = A[:,:,2]

 points = []
 for i in range(512):
 	for j in range(512):
 		point = [R[i,j], G[i,j], B[i,j]]
 		points.append(point)


#Initialization
k = 16
mu = []
for i in range(k):
	initial = random.randint(0,262144)
	mu.append(points[i])

c = []
check = True
while(check):

	c_old = c #to check for convergence we store the past c
	for i in range(len(points)):
		min_index = 0
		min_value = 100000000
		for j in range(k):
			val = (np.linalg.norm(points[i] - mu[j]))**2
			if (val < min_value):
				min_value = val
				min_index = j
		#At the end of iteration have the j value for each point
		c[i] = min_index #at the ith index
	#Now we have the new values of c chosen from the current cluster centroids

	#convergence check
	if (c_old == c):
		check = False
		continue

	#Now we re-calculate the centroids
	for j in range(k):
		denom = 0 #case where this ends up being zero?
		num = 0
		for index in c:
			if (index = j):
				denom += 1
				num += points[index]
		if denom == 0: continue 
		mu[j] = num/denom

	#Now we have the newly calculated cluster centroids, and we will repeat until convergence

#Now we have the c values for each cluster identity as well as mu, the cluster centers. 








#Take the matrix A from peppers-large.tiff, and replace each pixel’s (r, g, b) values with the value of the closest cluster centroid. 
#Display the new image, and compare it visually to the original image. 
#Include in your write-up all your code and a copy of your compressed image.

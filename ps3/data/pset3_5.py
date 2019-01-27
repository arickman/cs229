import matplotlib
from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
import numpy as np
import pdb

eps = 0.001
A = imread('peppers-small.tiff')
#plt.imshow(A)
#plt.show()

R = A[:,:,0]
G = A[:,:,1]
B = A[:,:,2]

numrows = len(R)    # 3 rows in your example
numcols = len(R[0]) 

points = []
for i in range(numrows):
	for j in range(numcols):
 		point = [R[i,j], G[i,j], B[i,j]]
 		points.append(point)

#Initialization
k = 16
mu = []
for i in range(k):
	initial = random.randint(0,numrows*numcols - 1)
	mu.append(points[initial])

np.array(points)
np.array(mu)
c = np.zeros(len(points))
check = True
iters = 0
#pdb.set_trace()
while(check):
	iters +=1
	#print(iters)
	#num = np.zeros((16,3))
	#denom = np.zeros((16,1))
	c_old = c #to check for convergence we store the past c
	for i in range(len(points)):
		min_index = 0
		min_value = 100000000
		for j in range(k):
			#calculate the numerator and denominator for the later calculation in 16 length vector
			#if (int(c[i]) == j): 
				#denom[j] += 1
				#num[j] += points[i]
			val = (np.linalg.norm(np.subtract(points[i], mu[j]), ord = 2))**2
			if (val < min_value):
				min_value = val
				min_index = j
		#At the end of iteration have the j value for each point
		c[i] = int(min_index) #at the ith index
	#Now we have the new values of c chosen from the current cluster centroids

	#convergence check
	#print("got to convergence check")
	if ((np.linalg.norm(np.subtract(c, c_old), ord = 2)**2) < eps) and (iters > 30):
		#print("Converged")
		check = False
		continue
	if (iters > 30):
		#print("Took more than 30 iterations and stopped.")
		check = False
		continue

	#Now we re-calculate the centroids
	#print("recalculating the centroids")
	num = np.zeros((k,3))
	denom = np.zeros((k,1))
	for i in range(len(c)):
		j_val = int(c[i]) #like 5
		denom[j_val] += 1
		num[j_val] += points[i]

	for j in range(k):
		if denom[j] == 0: continue
		mu[j] = num[j]/denom[j]

	#Now we have the newly calculated cluster centroids, and we will repeat until convergence

#Now we have the c values for each cluster identity as well as mu, the cluster centers. 
#########################################################################################################

Q = imread('peppers-large.tiff')
Q.setflags(write=1)
plt.imshow(Q)
plt.show()

R = Q[:,:,0]
G = Q[:,:,1]
B = Q[:,:,2]

numrows = len(R)    # 3 rows in your example
numcols = len(R[0]) 

for i in range(numrows):
	for j in range(numcols):
 		point = [R[i,j], G[i,j], B[i,j]]
 		np.array(point)
 		#get closest centroid
 		closest = np.zeros((3,1))
 		min_value = 10000000
 		for element in mu:
 			if (np.linalg.norm(np.subtract(point, element), ord = 2))**2 < min_value:
 				min_value = (np.linalg.norm(np.subtract(point, element)))**2
 				closest = element
 		#Now we have the closest centroid and we replace it in the original image
 		Q[i,j,0] = closest[0]
 		Q[i,j,1] = closest[1]
 		Q[i,j,2] = closest[2]

plt.imshow(Q)
plt.show()



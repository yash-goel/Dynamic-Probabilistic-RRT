# gaussian update of robot motion visualization
import numpy as np
import matplotlib.pyplot as plt

delt = 0.1

A = np.array([[1,0],[0,1]])
B = 1

def robot_mean_update(x_prev, u_prev):

	x_new = np.dot(A,x_prev) + np.dot(B,u_prev)*delt
	return x_new

def robot_covariance_update(cov_prev, cov_state):

	cov_new = np.dot(np.dot(A,cov_prev),np.transpose(A)) + cov_state
	return cov_new

def multivariate_gauss(pos, mu, sigma):

	n = mu.shape[0]
	sigma_det = np.linalg.det(sigma)
	sigma_inv = np.linalg.inv(sigma)
	N = np.sqrt((2*np.pi)**n*sigma_det)
	fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)

	return np.exp(-fac/2)/N

def multi_gauss(mu,sigma):

	num = 60
	x = np.linspace(-3,3,num)
	y = np.linspace(-3,4,num)
	x, y = np.meshgrid(x,y)

	## x and y into 3d matrix
	pos = np.empty(x.shape + (2,))
	pos[:,:,0] = x
	pos[:,:,1] = y

	return x,y,multivariate_gauss(pos, mu, sigma)

cov_state = np.array([[0.1,-0.05],[-0.05,0.15]])
cov_prev = cov_state
x_prev = np.array([0,0])

i = 1
x_pos = np.array([0])
y_pos = np.array([0])

while i < 10:

	u_prev = np.array([1,1])
	x_new = robot_mean_update(x_prev, u_prev)
	x_pos = np.append(x_pos, x_new[0])
	y_pos = np.append(y_pos, x_new[1])

	cov_new = robot_covariance_update(cov_prev, cov_state)
	x,y,z = multi_gauss(x_new, cov_new)
	
	## plot the mean cov for new robot position
	plt.clf()
	cs = plt.contour(x,y,z,1)
	plt.plot(x_pos,y_pos)
	plt.xlim(-5,5)
	plt.ylim(-5,5)
	plt.show()
	
	## update the mean prev cov prev
	x_prev = x_new
	cov_prev = cov_new

	i += 1


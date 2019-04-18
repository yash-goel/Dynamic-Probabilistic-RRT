## rrt utils
import numpy as np
import math

mu = np.array([1,2])
sigma = np.array([[0.1,0.05],[0.05,0.1]])

def bhattacharyya_distance(mu_1, mu_2, sigma_1, sigma_2):

	mean_diff = np.array(mu_1 - mu_2).reshape(2,1)
	sigma_com = np.array(sigma_1 + sigma_2).reshape(2,2)
	first_term = np.matmul(np.transpose(mean_diff),np.linalg.inv(sigma_com))
	second_term = np.matmul(first_term,mean_diff)
	dist = second_term/8 + 0.5*np.log(np.linalg.det(sigma_com)/(np.sqrt(np.linalg.det(sigma_1)*np.linalg.det(sigma_2))))
	return dist

def draw_cov(mu=mu,sigma=sigma, p=0.95):

	s = -2*math.log(1-p)
	eig_value, eig_vector = np.linalg.eig(s*sigma)
	
	min_eig_val = np.min(eig_value)
	max_eig_val = np.max(eig_value)

	min_eig_vec = eig_vector[:,np.argmin(eig_value)]
	max_eig_vec = eig_vector[:,np.argmax(eig_value)]

	angle = np.arctan2(max_eig_vec[1], max_eig_vec[0])

	if angle<0:
		angle += 2*math.pi

	a = s*np.sqrt(max_eig_val)
	b = s*np.sqrt(min_eig_val)

	theta_list = np.linspace(0,2*math.pi)

	ellipse_x = a*np.cos(theta_list)
	ellipse_y = b*np.sin(theta_list)

	## rotation matrix 
	R = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
	x = [ellipse_x, ellipse_y]
	r_ellip = np.matmul(R,x)

	return r_ellip[0,:] + mu[0], r_ellip[1,:] + mu[1] 




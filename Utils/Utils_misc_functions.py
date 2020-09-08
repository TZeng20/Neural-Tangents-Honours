import numpy as onp
import math
import jax.numpy as np
import matplotlib.pyplot as plt
from functools import partial
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import pickle
from jax.config import config
config.update("jax_enable_x64", True)
import neural_tangents as nt  # 64-bit precision enabled
from neural_tangents import stax
from jax import random, vmap

# METRICS

# MSE loss
loss = lambda y_hat, y: 0.5 * np.mean((y_hat - y) ** 2)

# Squared MSE
rmse = lambda fx, y_hat: np.sqrt(np.mean((fx - y_hat)**2))

# Accuracy
def accuracy(predicted, targets):
	target_class = np.argmax(targets, axis=1)
	predicted_class = np.argmax(predicted, axis = 1)
	return np.mean(predicted_class == target_class)

# F1
def f1_score(predicted, targets, depth = 10):
	predicted = np.argmax(predicted, axis = 1)
	predicted = tf.one_hot(predicted, depth = depth).numpy()
	true_postives = (predicted*targets).sum()
	false_positives = (predicted*(1-targets)).sum()
	false_negatives = ((1-predicted)*targets).sum()
	precision = true_postives/(true_postives + false_positives)
	recall = true_postives/(true_postives + false_negatives)
	return 2*(precision*recall)/(precision + recall)

def repeat_along_diag(a, r):
	m,n = a.shape
	out = onp.zeros((r,m,r,n), dtype=a.dtype)
	diag = onp.einsum('ijik->ijk',out)
	diag[:] = a
	return out.reshape(-1,n*r)

def marginal_log_likelihood(Y, cov, dim = 10):
	'''
	Calculates the marginal log likelihood of predictions
	'''
	# Shape (n, 1)
	y = Y.flatten('F')[:np.newaxis] 
	# Shape (1,n)
	y_t = Y.flatten('F')[np.newaxis, :]
	# Shape (nd x nd)
	cov_inv = np.linalg.inv(cov)
	# Shape (nd x nd)
	cov_inv_big = repeat_along_diag(cov_inv, dim)
	likelihood_term = -0.5*(y_t @ cov_inv_big @ y)
	
	cov_determinant = np.linalg.det(cov)
	determinant_term = -0.5*dim*np.log(cov_determinant + 1e-8)
	
	constant_term = -((dim*y.shape[0])/2)*np.log(2*np.pi)
	return likelihood_term + determinant_term + constant_term

def process_data(data_chunk, depth = 10, flatten = False):
	"""
	Standardise the images and one-hot encode the labels.
	"""
	image, label = tf.cast(data_chunk['image'], tf.float64), data_chunk['label']
	standardised_images = tf.image.per_image_standardization(image)
	if flatten == True:
		standardised_images = tf.reshape(standardised_images, (image.shape[0], -1))
	label = tf.one_hot(label, depth = depth)
	label = label - 0.1
	return {'image': standardised_images.numpy(), 'label': label.numpy()}

def generate_datasets(train_size, test_size = 20, flatten = False):
	'''
	Load raw dataset and then preprocess them. 
	'''
	ds_train, ds_test = tfds.load(
		'cifar10', split=['train[:%d]' % train_size,'test[:%d]' % test_size], batch_size=-1)
	permutation = onp.random.RandomState(seed=42).permutation(train_size)
	train_data = process_data(ds_train, flatten = flatten) 
	train_data['image'] = train_data['image'][permutation]
	train_data['label'] = train_data['label'][permutation]
	test_data = process_data(ds_test, flatten = flatten)
	return train_data, test_data
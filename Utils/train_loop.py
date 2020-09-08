import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import math

import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import random, vmap
from jax.api import jit, grad, jacobian
from jax.experimental import optimizers

import neural_tangents as nt  # 64-bit precision enabled
from neural_tangents import stax

# MSE loss
loss = lambda y_hat, y: 0.5 * np.mean((y_hat - y) ** 2)
# Squared MSE
rmse = lambda fx, y_hat: np.sqrt(np.mean((fx - y_hat)**2))


def _get_empirical_predictor(x_train, y_train, x_test, apply_fn, initial_params, learning_rate):

	ntk = nt.batch(nt.empirical_ntk_fn(apply_fn),batch_size=1, store_on_device = False)

	g_dd = ntk(x_train, x_train, initial_params)
	g_td = ntk(x_test, x_train, initial_params)

	predictor_emp = nt.predict.gradient_descent_mse(
		ntk_train_train = g_dd,
		y_train = y_train,
		learning_rate = learning_rate)

	predictor_emp_partial = partial(predictor_emp, ntk_test_train = g_td)
	return predictor_emp_partial


def full_batch_train(x_train, y_train, x_test, y_test,
	apply_fn, kernel_fn, initial_params,
	learning_rate=0.01, epochs = 100, print_frequency = 10):

	'''
	apply_fn: Foward NN function
	kernel_fn: kernel_fn of infinite network. 
	Can call using kernel_fn(x1,x2, 'nngp or ntk')
	initial_params: initial params of NN
	'''

	# Create optimizer
	opt_init, opt_apply, get_params = optimizers.sgd(learning_rate)
	# Initial state of params and some more information about them
	state = opt_init(initial_params)
	# Calculates the gradients with respect to the network parameters
	grad_loss = jit(grad(
		lambda params, x, y: loss(apply_fn(params, x), y)))

	# Analytical
	predictor_an = nt.predict.gradient_descent_mse_ensemble(
		kernel_fn=kernel_fn,
		x_train=x_train,
		y_train=y_train,
		learning_rate=learning_rate)

	# Empirical
	predictor_emp = _get_empirical_predictor(
		x_train, y_train, x_test,
		apply_fn, initial_params, learning_rate)

	# f_hat_0 (predictions of network at initialization)
	f0_train = apply_fn(initial_params, x_train)
	f0_test = apply_fn(initial_params, x_test)
	
	train_losslist = []
	test_losslist = []
	train_preds_og_list = []
	test_preds_og_list = []

	train_losslist_an = []
	test_losslist_an = []
	train_preds_an_list = []
	test_preds_an_list = []

	train_losslist_emp = []
	test_losslist_emp = []
	train_preds_emp_list = []
	test_preds_emp_list = []

	for i in range(epochs):
		t = i
		# Update params
		params = get_params(state)
		dloss_dparams = grad_loss(params, x_train, y_train) # loss w.r.t parameters
		state = opt_apply(i, dloss_dparams, state) #update parameters
		
		# Loss and predictions original network
		train_preds = apply_fn(params, x_train)
		train_loss = loss(train_preds, y_train)
		test_preds = apply_fn(params, x_test)
		test_loss = loss(test_preds, y_test)

		train_losslist.append(train_loss)
		test_losslist.append(test_loss)
		train_preds_og_list.append(train_preds)
		test_preds_og_list.append(test_preds)
		
		# Loss and predictions analytical
		#train_preds_an, test_preds_an = predictor_an(t, f0_train, f0_test, analytical_td)
		train_preds_an = predictor_an(t, x_train, 'ntk')
		test_preds_an = predictor_an(t, x_test, 'ntk')
		train_loss_an = loss(train_preds_an, y_train)
		test_loss_an = loss(test_preds_an, y_test)
		
		train_losslist_an.append(train_loss_an)
		test_losslist_an.append(test_loss_an)
		train_preds_an_list.append(train_preds_an)
		test_preds_an_list.append(test_preds_an)
		
		# Loss and predictions empirical
		train_preds_emp, test_preds_emp = predictor_emp(t, f0_train, f0_test)
		train_loss_emp = loss(train_preds_emp, y_train)
		test_loss_emp = loss(test_preds_emp, y_test)
		
		train_losslist_emp.append(train_loss_emp)
		test_losslist_emp.append(test_loss_emp)
		train_preds_emp_list.append(train_preds_emp)
		test_preds_emp_list.append(test_preds_emp)

		if i%print_frequency==0:
			print(i)
			print('original test loss is ', test_loss)
			print('analytical test loss is ', test_loss_an)
			print('empirical test loss is ', test_loss_emp)

	return {'og': [train_losslist, test_losslist, train_preds_og_list, test_preds_og_list],
	'an': [train_losslist_an, test_losslist_an, train_preds_an_list, test_preds_an_list],
	'emp': [train_losslist_emp, test_losslist_emp, train_preds_emp_list, test_preds_emp_list]}
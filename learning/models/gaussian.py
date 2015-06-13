#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from learning.models.rws import TopModule, Module, theano_rng
from learning.model import default_weights

_logger = logging.getLogger(__name__)
floatX = theano.config.floatX


class FixedDiagonalGaussianTop(TopModule):
    """ DiagonalGaussian top layer """ 
    def __init__(self, **hyper_params):
        super(FixedDiagonalGaussianTop, self).__init__()
        
        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')

        self.set_hyper_params(hyper_params)
    
    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor 
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        mu         = np.zeros(n_X, dtype=floatX)
        log_sigma2 = np.zeros(n_X, dtype=floatX)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*T.log(2*np.pi) - 0.5*log_sigma2 -0.5*(X-mu)**2 / T.exp(log_sigma2)
        log_prob = log_prob.sum(axis=1)

        return log_prob

    def sample(self, n_samples):
        """ Sample from this toplevel module and return X ~ P(X), log(P(X))

        Parameters
        ----------
        n_samples:
            number of samples to drawn

        Returns
        -------
        X:      T.tensor
            samples from this module
        log_p:  T.tensor
            log-probabilities for the samples returned in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        mu         = np.zeros(n_X, dtype=floatX)
        log_sigma2 = np.zeros(n_X, dtype=floatX)

        # Samples from multivariate diagonal Gaussian
        X = theano_rng.normal(
                size=(n_samples, n_X), 
                avg=mu, std=T.exp(0.5*log_sigma2),
                dtype=floatX)

        return X, self.log_prob(X)


class DiagonalGaussianTop(TopModule):
    """ DiagonalGaussian top layer """ 
    def __init__(self, **hyper_params):
        super(DiagonalGaussianTop, self).__init__()
        
        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')

        # Model parameters
        self.register_model_param('mu', help='gaussian mean', 
            default=lambda: np.zeros(self.n_X))
        self.register_model_param('log_sigma2', help='log sigma squared', 
            default=lambda: np.zeros(self.n_X))

        self.set_hyper_params(hyper_params)
    
    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor 
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        mu, log_sigma2 = self.get_model_params(['mu', 'log_sigma2'])

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*T.log(2*np.pi) - 0.5*log_sigma2 -0.5*(X-mu)**2 / T.exp(log_sigma2)
        log_prob = log_prob.sum(axis=1)

        return log_prob

    def sample(self, n_samples):
        """ Sample from this toplevel module and return X ~ P(X), log(P(X))

        Parameters
        ----------
        n_samples:
            number of samples to drawn

        Returns
        -------
        X:      T.tensor
            samples from this module
        log_p:  T.tensor
            log-probabilities for the samples returned in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        mu, log_sigma2 = self.get_model_params(['mu', 'log_sigma2'])

        # Sample from mean-zeros std.-one Gaussian 
        eps = theano_rng.normal(
                size=(n_samples, n_X), 
                avg=0., std=1., dtype=floatX)

        self.random_source = [eps]

        # ... and sca1le/translate samples
        X = eps * T.exp(0.5*log_sigma2) + mu

        return X, self.log_prob(X)

#----------------------------------------------------------------------------

class DiagonalGaussian(Module):
    """
    A stochastic diagonal gaussian parameterized by

    P(x | y):

        x ~ DiagonalGaussian(x; \mu, \sigma)

        \mu           = W_mu a0 + b_mu
        \log \sigma^2 = W_ls a0 + b_ls
           with    a0 = tanh(W0 a1 + b0)    (dimension n_hid[0])
            and    a1 = tanh(W1 a2 + b1)    (dimension n_hid[1])
                      ...
                   an = tanh(Wn  y  + bn)     (dimension n_hid[n])  
    """
    def __init__(self, **hyper_params):
        super(DiagonalGaussian, self).__init__()

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')
        self.register_hyper_param('n_hid', help='no. hidden variables')
        self.register_hyper_param('log_sigma2_min', help='log(sigma**2) cutoff', default=-np.inf)
        self.register_hyper_param('final_tanh', help='', default=False)
        self.register_hyper_param('initial_tanh', help='', default=False)
    
        self.set_hyper_params(hyper_params)

        # sanitize hyper params
        if isinstance(self.n_hid, int):
            self.n_hid = (self.n_hid, )
        if isinstance(self.n_hid, list):
            self.n_hid = tuple(self.n_hid)
        self.n_layers = len(self.n_hid)
        
        hidden_size = self.n_hid + (self.n_Y,)
        for i in reversed(xrange(self.n_layers)):
            W_name = "W%d" % i
            b_name = "b%d" % i

            n_upper = hidden_size[i+1]
            n_lower = hidden_size[i]

            def create_W_init(n_upper, n_lower):
                return lambda: default_weights(n_upper, n_lower)
            def create_b_init(n_lower):
                return lambda: np.zeros(n_lower)
            
            self.register_model_param(W_name, default=create_W_init(n_upper, n_lower))
            self.register_model_param(b_name, default=create_b_init(n_lower))

        # lowers layer parametrization... higher layers will be parameterized in self.setup()
        self.register_model_param('W_mu', default=lambda: default_weights(hidden_size[0], self.n_X))
        self.register_model_param('W_ls', default=lambda: default_weights(hidden_size[0], self.n_X))
        self.register_model_param('b_mu', default=lambda: np.zeros(self.n_X))
        self.register_model_param('b_ls', default=lambda: np.zeros(self.n_X))

    def setup(self):
        super(DiagonalGaussian, self).setup()

    def log_prob(self, X, Y):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer
        X:      T.tensor
            samples from the lower layer

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X and Y
        """
        W_mu, W_ls = self.get_model_params(['W_mu', 'W_ls'])
        b_mu, b_ls = self.get_model_params(['b_mu', 'b_ls'])

        # Compute hidden layer activations...
        ai = Y
        if self.initial_tanh:
            ai = T.tanh(ai)
        for i in reversed(range(self.n_layers)):
            Wi = self.get_model_param('W%d'%i)
            bi = self.get_model_param('b%d'%i)
            ai = T.tanh(T.dot( ai, Wi) + bi)

        # ... and Gaussian params
        mu = T.dot(ai, W_mu) + b_mu
        if self.final_tanh:
            mu = T.tanh(mu)
        log_sigma2 = T.dot(ai, W_ls) + b_ls
        log_sigma2 = T.maximum(log_sigma2, self.log_sigma2_min)

        # Calculate multivariate diagonal Gaussian
        log_prob =  -0.5*T.log(2*np.pi) - 0.5*log_sigma2 -0.5*(X-mu)**2 / T.exp(log_sigma2)
        log_prob = T.sum(log_prob, axis=1)

        return log_prob

    def sample(self, Y):
        """ Given samples from the upper layer Y, sample values from X
            and return then together with their log probability.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
            samples from the lower layer
        log_p:  T.tensor
            log-posterior for the samples returned in X
        """
        n_X, = self.get_hyper_params(['n_X'])

        W_mu, W_ls = self.get_model_params(['W_mu', 'W_ls'])
        b_mu, b_ls = self.get_model_params(['b_mu', 'b_ls'])

        n_samples = Y.shape[0]

        # Compute hidden layer activations...
        ai = Y
        if self.initial_tanh:
            ai = T.tanh(ai)
        for i in reversed(range(self.n_layers)):
            Wi = self.get_model_param('W%d'%i)
            bi = self.get_model_param('b%d'%i)
            ai = T.tanh(T.dot(ai, Wi) + bi)

        # ... and Gaussian params
        mu = T.dot(ai, W_mu) + b_mu
        if self.final_tanh:
            mu = T.tanh(mu)
        log_sigma2 = T.dot(ai, W_ls) + b_ls
        log_sigma2 = T.maximum(log_sigma2, self.log_sigma2_min)

        # Sample from mean-zeros std.-one Gaussian 
        eps = theano_rng.normal(
                size=(n_samples, n_X), 
                avg=0., std=1., dtype=floatX)
        self.random_source = [eps]

        # ... and scale/translate samples
        X = eps * T.exp(0.5*log_sigma2) + mu

        return X, self.log_prob(X, Y)

    def sample_expected(self, Y):
        """ Given samples from the upper layer Y, return 
            the probability for the individual X elements

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
        """
        n_X, = self.get_hyper_params(['n_X'])

        W_mu, W_ls = self.get_model_params(['W_mu', 'W_ls'])
        b_mu, b_ls = self.get_model_params(['b_mu', 'b_ls'])

        n_samples = Y.shape[0]

        # Compute hidden layer activations...
        ai = Y
        if self.initial_tanh:
            ai = T.tanh(ai)
        for i in reversed(range(self.n_layers)):
            Wi = self.get_model_param('W%d'%i)
            bi = self.get_model_param('b%d'%i)
            ai = T.tanh(T.dot( ai, Wi) + bi)

        # ... and Gaussian params
        mu = T.dot(ai, W_mu) + b_mu
        if self.final_tanh:
            mu = T.tanh(mu)

        return mu

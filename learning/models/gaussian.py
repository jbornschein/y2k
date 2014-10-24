#!/usr/bin/env python 

from __future__ import division

import logging

import numpy as np

import theano 
import theano.tensor as T
from theano.printing import Print

from learning.rws import TopModule, Module, theano_rng
from learning.model import default_weights

_logger = logging.getLogger(__name__)
floatX = theano.config.floatX


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

        # Samples from multivariate diagonal Gaussian
        X = theano_rng.normal(
                size=(n_samples, n_X), 
                avg=mu, std=T.exp(0.5*log_sigma2),
                dtype=floatX)

        return X, self.log_prob(X)

#----------------------------------------------------------------------------

class DiagonalGaussian(Module):
    """
    A stochastic diagonal gaussian parameterized by

    P(x | y):

        x ~ DiagonalGaussian(x; \mu, \sigma)

        \mu           = W1 h + b1
        \log \sigma^2 = W2 h + b2
           with     h = tanh(W3 y + b3)
    """
    def __init__(self, **hyper_params):
        super(DiagonalGaussian, self).__init__()

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')
        self.register_hyper_param('n_hid', help='no. hidden variables')

        # MLP parametrization
        self.register_model_param('W1', default=lambda: default_weights(self.n_hid, self.n_X))
        self.register_model_param('W2', default=lambda: default_weights(self.n_hid, self.n_X))
        self.register_model_param('W3', default=lambda: default_weights(self.n_Y, self.n_hid))
        self.register_model_param('b1', default=lambda: np.zeros(self.n_X))
        self.register_model_param('b2', default=lambda: np.zeros(self.n_X))
        self.register_model_param('b3', default=lambda: np.zeros(self.n_hid))

        self.set_hyper_params(hyper_params)

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
        W1, W2, W3 = self.get_model_params(['W1', 'W2', 'W3'])
        b1, b2, b3 = self.get_model_params(['b1', 'b2', 'b3'])

        # Compute gaussian params...
        h = T.tanh(T.dot(Y, W3) + b3)
        mu = T.dot(h, W1) + b1
        log_sigma2 = T.dot(h, W2) + b2

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

        W1, W2, W3 = self.get_model_params(['W1', 'W2', 'W3'])
        b1, b2, b3 = self.get_model_params(['b1', 'b2', 'b3'])

        n_samples = Y.shape[0]

        # Compute gaussian params...
        h = T.tanh(T.dot(Y, W3) + b3)
        mu = T.dot(h, W1) + b1
        log_sigma2 = T.dot(h, W2) + b2

        X = theano_rng.normal(
                size=(n_samples, n_X), 
                avg=mu, std=T.exp(0.5*log_sigma2),
                dtype=floatX)

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

        W1, W2, W3 = self.get_model_params(['W1', 'W2', 'W3'])
        b1, b2, b3 = self.get_model_params(['b1', 'b2', 'b3'])

        n_samples = Y.shape[0]

        # Compute gaussian params...
        h = T.tanh(T.dot(Y, W3) + b3)
        mu = T.dot(h, W1) + b1

        return mu


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


#----------------------------------------------------------------------------

class NoisyOR(Module):
    """ SigmoidBeliefLayer """
    def __init__(self, **hyper_params):
        super(NoisyOR, self).__init__()

        self.tol = 1e-5

        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')
        self.register_hyper_param('type', help='type og NoisyOR layer')

        # Sigmoid Belief Layer
        self.register_model_param('W',  help='weights', default=lambda: default_weights(self.n_Y, self.n_X) )
        self.register_model_param('W_', help='weights', default=lambda: default_weights(self.n_Y, self.n_X) )

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
        n_X, n_Y = self.get_hyper_params(['n_X', 'n_Y'])
        W, W_    = self.get_model_params(['W', 'W_'])
        #W,       = self.get_model_params(['W'])

        n_samples = Y.shape[0]

        # posterior P(X|Y)
        Y = Y.reshape( (n_samples, 1, n_Y) ) 
        prob_X  = 1 - T.prod(1. - T.shape_padleft(W.T) * Y, axis=2)
        prob_X += 1 - T.prod(1. - T.shape_padleft(W_.T) * (1-Y), axis=2)
        prob_X  = T.maximum(prob_X, self.tol) 
        prob_X  = T.minimum(prob_X, 1-self.tol) 

        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
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
        n_X, n_Y = self.get_hyper_params(['n_X', 'n_Y'])
        W, W_    = self.get_model_params(['W', 'W_'])
        #W,       = self.get_model_params(['W'])

        n_samples = Y.shape[0]

        # sample X given Y
        Y = Y.reshape( (n_samples, 1, n_Y) ) 
        prob_X  = 1 - T.prod(1. - T.shape_padleft(W.T) * Y, axis=2)
        prob_X += 1 - T.prod(1. - T.shape_padleft(W_.T) * (1-Y), axis=2)
        prob_X  = T.maximum(prob_X, self.tol) 
        prob_X  = T.minimum(prob_X, 1-self.tol) 

        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= prob_X, dtype=floatX)

        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return X, log_prob

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
        n_X, n_Y = self.get_hyper_params(['n_X', 'n_Y'])
        #W,       = self.get_model_params(['W'])
        W, W_    = self.get_model_params(['W', 'W_'])

        n_samples = Y.shape[0]

        # sample X given Y
        Y = Y.reshape( (n_samples, 1, n_Y) ) 
        prob_X  = 1 - T.prod(1. - T.shape_padleft(W.T) * Y, axis=2)
        prob_X += 1 - T.prod(1. - T.shape_padleft(W_.T) * (1-Y), axis=2)
        prob_X  = T.maximum(prob_X, self.tol) 
        prob_X  = T.minimum(prob_X, 1-self.tol) 

        return prob_X


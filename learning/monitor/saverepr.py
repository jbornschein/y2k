#!/usr/bin/env python 

from __future__ import division

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

from learning.dataset import DataSet
from learning.model import Model
from learning.models.rws import f_replicate_batch, f_logsumexp
from learning.monitor import Monitor, reasonable_batch_size
import learning.utils.datalog as datalog


_logger = logging.getLogger()


class SaveRepr(Monitor):
    """ Save latent representation of a dataset """
    
    def __init__(self, data, name=None, n_samples=10):
        if name is None:
            name="saverepr"
        super(SaveRepr, self).__init__(name)

        assert isinstance(data, DataSet)
        self.dataset = data

        assert isinstance(n_samples, int)
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model
        
        p_layers = model.p_layers
        n_layers = len(p_layers)
        dataset = self.dataset

        n_samples = T.iscalar("n_samples")
        X_batch = T.fmatrix('X_batch')
        Y_batch = T.fmatrix('Y_batch')
        
        batch_size = X_batch.shape[0]

        # prepare batches
        X_batch, Y_batch = dataset.late_preproc(X_batch, Y_batch)
        X = f_replicate_batch(X_batch, n_samples)
        Y = f_replicate_batch(Y_batch, n_samples)

        # sample hidden layers
        samples, log_p, log_q = model.sample_q(X, Y)

        # reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        log_q_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, n_samples, p_layers[l].n_X))
            log_q[l] = log_q[l].reshape((batch_size, n_samples))
            log_p[l] = log_p[l].reshape((batch_size, n_samples))
            log_p_all += log_p[l]   # agregate all layers
            log_q_all += log_q[l]   # agregate all layers

        # Approximate log P(X)
        log_px = f_logsumexp(log_p_all-log_q_all, axis=1) - T.log(n_samples)

        self.do_sample_h = theano.function(
                        inputs=[X_batch, Y_batch, n_samples],
                        outputs=[samples[-1], log_p_all],
                        name="do_sample_h", 
                        on_unused_input="ignore",
                )

    def on_init(self, model):
        self.compile()

    def on_iter(self, model):
        n_samples = n_samples
        dataset = self.dataset
        X = dataset.X
        Y = dataset.Y

        n_datapoints = dataset.n_datapoints
        batch_size = reasonable_batch_size(n_samples)
        n_batches = dataset.n_datapoints 
        first = 0
        for n in xrange(0, n_datapoints):
            #last = first + batch_size
            first = n 
            last = n+1
            samples, log_p = self.do_sample_h(X[first:last], Y[first:last], n_samples)

            self.dlog.append_all({
                'sample_h': samples[0],
                'log_p'   : log_p[0]
                'y'       : Y[n],
            })


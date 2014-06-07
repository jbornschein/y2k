#!/usr/bin/env python 

from __future__ import division

import abc
import logging

import numpy as np

import theano 
import theano.tensor as T

from dataset import DataSet
from model import Model
from hyperbase import HyperBase
import utils.datalog as datalog

_logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------
class Monitor(HyperBase):
    """ Abtract base class to monitor stuff """
    __metaclass__ = abc.ABCMeta

    def __init__(self):   
        """ 
        """
        self.dlog = datalog.getLogger(__name__)
        self.logger = logging.getLogger(__name__)

    def compile(self):
        pass

    def on_init(self, model):
        """ Called after the model has been initialized; directly before 
            the first learning epoch will be performed 
        """
        pass

    @abc.abstractmethod
    def on_iter(self, model):
        """ Called whenever a full training epoch has been performed
        """
        pass


#-----------------------------------------------------------------------------
class DLogHyperParams(Monitor):
    def __init__(self):
        super(DLogHyperParams, self).__init__()

    def on_iter(self, model):
        model.hyper_params_to_dlog(self.dlog)


#-----------------------------------------------------------------------------
class DLogModelParams(Monitor):
    """
    Write all model parameters to a DataLogger called "model_params".
    """
    def __init__(self):
        super(DLogModelParams, self).__init__()

    def on_iter(self, model):
        self.logger.info("Saving model parameters")
        model.model_params_to_dlog(self.dlog)


#-----------------------------------------------------------------------------
class MonitorLL(Monitor):
    """ Monitor the LL after each training epoch on an arbitrary 
        test or validation data set
    """
    def __init__(self, data, n_samples):
        super(MonitorLL, self).__init__()

        assert isinstance(data, DataSet)
        self.dataset = data

        if isinstance(n_samples, int):
            n_samples = [n_samples]
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        dataset = self.dataset
        X, Y = dataset.preproc(dataset.X, dataset.Y)
        self.X = theano.shared(X, "X")
        self.Y = theano.shared(Y, "Y")

        batch_idx  = T.iscalar('batch_idx')
        batch_size = T.iscalar('batch_size')

        self.logger.info("compiling do_loglikelihood")
        n_samples = T.iscalar("n_samples")
        batch_idx = T.iscalar("batch_idx")
        batch_size = T.iscalar("batch_size")

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch, Y_batch = dataset.late_preproc(self.X[first:last], self.Y[first:last])
        
        log_PX, _, _, _, KL, Hp, Hq = model.log_likelihood(X_batch, n_samples=n_samples)
        batch_log_PX = T.sum(log_PX)
        batch_KL = [T.sum(kl) for kl in KL]
        batch_Hp = [T.sum(hp) for hp in Hp]
        batch_Hq = [T.sum(hq) for hq in Hq]

        self.do_loglikelihood = theano.function(  
                            inputs=[batch_idx, batch_size, n_samples], 
                            outputs=[batch_log_PX] + batch_KL + batch_Hp + batch_Hq, 
                            name="do_likelihood")

    def on_init(self, model):
        self.compile(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_datapoints = self.dataset.n_datapoints

        all_log_px = np.zeros(n_datapoints)   # HACK
        
        #
        for K in n_samples:
            if K <= 10:
                batch_size = 100
            elif K <= 100:
                batch_size = 10
            else:
                batch_size = 1
    
            n_layers = len(model.p_layers)

            L = 0
            KL = np.zeros(n_layers)
            Hp = np.zeros(n_layers)
            Hq = np.zeros(n_layers)
        
            # Iterate over dataset
            for batch_idx in xrange(n_datapoints//batch_size):
                outputs = self.do_loglikelihood(batch_idx, batch_size, K)
                batch_L , outputs = outputs[0], outputs[1:]
                batch_KL, outputs = outputs[:n_layers], outputs[n_layers:]
                batch_Hp, outputs = outputs[:n_layers], outputs[n_layers:]
                batch_Hq          = outputs[:n_layers]
                
                all_log_px[batch_idx] = batch_L

                L += batch_L
                KL += np.array(batch_KL)
                Hp += np.array(Hp)
                Hq += np.array(Hq)
                


            L /= n_datapoints
            KL /= n_datapoints
            Hp /= n_datapoints
            Hq /= n_datapoints

            prefix = "%d." % K

            global validation_LL
            validation_LL = L

            bound = np.std(all_log_px, ddof=1) / np.sqrt(n_datapoints)

            self.logger.info("MonitorLL (%d datpoints, %d samples): LL=%5.2f (+-%3.2f) KL=%s" % (n_datapoints, K, L, bound, KL))
            self.dlog.append_all({
                prefix+"log_px": all_log_px,
                prefix+"LL_bound": bound,
                prefix+"LL": L,
                prefix+"KL": KL,
                prefix+"Hp": Hp,
                prefix+"Hq": Hq,
            })
        

#-----------------------------------------------------------------------------
class SampleFromP(Monitor):
    """ Draw a number of samples from the P-Model """
    def __init__(self, n_samples=100, data=None):
        super(SampleFromP, self).__init__()

        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)

        self.logger.info("compiling do_sample")

        n_samples = T.iscalar('n_samples')
        n_samples.tag.test_value = self.n_samples
        samples, log_p = model.sample_p(n_samples)

        self.do_sample = theano.function(
                            inputs=[n_samples],
                            outputs=[log_p] + samples,
                            name="do_sample")

    def on_init(self, model):
        self.compile(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_layers = len(model.p_layers)

        outputs = self.do_sample(n_samples)
        log_p = outputs[0]
        samples = outputs[1:]

        self.logger.info("SampleFromP(n_samples=%d)" % n_samples)
        self.dlog.append("log_p", log_p)
        for l in xrange(n_layers):
            prefix = "L%d" % l
            self.dlog.append(prefix, samples[l])


#-----------------------------------------------------------------------------
from isws import f_replicate_batch, f_logsumexp

class GradDetail(Monitor):
    def __init__(self, data, n_samples):
        super(GradDetail, self).__init__()

        assert isinstance(data, DataSet)
        self.dataset = data

        assert isinstance(n_samples, int)
        self.n_samples = n_samples

    def compile(self, model):
        assert isinstance(model, Model)
        self.model = model

        dataset = self.dataset
        X, Y = dataset.preproc(dataset.X, dataset.Y)
        self.X = theano.shared(X, "X")
        self.Y = theano.shared(Y, "Y")

        #---------------------------------------------------------------------------------
        self.logger.info("compiling do_grad_detail")

        n_samples  = self.n_samples
        idx        = T.iscalar('idx')
        batch_size = 1

        X_batch, Y_batch = dataset.late_preproc(self.X[idx:idx+1], self.Y[idx:idx+1])
        
        p_layers = model.p_layers
        q_layers = model.q_layers
        n_layers = len(p_layers)

        # Prepare input for layers
        samples = [None]*n_layers
        log_q   = [None]*n_layers
        log_p   = [None]*n_layers

        samples[0] = f_replicate_batch(X_batch, n_samples)                   # 
        log_q[0]   = T.zeros([batch_size*n_samples])

        # Generate samples (feed-forward)
        for l in xrange(n_layers-1):
            samples[l+1], log_q[l+1] = q_layers[l].sample(samples[l])
        
        # Get log_probs from generative model
        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in xrange(n_layers-1, 0, -1):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])

        # Reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        log_q_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, n_samples, p_layers[l].n_X))
            log_q[l] = log_q[l].reshape((batch_size, n_samples))
            log_p[l] = log_p[l].reshape((batch_size, n_samples))

            log_p_all += log_p[l]   # agregate all layers
            log_q_all += log_q[l]   # agregate all layers

        # Unnormalized sampling weights
        log_w = log_p_all - log_q_all     # shape: (1, n_samples)

        # Approximate P(X)
        log_px = f_logsumexp(log_w, axis=1)

        cost_p = log_p_all
        cost_q = log_q_all
        
        gradients = []
        for s in xrange(n_samples):
            for nl, layer in enumerate(p_layers):
                for name, shvar in layer.get_model_params().iteritems():
                    if name == 'W' and nl == 1:
                        gradients.append(T.grad(cost_p[0,s], shvar)) #, consider_constant=[w]))
 
        self.do_grad_detail = theano.function(  
                            inputs=[idx], 
                            outputs=[log_px, log_w] + gradients,
                            name="do_grad_detail")

    def on_init(self, model):
        self.compile(model)

    def on_iter(self, model):
        n_samples = self.n_samples
        n_datapoints = self.dataset.n_datapoints

        prefix = "graddetail."
        n_layers = len(model.p_layers)

        # Iterate over dataset
        for n in xrange(n_datapoints):
            print "... datapoint %d ..." % n
            outputs = self.do_grad_detail(n)
            log_px, log_w, outputs = outputs[0], outputs[1], outputs[2:]
                
            self.dlog.append_all({
                prefix+"log_px": log_px,
                prefix+"log_w" : log_w,
            })

            for s in xrange(n_samples):
                for nl, layer in enumerate(model.p_layers):
                    for name, shvar in layer.get_model_params().iteritems():
                        if name == 'W' and nl == 1:
                            grad, outputs = outputs[0], outputs[1:]
                    
                            #import pdb; pdb.set_trace()
                            self.dlog.append("%sL%d.%s" % (prefix, nl, name), grad)


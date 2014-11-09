#!/usr/bin/env python 

from __future__ import division, print_function

import sys
sys.path.append("../")

import os
import os.path
import logging
from time import time
import cPickle as pickle

import numpy as np

import theano
import theano.tensor as T
import ipdb

from learning.rws  import LayerStack, f_replicate_batch

logger = logging.getLogger()

class Inpainter:
    def __init__(self, model, n_iterations=100, n_samples=1):
        self.model = model
        self.n_iterations = n_iterations
        self.n_samples = n_samples
    
        #--------------------------------------------------------------------
        # Prepare and compile theano function
        p_layers= model.p_layers
        q_layers= model.q_layers
        n_layers = len(p_layers)
        n_X = p_layers[0].n_X

        def f_corrupt(mask, src1, src2):
            mask = T.addbroadcast(mask, 0)
            return mask*src2 + (1-mask)*src1
        
        #n_samples = T.iscalar('n_samples')
        n_iterations = T.iscalar('n_iterations')
        X_batch = T.fmatrix('X_batch')
        mask = T.fmatrix('mask')

        #batch_size = X_batch.shape[0]

        def f_recons_iteration(prev_X, m, X):
            # Reconstruct with lowest layer
            h1, _ = q_layers[0].sample(prev_X)
            X_recons, _ = p_layers[0].sample(h1)
            X_recons = f_corrupt(m, X, X_recons)
            return X_recons

        X_recons, updates = theano.scan(
            fn=f_recons_iteration, 
            outputs_info=[T.zeros_like(X_batch)],
            non_sequences=(mask, X_batch),
            n_steps=n_iterations
        )

        X_recons = X_recons[-1]

        # Evaluate reconstructions
        #log_px, _, _, _, _, _, _ = model.log_likelihood(X_recons, n_samples=n_samples)
        #X_recons = X_recons.reshape( (batch_size, n_samples, n_X) )
        #log_px = log_px.reshape( (batch_size, n_samples) )
        #best = T.argmax(log_px, axis=1)
        #X_recons = X_recons[T.arange(batch_size), best].reshape( [batch_size, n_X] )
        #X_recons = X_recons[:, 0].reshape( [batch_size, n_X] )
        
        
        logger.info("Compiling do_inpainting...") 
        self.do_inpainting = theano.function(
            inputs=[X_batch, mask, n_iterations], 
            outputs=X_recons,
            updates=updates,
            allow_input_downcast=True,
        )

    def corrupt(self, mask, src1, src2=None):
        if src2 is None:
            src2 = np.zeros_like(src1)
        return mask*src2 + (1-mask)*src1

    def inpaint(self, X_batch, mask, n_iterations=None):
        model = self.model
        n_samples = self.n_samples
        if n_iterations is None:
            n_iterations = self.n_iterations

        X_recons = self.do_inpainting(X_batch, mask, n_iterations)

        return X_recons

def run_monitors(model, monitors):
    for m in monitors:
        m.on_iter(model)

def run_inpainting(args):
    from learning.utils.datalog import dlog, StoreToH5, TextPrinter

    from learning.experiment import Experiment
    from learning.monitor import MonitorLL, DLogModelParams, SampleFromP
    from learning.dataset import MNIST
    from learning.preproc import PermuteColumns, Binarize

    from learning.rws  import LayerStack, f_replicate_batch
    from learning.sbn  import SBN, SBNTop
    from learning.darn import DARN, DARNTop
    from learning.nade import NADE, NADETop

    import h5py

    logger.debug("Arguments %s" % args)
    tags = []

    # Layer models
    layer_models = {
        "sbn" : (SBN, SBNTop),
        "darn": (DARN, DARNTop), 
        "nade": (NADE, NADETop),
    }

    if not args.p_model in layer_models:
        raise "Unknown P-layer model %s" % args.p_model
    p_layer, p_top = layer_models[args.p_model]

    if not args.q_model in layer_models:
        raise "Unknown P-layer model %s" % args.p_model
    q_layer, q_top = layer_models[args.q_model]

    # Layer sizes
    layer_sizes = [int(s) for s in args.layer_sizes.split(",")]

    n_X = 28*28

    p_layers = []
    q_layers = []

    for ls in layer_sizes:
        n_Y = ls
        p_layers.append(
            p_layer(n_X=n_X, n_Y=n_Y, clamp_sigmoid=True)
        )
        q_layers.append(
            q_layer(n_X=n_Y, n_Y=n_X)
        )
        n_X = n_Y
    p_layers.append( p_top(n_X=n_X, clamp_sigmoid=True) )
            

    model = LayerStack(
        p_layers=p_layers,
        q_layers=q_layers
    )
    model.setup()

    # Dataset
    if args.shuffle:
        np.random.seed(23)
        preproc = [Binarize(late=False), PermuteColumns()]
        tags += ["shuffle"]
    else:
        preproc = [Binarize(late=False)]

    tags.sort()

    expname = args.cont
    if expname[-1] == "/":
        expname = expname[:-1]
    
    #result_dir = "reruns/%s" % os.path.basename(expname)
    #results_fname = result_dir+"/results.h5"
    #logger.info("Output logging to %s" % result_dir)
    #os.makedirs(result_dir)
    #dlog.set_handler("*", StoreToH5, results_fname)

    fname = args.cont + "/results.h5" 
    logger.info("Loading from %s" % fname)
    with h5py.File(fname, "r") as h5:
        #----------------------------------------------------------------------
        import pylab 

        logger.info("Loading model...")
        model.model_params_from_h5(h5, row=-1)

    #----------------------------------------------------------------------
    inpainter = Inpainter(model, n_iterations=100)


    #def make_mask():
    #    mask = np.zeros( (28, 28) )
    #    mask[:,0:14] = 1.
    #    mask = mask.reshape([1, 28*28])
    #    return mask

    def make_mask():
        border = 5
        size_x, size_y = 10, 10

        pos_x = np.random.randint(low=border, high=(28-size_x-border))
        pos_y = np.random.randint(low=border, high=(28-size_y-border))
        
        mask = np.zeros( (28, 28) )
        mask[pos_y:(pos_y+size_y),pos_x:(pos_x+size_x)] = 1.
        mask = mask.reshape([1, 28*28])
        return mask

    def make_mask():
        mask = np.random.uniform(size=(28, 28) )
        mask = 1.*(mask > (1./3))
        mask = mask.reshape([1, 28*28])
        return mask
        
    
    #----------------------------------------------------------------------
    if args.viz:
        idx = [10, 120, 1020, 2020, 3020, 4020, 5020, 6020, 7020, 8020, 9020]
        batch_size = len(idx)

        logger.info("Loading dataset...")
        testset = MNIST(which_set='test', preproc=preproc)

        X = testset.X[idx, :]
        X, Y = testset.preproc(X, None)

        pylab.figure()
        for r in xrange(batch_size):
            pylab.subplot(batch_size, n_samples+2, (n_samples+2)*r+1)
            pylab.axis('off'); pylab.imshow(X[r].reshape((28,28)), interpolation='nearest')
            pylab.subplot(batch_size, n_samples+2, (n_samples+2)*r+2)
            pylab.axis('off'); pylab.imshow(X_corruped[r].reshape((28,28)), interpolation='nearest')
    
            X_recons = X_corruped.copy()
            for s in xrange(n_samples):
                X_recons = inpainter.inpaint(X_corruped, mask, n_iterations=10*s+1)

                pylab.subplot(batch_size, n_samples+2, (n_samples+2)*r+s+3)
                pylab.axis('off'); pylab.imshow(X_recons[r].reshape((28,28)), interpolation='nearest')
        pylab.show(block=True)
        exit(0)

    #----------------------------------------------------------------------

    with h5py.File(args.output, "w") as h5:
        batch_size = 100

        for ds_name in ('train', 'valid', 'test'):
            logger.info("Loading dataset %s ..." % ds_name)
            dataset = MNIST(which_set=ds_name, preproc=preproc)
            n_datapoints = dataset.n_datapoints
            X, Y = dataset.X, dataset.Y
            X, Y = dataset.preproc(X, Y)

            X_ = h5.create_dataset('%s-X' % ds_name, X.shape, X.dtype)
            Y_ = h5.create_dataset('%s-Y' % ds_name, Y.shape, Y.dtype)
            X_corr_ = h5.create_dataset('%s-X-corr' % ds_name, X.shape, X.dtype)
            X_recons_ = h5.create_dataset('%s-X-recons' % ds_name, X.shape, X.dtype)
            
            for first in xrange(0, n_datapoints, batch_size):
                last = first + batch_size
    
                X_batch = X[first:last]
                Y_batch = Y[first:last]
            
                # New corruption mask
                mask = make_mask()

                X_corruped = inpainter.corrupt(mask, X_batch, np.zeros_like(X_batch))
                X_recons = inpainter.inpaint(X_corruped, mask)

                X_[first:last] = X_batch
                Y_[first:last] = Y_batch
                X_corr_[first:last] = X_corruped
                X_recons_[first:last] = X_recons

    logger.info("Finished.")

    #experiment.print_summary()

#=============================================================================
if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--output', type=str, default=None, 
        help="Output file name")
    parser.add_argument('cont', 
        help="Continue a previous in result_dir")
    parser.add_argument('p_model', default="SBN", 
        help="SBN, DARN or NADE (default: SBN")
    parser.add_argument('q_model', default="SBN",
        help="SBN, DARN or NADE (default: SBN")
    parser.add_argument('layer_sizes', default="200,200,10", 
        help="Comma seperated list of sizes. Layer cosest to the data comes first")
    args = parser.parse_args()

    FORMAT = '[%(asctime)s] %(module)-15s %(message)s'
    DATEFMT = "%H:%M:%S"
    logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

    run_inpainting(args)

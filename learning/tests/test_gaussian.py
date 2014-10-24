import unittest 

import numpy as np

import theano 
import theano.tensor as T

import testing
from test_rws import RWSLayerTest, RWSTopLayerTest

# Unit Under Test
from learning.models.gaussian import DiagonalGaussian, DiagonalGaussianTop

#-----------------------------------------------------------------------------

class TestDiagonalGaussianTop(RWSTopLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DiagonalGaussianTop(
                        n_X=8
                    )
        self.layer.setup()

class TestDiagonalGaussian(RWSLayerTest, unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.layer = DiagonalGaussian(
                        n_X=16,
                        n_Y=8,
                        n_hid=20,
                    )
        self.layer.setup()

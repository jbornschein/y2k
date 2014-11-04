import unittest 

import numpy as np

import theano 
import theano.tensor as T

from learning.tests.toys import *

# Unit Under Test
from learning.monitor import * 
from learning.monitor.bootstrap import * 
from learning.monitor.saverepr import SaveRepr

def test_MonitorLL():
    dataset = get_toy_data()
    model = get_toy_model()
    n_samples = (1, 5, 25, 100, 500)
    monitor = MonitorLL(dataset, n_samples)
    monitor.compile(model)

def test_BootstrapLL():
    dataset = get_toy_data()
    model = get_toy_model()
    n_samples = (1, 5, 25, 100, 500)
    monitor = BootstrapLL(dataset, n_samples)
    monitor.compile(model)

def test_SaveRepr():
    n_samples = 10
    dataset = get_toy_data()
    model = get_toy_model()

    monitor = SaveRepr(dataset, name="testset-repr", n_samples=n_samples)
    monitor.compile(model)


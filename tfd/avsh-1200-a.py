#

import numpy as np

from learning.datasets.tfd import TorontoFaceDataset
from learning.preproc import PermuteColumns, QuantNoise
from learning.training import Trainer
from learning.termination import LogLikelihoodIncrease, EarlyStopping
from learning.monitor import MonitorLL, DLogModelParams, SampleFromP

from learning.models.rws  import LayerStack
from learning.models.gaussian  import DiagonalGaussian, DiagonalGaussian, DiagonalGaussianTop, FixedDiagonalGaussianTop

n_vis = 48*48

preproc = [QuantNoise()]
#preproc = []

trainset = TorontoFaceDataset(which_set='unlabeled+train', preproc=preproc)
valiset = TorontoFaceDataset(which_set='valid', preproc=preproc)
smallset = TorontoFaceDataset(which_set='valid', preproc=preproc, n_datapoints=100)
testset = TorontoFaceDataset(which_set='test', preproc=preproc)

p_layers=[
    DiagonalGaussian(
        n_X=n_vis,
        n_Y=100,
        n_hid=[200],
    ),
    FixedDiagonalGaussianTop( 
        n_X=100,
    ),
]

q_layers=[
    DiagonalGaussian(
        n_Y=n_vis,
        n_X=100,
        n_hid=[200],
    )
]

model = LayerStack(
    p_layers=p_layers,
    q_layers=q_layers,
)

trainer = Trainer(
    n_samples=10,
    learning_rate_p=1e-6,
    #learning_rate_q=1e-6,
    learning_rate_q=0.0,
    #learning_rate_s=1e-6,
    learning_rate_s=1e-6,
    weight_decay=0.0,
    batch_size=25,
    dataset=trainset, 
    model=model,
    termination=EarlyStopping(lookahead=10),
    epoch_monitors=[
        DLogModelParams(), 
        #SampleFromP(n_samples=100),
        MonitorLL(name="valiset", data=valiset, n_samples=[1, 5, 10, 25, 100]),
    ],
    final_monitors=[
        MonitorLL(name="final-valiset", data=valiset, n_samples=[1, 5, 10, 25, 100, 500, 1000]),
        MonitorLL(name="final-testset", data=testset, n_samples=[1, 5, 10, 25, 100, 500, 1000]),
    ],
    #step_monitors=[
    #    DLogModelParams(), 
    #    MonitorLL(name="small", data=smallset, n_samples=[10])
    #],
    #monitor_nth_step=100,
)

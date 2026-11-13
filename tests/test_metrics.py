import numpy as np
import pandas as pd
import torch
from inside_transformer.core import logit_difference
from inside_transformer.analyze import cluster_ci


def test_logit_difference_uses_last_position_and_per_row_answers():
    logits=torch.tensor([[[99.,0.,0.],[1.,5.,2.]],[[99.,0.,0.],[9.,3.,4.]]])
    actual=logit_difference(logits,torch.tensor([1,2]),torch.tensor([2,0]))
    torch.testing.assert_close(actual,torch.tensor([3.,-5.]))


def test_metric_sign_stays_clean_oriented():
    clean=torch.tensor([[[0.,4.,1.]]]);corrupt=torch.tensor([[[0.,1.,4.]]])
    a=torch.tensor([1]);b=torch.tensor([2])
    assert logit_difference(clean,a,b).item()==3
    assert logit_difference(corrupt,a,b).item()==-3


def test_cluster_bootstrap_preserves_repeated_observations():
    data=pd.DataFrame({"pair":["a"]*4+["b"]*4,"gain":[2.]*4+[4.]*4})
    s=cluster_ci(data)
    assert s==cluster_ci(data)
    assert s["mean"]==3 and s["clusters"]==2 and s["n"]==8
    assert s["low"]==2 and s["high"]==4

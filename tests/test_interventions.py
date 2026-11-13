import torch
from inside_transformer.core import patched_logits, residual_filter, logit_difference
from inside_transformer.data import build_dataset
from inside_transformer.experiment import validate_official_helper


def example(model):
    r=build_dataset(model)[0][0]
    c=torch.tensor([r["clean_tokens"]]);k=torch.tensor([r["corrupted_tokens"]])
    return r,c,k


def test_noop_and_deterministic_inference(model):
    _,c,k=example(model)
    original,cache=model.run_with_cache(k,names_filter=residual_filter)
    assert torch.equal(original,model(k))
    patched=patched_logits(model,k,cache,6,-1)
    torch.testing.assert_close(patched,original,rtol=0,atol=1e-5)


def test_prefix_causal_null(model):
    r,c,k=example(model)
    _,cache=model.run_with_cache(c,names_filter=residual_filter)
    original=model(k)
    for pos in ["bos","name1","name2"]:
        patched=patched_logits(model,k,cache,9,r["positions"][pos])
        torch.testing.assert_close(patched,original,rtol=0,atol=1e-5)


def test_embedding_repair_positive_control(model):
    r,c,k=example(model)
    clean,cache=model.run_with_cache(c,names_filter=residual_filter)
    patched=patched_logits(model,k,cache,0,r["positions"]["subject"])
    # Earlier positions are identical; later positions' embeddings also identical.
    torch.testing.assert_close(patched,clean,rtol=0,atol=1e-5)


def test_custom_patch_matches_official_api(model):
    r,_,_=example(model)
    assert validate_official_helper(model,r)["official_custom_absolute_error"]<1e-4


def test_hooks_removed_after_intervention(model):
    r,c,k=example(model)
    original=model(k)
    _,cache=model.run_with_cache(c,names_filter=residual_filter)
    patched_logits(model,k,cache,8,-1)
    assert torch.equal(model(k),original)

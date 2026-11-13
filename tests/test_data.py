from inside_transformer.data import build_dataset, NAMES


def test_token_alignment_and_single_token_answers(model):
    rows,excluded=build_dataset(model)
    assert len(rows)==128 and not excluded
    for r in rows:
        assert len(r["clean_tokens"])==len(r["corrupted_tokens"])
        assert [i for i,(c,k) in enumerate(zip(r["clean_tokens"],r["corrupted_tokens"])) if c!=k]==[r["positions"]["subject"]]
        assert model.tokenizer.decode([r["answer"]])==" "+r["a"]
        assert model.tokenizer.decode([r["foil"]])==" "+r["b"]


def test_split_disjointness_and_counterbalancing(model):
    rows,_=build_dataset(model)
    assert set(NAMES["development"]).isdisjoint(NAMES["heldout"])
    for r in rows:
        opposite=next(s for s in rows if s["pool"]==r["pool"] and s["pair"]==r["pair"] and s["template"]==r["template"] and s["order"]!=r["order"])
        assert r["a"]==opposite["b"] and r["b"]==opposite["a"]
    assert len({r["pair"] for r in rows})==16


def test_dataset_reproducible(model):
    assert build_dataset(model)==build_dataset(model)

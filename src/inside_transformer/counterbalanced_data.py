"""Prospective factorial IOI data, independent of model prediction outcomes."""
from collections import Counter

TEMPLATES = [
    "When {first} and {second} met at the station, {giver} gave a ticket to",
    "Before leaving the library, {first} and {second} talked for a while. Then {giver} lent a book to",
]
NAMES = {
    "evaluation": ["Oliver", "Charlotte", "Henry", "Grace", "Edward", "Victoria", "Charles", "Margaret",
                   "Joseph", "Catherine", "Andrew", "Christine", "Patrick", "Diana", "Simon", "Rose"],
    "donor": ["Anthony", "Angela", "Steven", "Melissa", "Brian", "Nancy", "Kevin", "Deborah",
              "Frank", "Carol", "Stephen", "Linda", "Eric", "Sandra", "Adam", "Karen"],
}


def candidates():
    rows = []
    for pool, names in NAMES.items():
        for block in range(8):
            identities = names[2*block:2*block+2]
            for template, text in enumerate(TEMPLATES):
                for order in range(2):
                    first, second = identities[::1 if order == 0 else -1]
                    for giver_identity in range(2):
                        giver, answer = identities[giver_identity], identities[1-giver_identity]
                        rows.append(dict(id=f"{pool}-b{block}-t{template}-o{order}-g{giver_identity}",
                            pool=pool, block=block, pair=f"block-{block}", template=template, order=order,
                            giver_identity=giver_identity, answer_identity=1-giver_identity,
                            first=first, second=second, giver=giver, answer_name=answer,
                            foil_name=giver, answer_position=1 if answer==first else 2,
                            clean=text.format(first=first,second=second,giver=giver),
                            corrupted=text.format(first=first,second=second,giver=answer)))
    return rows


def tokenize_row(model, row):
    row = dict(row)
    first, second = [model.tokenizer.encode(" "+row[k], add_special_tokens=False) for k in ["first", "second"]]
    assert len(first)==len(second)==1, "name is not a single leading-space token"
    answer, foil = [model.tokenizer.encode(" "+row[k],add_special_tokens=False)[0] for k in ["answer_name","foil_name"]]
    c,k=[model.to_tokens(row[text],prepend_bos=True)[0].cpu().tolist() for text in ["clean","corrupted"]]
    assert len(c)==len(k), "clean/corrupted length mismatch"
    changed=[i for i,(x,y) in enumerate(zip(c,k)) if x!=y]
    assert len(changed)==1, "not one changed token"
    assert c[changed[0]]==foil and k[changed[0]]==answer, "giver replacement mismatch"
    assert c.count(answer)==1 and c.count(foil)==2, "name occurrences mismatch"
    for text, tok, target in [("clean",c,answer),("corrupted",k,foil)]:
        extended=model.to_tokens(row[text]+" "+model.tokenizer.decode([target]).lstrip(),prepend_bos=True)[0].cpu().tolist()
        assert extended==tok+[target], "answer continuation boundary mismatch"
    positions={"bos":0,"name1":c.index(first[0]),"name2":c.index(second[0]),"subject":changed[0],"final":len(c)-1}
    assert positions["name1"]<positions["name2"]<positions["subject"]<positions["final"]
    row.update(answer=answer,foil=foil,clean_tokens=c,corrupted_tokens=k,positions=positions)
    return row


def build_dataset(model, old_rows):
    old_names={r[k] for r in old_rows for k in ["a","b"]}
    new_names=[n for names in NAMES.values() for n in names]
    assert len(new_names)==len(set(new_names)) and old_names.isdisjoint(new_names)
    old_text={r[k] for r in old_rows for k in ["clean","corrupted"]}
    accepted, ledger = [], []
    for row in candidates():
        assert row["clean"] not in old_text and row["corrupted"] not in old_text
        try:
            accepted.append(tokenize_row(model,row))
            ledger.append(dict(row,tokenizer_passed=True,exclusion_reason=None))
        except AssertionError as exc:
            ledger.append(dict(row,tokenizer_passed=False,exclusion_reason=str(exc)))
    failed_blocks={r["block"] for r in ledger if not r["tokenizer_passed"]}
    rows=[r for r in accepted if r["block"] not in failed_blocks]
    for row in ledger:
        row["included"]=row["block"] not in failed_blocks
        if row["tokenizer_passed"] and not row["included"]:
            row["exclusion_reason"]="another candidate in this evaluation+donor block failed tokenization"
    validate_balance(rows)
    return rows, ledger


def validate_balance(rows):
    for pool in NAMES:
        subset=[r for r in rows if r["pool"]==pool]
        blocks={r["block"] for r in subset}
        for block in blocks:
            part=[r for r in subset if r["block"]==block]
            assert len(part)==8
            assert Counter(r["answer_position"] for r in part)=={1:4,2:4}
            assert Counter((r["template"],r["order"],r["giver_identity"]) for r in part)=={
                (t,o,g):1 for t in range(2) for o in range(2) for g in range(2)}
            for identity in range(2):
                assert Counter(r["answer_position"] for r in part if r["answer_identity"]==identity)=={1:2,2:2}


def donor_for(row, donors, same_position=True):
    target=row["answer_position"] if same_position else 3-row["answer_position"]
    matches=[d for d in donors if (d["block"],d["template"],d["order"],d["answer_position"])==
             (row["block"],row["template"],row["order"],target)]
    assert len(matches)==1
    donor=matches[0]
    assert {row["first"],row["second"]}.isdisjoint({donor["first"],donor["second"]})
    assert row["positions"]==donor["positions"] and len(row["clean_tokens"])==len(donor["clean_tokens"])
    return donor

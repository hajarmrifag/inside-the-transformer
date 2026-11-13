"""Deterministic, counterbalanced IOI examples; no model-based exclusions."""
import json
from pathlib import Path

TEMPLATES = [
    "When {a} and {b} went to the store, {s} gave a book to",
    "After {a} and {b} arrived at the park, {s} handed a drink to",
    "While {a} and {b} were at the office, {s} passed a note to",
    "Yesterday {a} and {b} visited the garden, and {s} sent a letter to",
]
NAMES = {
    "development": ["John", "Mary", "Alice", "Bob", "David", "Sarah", "James", "Emma", "Michael", "Laura", "Robert", "Anna", "Daniel", "Lisa", "Peter", "Susan"],
    "heldout": ["Tom", "Jane", "Paul", "Lucy", "Mark", "Julia", "George", "Helen", "William", "Rachel", "Richard", "Emily", "Thomas", "Rebecca", "Jack", "Elizabeth"],
}
POSITIONS = ["bos", "name1", "name2", "subject", "final"]


def build_dataset(model):
    accepted, excluded = [], []
    for pool, names in NAMES.items():
        for pair in range(8):
            for order in range(2):
                a, b = names[2*pair:2*pair+2][::1 if order == 0 else -1]
                for template_id, template in enumerate(TEMPLATES):
                    row = dict(id=f"{pool}-{pair}-{order}-{template_id}", pool=pool,
                               pair=f"{pool}-{pair}", order=order, template=template_id,
                               split=("dev" if template_id < 2 else "new_template") +
                                     ("_names" if pool == "development" else "_new_names"),
                               a=a, b=b, clean=template.format(a=a,b=b,s=b),
                               corrupted=template.format(a=a,b=b,s=a))
                    try:
                        ai = model.tokenizer.encode(" "+a, add_special_tokens=False)
                        bi = model.tokenizer.encode(" "+b, add_special_tokens=False)
                        assert len(ai) == len(bi) == 1, "answer is not single-token"
                        c = model.to_tokens(row["clean"], prepend_bos=True)[0].cpu().tolist()
                        k = model.to_tokens(row["corrupted"], prepend_bos=True)[0].cpu().tolist()
                        assert len(c) == len(k), "sequence length mismatch"
                        assert model.to_tokens(row["clean"]+" "+a, prepend_bos=True)[0].cpu().tolist()==c+ai, "clean answer boundary mismatch"
                        assert model.to_tokens(row["corrupted"]+" "+b, prepend_bos=True)[0].cpu().tolist()==k+bi, "counterfactual answer boundary mismatch"
                        changed = [i for i,(x,y) in enumerate(zip(c,k)) if x != y]
                        assert len(changed)==1, "not exactly one changed token"
                        apos = [i for i,x in enumerate(c) if x == ai[0]]
                        bpos = [i for i,x in enumerate(c) if x == bi[0]]
                        assert len(apos)==1 and len(bpos)==2, "name occurrence mismatch"
                        assert changed[0]==bpos[1], "subject position mismatch"
                        assert c[changed[0]]==bi[0] and k[changed[0]]==ai[0]
                        row.update(answer=ai[0], foil=bi[0], clean_tokens=c, corrupted_tokens=k,
                                   positions=dict(zip(POSITIONS,[0,apos[0],bpos[0],changed[0],len(c)-1])))
                        accepted.append(row)
                    except AssertionError as exc:
                        row["exclusion_reason"] = str(exc)
                        excluded.append(row)
    return accepted, excluded


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")

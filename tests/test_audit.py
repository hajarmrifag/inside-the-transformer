import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("numeric_audit",ROOT/"scripts/audit_results.py")
numeric_audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric_audit)


def test_independent_numeric_audit():
    summary,ledger=numeric_audit.audit(ROOT/"results",ROOT/"results/counterbalanced-v1")
    assert summary["passed"] and summary["fields_recomputed"]>2500
    assert any(x["claim"]=="extension.matched_minus_balanced.low" for x in ledger)


def test_audit_detects_tampered_paired_contrast_interval(tmp_path):
    source=ROOT/"results/counterbalanced-v1"
    target=tmp_path/"changed"
    shutil.copytree(source,target)
    path=target/"summary.json"
    summary=json.loads(path.read_text())
    summary["contrasts"]["matched_minus_balanced"]["low"]+=1.0
    path.write_text(json.dumps(summary))
    with pytest.raises(AssertionError,match="extension.matched_minus_balanced.low"):
        numeric_audit.audit(ROOT/"results",target)

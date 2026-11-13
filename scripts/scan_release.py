"""Heuristic privacy/size scan of exactly the locally staged Git file set."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PATTERNS={
    "personal_absolute_path":re.compile(r"/(?:Users|home|private/(?:tmp|var))/[^\s\"<>]+"),
    "private_key":re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_format":re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
}
GENERATED={".venv",".pytest_cache",".matplotlib","__pycache__",".git",".cache"}
WEIGHTS={".pt",".pth",".bin",".safetensors",".onnx"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=ROOT/"audit/release-scan.json")
    args=parser.parse_args()
    names=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0")
    names=[n for n in names if n];hits=[];total=0;largest=0
    for name in names:
        path=ROOT/name
        if path.is_symlink():hits.append(dict(path=name,kind="symlink_requires_review"));continue
        size=path.stat().st_size;total+=size;largest=max(largest,size)
        if size>10*1024*1024:hits.append(dict(path=name,kind="over_10_MiB"))
        if any(p in GENERATED or p.endswith(".egg-info") for p in Path(name).parts):hits.append(dict(path=name,kind="generated_artifact"))
        if path.suffix.lower() in WEIGHTS:hits.append(dict(path=name,kind="model_or_binary_weight_file"))
        try:lines=path.read_text().splitlines()
        except UnicodeError:continue
        for number,line in enumerate(lines,1):
            for kind,pattern in PATTERNS.items():
                if pattern.search(line):hits.append(dict(path=name,line=number,kind=kind))
    result=dict(scope="staged files only; heuristic patterns, not a guarantee that every possible secret is absent",
        files_scanned=len(names),total_bytes=total,largest_file_bytes=largest,findings=hits,passed=not hits)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if hits:raise SystemExit(1)


if __name__=="__main__":main()

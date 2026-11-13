"""Download the pinned public model/tokenizer once; no login or training."""
from inside_transformer.experiment import MODEL_REVISION, load_model

if __name__ == "__main__":
    model=load_model(local_files_only=False)
    print(f"Cached gpt2 model and tokenizer revision {MODEL_REVISION}.")

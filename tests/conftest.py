import pytest
from inside_transformer.experiment import seed_everything, load_model


@pytest.fixture(scope="session")
def model():
    seed_everything()
    return load_model(local_files_only=True)

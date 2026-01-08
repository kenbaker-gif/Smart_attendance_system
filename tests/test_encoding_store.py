from pathlib import Path
import pickle
import numpy as np
from app.utils.encoding_store import load_encodings, save_encodings_npz


def test_npz_save_and_load(tmp_path):
    encs = [np.ones(128)]
    ids = ["2400102415"]
    out = tmp_path / "enc.npz"

    save_encodings_npz(out, encs, ids)

    got = load_encodings(out)
    assert got is not None
    assert "encodings" in got and "ids" in got
    assert got["encodings"].shape[0] == 1


def test_migrate_from_pickle(tmp_path):
    # Create a legacy pickle file
    pkl = tmp_path / "legacy_encodings.pkl"
    data = {"encodings": [np.ones(128)], "ids": ["2400102415"]}
    with open(pkl, "wb") as fh:
        pickle.dump(data, fh)

    got = load_encodings(pkl)
    assert got is not None
    assert (tmp_path / "legacy_encodings.npz").exists()
    # The original should be renamed to .pkl.migrated
    assert (tmp_path / "legacy_encodings.pkl.migrated").exists()
    assert got["encodings"].shape[0] == 1

import importlib
import sys
import types
import pickle
from pathlib import Path

import numpy as np


def _make_fake_face_recognition():
    m = types.ModuleType("face_recognition")

    def load_image_file(path):
        # Return a dummy object
        return b"fake-image"

    def face_locations(img, model="hog"):
        return [(0, 0, 10, 10)]

    def face_encodings(img, locs):
        return [np.ones(128, dtype=float)]

    m.load_image_file = load_image_file
    m.face_locations = face_locations
    m.face_encodings = face_encodings
    return m


def test_generate_encodings_creates_pickle(tmp_path, monkeypatch):
    # Arrange: prepare fake face_recognition module before importing
    sys.modules["face_recognition"] = _make_fake_face_recognition()

    # Import the module under test by path (avoid relying on package importability)
    import importlib.util
    rec_path = Path(__file__).resolve().parents[1] / "streamlit" / "recognition.py"
    spec = importlib.util.spec_from_file_location("streamlit.recognition", rec_path)
    rec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rec
    spec.loader.exec_module(rec)

    # Create sample data: one student folder with one image file
    student_dir = tmp_path / "2400102415"
    student_dir.mkdir(parents=True)
    img_file = student_dir / "1.jpg"
    img_file.write_bytes(b"fake-jpg-data")

    out_file = tmp_path / "encodings_test.pkl"

    # Act
    ok = rec.generate_encodings(images_dir=tmp_path, output_path=out_file)

    # Assert
    assert ok is True
    assert out_file.exists()

    with open(out_file, "rb") as fh:
        data = pickle.load(fh)

    assert "encodings" in data and "ids" in data
    assert len(data["encodings"]) == 1
    assert data["ids"][0] == "2400102415"


def test_generate_encodings_no_images_returns_false(tmp_path, monkeypatch):
    sys.modules["face_recognition"] = _make_fake_face_recognition()
    import importlib.util
    rec_path = Path(__file__).resolve().parents[1] / "streamlit" / "recognition.py"
    spec = importlib.util.spec_from_file_location("streamlit.recognition", rec_path)
    rec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rec
    spec.loader.exec_module(rec)

    ok = rec.generate_encodings(images_dir=tmp_path, output_path=tmp_path / "out.pkl")
    assert ok is False

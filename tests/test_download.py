from hashlib import sha256
from io import BytesIO
import pytest
from churn import data
from churn.config import ROOT


def test_download_retries_partial_transfer_and_validates_before_replacing(
    tmp_path, monkeypatch
):
    content = (ROOT / "data/sample.csv").read_bytes()
    calls = []

    def response(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise data.IncompleteRead(b"partial", 100)
        return BytesIO(content)

    monkeypatch.setattr(data, "urlopen", response)
    monkeypatch.setattr(data, "SHA256", sha256(content).hexdigest())
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    output = tmp_path / "dataset.csv"
    assert data.download(output).read_bytes() == content
    assert len(calls) == 2


def test_checksum_mismatch_preserves_existing_file(tmp_path, monkeypatch):
    output = tmp_path / "dataset.csv"
    output.write_text("existing")
    monkeypatch.setattr(
        data, "urlopen", lambda *args, **kwargs: BytesIO(b"changed source")
    )
    with pytest.raises(ValueError, match="checksum"):
        data.download(output)
    assert output.read_text() == "existing"

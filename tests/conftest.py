import pytest


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch, tmp_path):
    """Tests never read the developer's real .env."""
    from g1 import vlm
    monkeypatch.setattr(vlm, "DOTENV", tmp_path / "no.env")

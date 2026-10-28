import pytest


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch, tmp_path):
    """Tests never read the developer's real .env."""
    from g1 import vlm
    monkeypatch.setattr(vlm, "DOTENV", tmp_path / "no.env")


@pytest.fixture(autouse=True)
def no_code_snapshot(monkeypatch):
    """Tests never write refs/g1-runs/* into the developer's repo (test_task tests the real one)."""
    from g1.agent import task
    real = task.code_snapshot
    monkeypatch.setattr(task, "code_snapshot", lambda root=None: {"head": "test", "snapshot": "test", "ref": None})
    return real

from scalp_system.storage.repository import SQLiteRepository
from scalp_system.ui.dashboard import create_dashboard_app


def test_dashboard_routes(tmp_path):
    repo_path = tmp_path / "signals.sqlite3"
    repository = SQLiteRepository(repo_path)
    repository.persist_signal("BBG000000001", 1, 0.87)
    app = create_dashboard_app(repository)
    client = app.test_client()

    response = client.get("/api/summary")
    assert response.status_code == 200
    summary = response.json()
    assert summary["total_signals"] == 1
    assert summary["latest"]["figi"] == "BBG000000001"

    response = client.get("/api/signals")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["figi"] == "BBG000000001"

    response = client.get("/")
    assert response.status_code == 200
    assert "Scalp System Dashboard" in response.get_data(as_text=True)

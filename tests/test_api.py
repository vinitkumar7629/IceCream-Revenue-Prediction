import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Ice Cream Revenue Predictor" in response.text


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "LinearRegression"
    assert data["slope"] == pytest.approx(39.02, abs=0.01)
    assert data["training_range_celsius"] == [14.0, 46.0]


def test_predict_inside_training_range():
    response = client.post("/predict", json={"temperature": 35})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_revenue"] == pytest.approx(1001.54, abs=0.01)
    assert data["in_training_range"] is True
    assert data["note"] is None


def test_hotter_means_more_revenue():
    cool = client.post("/predict", json={"temperature": 20}).json()
    hot = client.post("/predict", json={"temperature": 40}).json()
    assert hot["predicted_revenue"] > cool["predicted_revenue"]


def test_prediction_outside_training_range_is_flagged():
    response = client.post("/predict", json={"temperature": 55})
    assert response.status_code == 200
    data = response.json()
    assert data["in_training_range"] is False
    assert "extrapolation" in data["note"]


def test_negative_revenue_is_clipped_to_zero():
    response = client.post("/predict", json={"temperature": 0})
    assert response.status_code == 200
    assert response.json()["predicted_revenue"] == 0.0


@pytest.mark.parametrize("bad_temperature", [100, -50, "hot", None])
def test_invalid_temperature_is_rejected(bad_temperature):
    response = client.post("/predict", json={"temperature": bad_temperature})
    assert response.status_code == 422


def test_missing_temperature_is_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422
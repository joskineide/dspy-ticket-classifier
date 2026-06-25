import pytest
from unittest.mock import AsyncMock, patch

from app.schemas.ticket import ClassifyResponse


@pytest.mark.parametrize("labels", [
    ["bug"],
    ["feature"],
    ["feedback"],
    ["out_of_scope"],
    ["bug", "feedback"],               # multi-label: a crashing feature that frustrates the user
    ["feature", "feedback"],           # multi-label: enthusiastic wrong-premise request
])
async def test_classify_valid_labels(client, labels):
    mock_result = ClassifyResponse(labels=labels, reasoning=None, model="test/model")
    with patch("app.router.classify.classify_ticket", new=AsyncMock(return_value=mock_result)):
        response = await client.post("/v1/classify", json={"ticket": "login button broken"})
    assert response.status_code == 200
    assert response.json()["labels"] == labels


async def test_classify_rejects_empty_ticket(client):
    response = await client.post("/v1/classify", json={"ticket": ""})
    assert response.status_code == 422


async def test_classify_rejects_invalid_label(client):
    # If the model returns any out-of-vocabulary label, the router should 422.
    mock_result = ClassifyResponse(labels=["unknown"], reasoning=None, model="test/model")
    with patch("app.router.classify.classify_ticket", new=AsyncMock(return_value=mock_result)):
        response = await client.post("/v1/classify", json={"ticket": "some ticket"})
    assert response.status_code == 422


async def test_classify_rejects_partially_invalid_labels(client):
    # Even one bad label in an otherwise valid list should 422.
    mock_result = ClassifyResponse(labels=["bug", "complaint"], reasoning=None, model="test/model")
    with patch("app.router.classify.classify_ticket", new=AsyncMock(return_value=mock_result)):
        response = await client.post("/v1/classify", json={"ticket": "some ticket"})
    assert response.status_code == 422

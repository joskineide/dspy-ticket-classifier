import pytest
from unittest.mock import AsyncMock, patch

from app.schemas.ticket import ClassifyResponse


# Once classify_ticket() is implemented, replace the mock with a real call
# (or keep the mock so tests don't require a live LM).

async def test_classify_returns_501_before_implementation(client):
    # The endpoint returns 501 Not Implemented while the DSPy service is still a stub.
    response = await client.post("/v1/classify", json={"ticket": "login button broken"})
    assert response.status_code == 501


async def test_classify_rejects_empty_ticket(client):
    response = await client.post("/v1/classify", json={"ticket": ""})
    assert response.status_code == 422


@pytest.mark.parametrize("label", ["bug", "billing", "feature", "security"])
async def test_classify_valid_label(client, label):
    # TODO: fill this test in once classify_ticket() is implemented.
    # Pattern: patch "app.router.classify.classify_ticket" with an AsyncMock
    # that returns a ClassifyResponse, then assert the endpoint returns 200
    # and the label passes through.
    pass


async def test_classify_rejects_invalid_label(client):
    # If the model returns an out-of-vocabulary label, the router should 422.
    mock_result = ClassifyResponse(label="unknown", reasoning=None, model="test/model")
    with patch("app.router.classify.classify_ticket", new=AsyncMock(return_value=mock_result)):
        response = await client.post("/v1/classify", json={"ticket": "some ticket"})
    assert response.status_code == 422

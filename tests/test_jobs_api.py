"""
Integration tests for the jobs API endpoints.
Uses the test client with SQLite + stub provider (no real infrastructure needed).
"""
import pytest
from httpx import AsyncClient


STUB_IMAGE_JOB = {
    "type": "image_generation",
    "priority": "normal",
    "provider": "stub",
    "model": "stub-model",
    "input": {
        "prompt": "A cyberpunk motorcycle in the rain",
        "width": 512,
        "height": 512,
        "steps": 10,
    },
}


@pytest.mark.asyncio
async def test_create_job_returns_202(client: AsyncClient):
    response = await client.post("/api/v1/jobs", json=STUB_IMAGE_JOB)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["provider"] == "stub"
    assert body["type"] == "image_generation"
    assert "id" in body


@pytest.mark.asyncio
async def test_get_job_returns_job(client: AsyncClient):
    create = await client.post("/api/v1/jobs", json=STUB_IMAGE_JOB)
    job_id = create.json()["id"]

    response = await client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["id"] == job_id


@pytest.mark.asyncio
async def test_list_jobs(client: AsyncClient):
    await client.post("/api/v1/jobs", json=STUB_IMAGE_JOB)
    await client.post("/api/v1/jobs", json={**STUB_IMAGE_JOB, "priority": "high"})

    response = await client.get("/api/v1/jobs")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


@pytest.mark.asyncio
async def test_cancel_queued_job(client: AsyncClient):
    create = await client.post("/api/v1/jobs", json=STUB_IMAGE_JOB)
    job_id = create.json()["id"]

    cancel = await client.delete(f"/api/v1/jobs/{job_id}")
    assert cancel.status_code == 200
    assert cancel.json()["cancelled"] is True

    get = await client.get(f"/api/v1/jobs/{job_id}")
    assert get.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_create_job_invalid_provider(client: AsyncClient):
    bad_payload = {**STUB_IMAGE_JOB, "provider": "nonexistent_provider"}
    response = await client.post("/api/v1/jobs", json=bad_payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_idempotency_key_deduplication(client: AsyncClient):
    payload = {**STUB_IMAGE_JOB, "idempotency_key": "unique-key-001"}
    r1 = await client.post("/api/v1/jobs", json=payload)
    r2 = await client.post("/api/v1/jobs", json=payload)

    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["id"] == r2.json()["id"]  # same job returned


@pytest.mark.asyncio
async def test_get_nonexistent_job_returns_404(client: AsyncClient):
    response = await client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

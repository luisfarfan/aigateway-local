import asyncio
import sys
from uuid import UUID
from src.modules.jobs.repository import JobRepository
from src.core.database import AsyncSessionLocal

async def check_job(job_id_str):
    try:
        job_id = UUID(job_id_str)
        async with AsyncSessionLocal() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job:
                print(f"Job ID: {job.id}")
                print(f"Status: {job.status}")
                print(f"Created At: {job.created_at}")
                print(f"Modality: {job.modality}")
                print(f"Payload: {job.payload}")
            else:
                print(f"Job {job_id_str} not found in DB.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(check_job(sys.argv[1]))
    else:
        print("Usage: python3 check_job.py <job_id>")

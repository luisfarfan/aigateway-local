
import asyncio
from src.core.database import AsyncSessionLocal
from src.modules.jobs.repository import JobRepository
from uuid import UUID

async def main():
    async with AsyncSessionLocal() as sess:
        repo = JobRepository(sess)
        job_id = UUID('89812da3-f1bf-4da7-bb32-ffc4d4913719')
        arts = await repo.get_artifacts(job_id)
        print(f"Artifacts for {job_id}:")
        for a in arts:
            print(f"- Type: {a.artifact_type}, URL: {a.public_url}")

if __name__ == "__main__":
    asyncio.run(main())

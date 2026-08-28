
import asyncio
from src.core.database import AsyncSessionLocal
from src.modules.jobs.models import Artifact
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as sess:
        res = await sess.execute(select(Artifact))
        arts = res.scalars().all()
        print(f"Total artifacts: {len(arts)}")
        for a in arts:
            print(f"- Job: {a.job_id}, Type: {a.artifact_type}, File: {a.filename}")

if __name__ == "__main__":
    asyncio.run(main())

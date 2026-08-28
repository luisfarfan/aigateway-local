import asyncio
from src.core.database import AsyncSessionLocal
from src.modules.jobs.models import Job
from uuid import UUID

async def check():
    async with AsyncSessionLocal() as s:
        # Get the MOST RECENT job if no ID provided
        from sqlmodel import select, col
        stmt = select(Job).order_by(col(Job.created_at).desc()).limit(5)
        res = await s.execute(stmt)
        jobs = res.scalars().all()
        for j in jobs:
            print(f"ID: {j.id} | Status: {j.status} | Worker: {j.worker_id} | Error: {j.error_message}")

if __name__ == "__main__":
    asyncio.run(check())

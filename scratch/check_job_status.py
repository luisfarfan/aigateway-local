import asyncio
from sqlalchemy import text
from src.core.database import engine

async def check_job():
    async with engine.connect() as conn:
        res = await conn.execute(text("SELECT status, progress_percent, error_message, worker_id FROM jobs WHERE id = 'fb32e099-7e59-4037-a536-6079335fd43d'"))
        row = res.fetchone()
        if row:
            print(f"Status: {row[0]}")
            print(f"Progress: {row[1]}%")
            print(f"Error: {row[2]}")
            print(f"Worker: {row[3]}")
        else:
            print("Job not found")

if __name__ == "__main__":
    asyncio.run(check_job())

import asyncio
import redis
from sqlalchemy import text
from src.core.database import AsyncSessionLocal
from src.core.config import get_settings

async def clean():
    settings = get_settings()
    
    # 1. Flush Redis
    try:
        r = redis.from_url(settings.redis_url)
        r.flushall()
        print("✅ Redis queues cleared.")
    except Exception as e:
        print(f"❌ Failed to clear Redis: {e}")
    
    # 2. Truncate DB Tables
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("TRUNCATE jobs, artifacts RESTART IDENTITY CASCADE;"))
            await session.commit()
        print("✅ Database records cleared.")
    except Exception as e:
        print(f"❌ Failed to clear Database: {e}")

if __name__ == "__main__":
    asyncio.run(clean())

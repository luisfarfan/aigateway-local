import asyncio
from sqlalchemy import text
from src.core.database import engine

async def update_enums():
    # Get a raw connection to set isolation level
    conn = await engine.connect()
    # For asyncpg, we might need a different approach for autocommit
    # But let's try to just execute in a loop
    for val in ['autonomous_mission', 'video_assembly']:
        try:
            # We execute it in a separate transaction-less way if possible
            # sqlalchemy 2.0+ uses transactions by default, so we commit after each
            await conn.execute(text(f"ALTER TYPE jobtype ADD VALUE '{val}'"))
            await conn.commit()
            print(f"Added {val} to jobtype enum")
        except Exception as e:
            await conn.rollback()
            print(f"Skipping {val} (already exists?): {e}")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(update_enums())

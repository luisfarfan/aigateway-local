import asyncio
from sqlalchemy import text
from src.core.database import engine

async def update_enums():
    # Use a raw connection from the engine and execute without a transaction block
    async with engine.connect() as conn:
        print("Updating jobtype enum (autocommit mode)...")
        try:
            # We need to use execution_options(isolation_level="AUTOCOMMIT") for ALTER TYPE ADD VALUE
            # However, for simplicity and compatibility with some async drivers, we'll try to execute directly
            # outside of any begin() block.
            await conn.execute(text("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'AUTONOMOUS_MISSION';"))
            await conn.execute(text("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'VIDEO_ASSEMBLY';"))
            await conn.commit()
            print("Enum update SUCCESS")
        except Exception as e:
            print(f"Enum update failed: {e}")

if __name__ == "__main__":
    asyncio.run(update_enums())

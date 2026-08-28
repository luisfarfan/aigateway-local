import asyncio
from sqlalchemy import text
from src.core.database import engine
from src.core.domain import JobType, JobStatus, JobPriority

async def sync_enums():
    async with engine.connect() as conn:
        # Enums and their expected values from domain.py
        enums = {
            "jobtype": [t.value for t in JobType] + [t.name for t in JobType],
            "jobstatus": [s.value for s in JobStatus] + [s.name for s in JobStatus],
            "jobpriority": [p.value for p in JobPriority] + [p.name for p in JobPriority]
        }
        
        for enum_name, values in enums.items():
            print(f"Syncing {enum_name}...")
            for val in values:
                try:
                    # ADD VALUE cannot be run in a transaction in many PG versions
                    # We try to commit after each one
                    await conn.execute(text(f"ALTER TYPE {enum_name} ADD VALUE '{val}'"))
                    await conn.commit()
                    print(f"  Added {val}")
                except Exception as e:
                    await conn.rollback()
                    # Error is expected if it already exists
                    pass
        print("Enum synchronization completed.")

if __name__ == "__main__":
    asyncio.run(sync_enums())

import asyncio
from arq import run_worker
from src.core.redis import get_arq_redis_settings
from workers.executor import run_job

async def execute_job(ctx, job_id_str):
    print(f"RECIEVED JOB: {job_id_str}")
    return await run_job(ctx, job_id_str)

class Settings:
    functions = [execute_job]
    redis_settings = get_arq_redis_settings()
    queues = ["arq:queue:high", "arq:queue:normal", "arq:queue:low"]

if __name__ == "__main__":
    run_worker(Settings)

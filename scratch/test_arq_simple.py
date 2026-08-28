import asyncio
import structlog
from arq import create_pool, run_worker
from src.core.redis import get_arq_redis_settings
from uuid import uuid4

async def stub_task(ctx, val):
    print(f"STUB TASK EXECUTED WITH {val}")
    return val

class WorkerSettings:
    functions = [stub_task]
    redis_settings = get_arq_redis_settings()
    queues = ["test_queue"]

async def main():
    # 1. Start worker in background
    # (Simplified: we use arq.run_worker which is blocking, so we need a thread or task)
    pass

if __name__ == "__main__":
    # Just test enqueue and check keyspace again with very simple names
    async def test():
        pool = await create_pool(get_arq_redis_settings())
        print(f"Redis Settings: {pool.redis_settings}")
        await pool.enqueue_job("stub_task", "hello", _queue_name="test_queue")
        
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        print(f"Keys: {r.keys('arq:queue:test_queue')}")
        
    asyncio.run(test())

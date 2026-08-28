import asyncio
from arq import create_pool
from src.core.redis import get_arq_redis_settings
from uuid import uuid4

async def test():
    pool = await create_pool(get_arq_redis_settings())
    job_id = uuid4()
    print(f"Enqueuing test job {job_id} to 'high' queue...")
    res = await pool.enqueue_job("execute_job", str(job_id), _queue_name="high")
    print(f"Enqueue result: {res}")
    
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    keys = r.keys("arq:queue:*")
    print(f"Queue keys after enqueue: {keys}")
    for k in keys:
        print(f"Length of {k.decode()}: {r.llen(k)}")

if __name__ == "__main__":
    asyncio.run(test())

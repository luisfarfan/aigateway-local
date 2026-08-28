import asyncio
from arq.connections import create_pool
from src.core.redis import get_arq_redis_settings

async def inspect_arq():
    arq = await create_pool(get_arq_redis_settings())
    queues = ["high", "normal", "low"]
    # arq is the pool which acts as a Redis client
    for q in queues:
        key = f"arq:queue:{q}"
        size = await arq.zcard(key)
        print(f"Queue {q} size: {size}")
        if size > 0:
            jobs = await redis.zrange(key, 0, -1)
            print(f"  Jobs in {q}: {jobs}")
    
    # Check results
    results = await arq.all_job_results()
    print(f"Total results: {len(results)}")
    for r in results:
        print(f"  Job {r.job_id}: {r.success} (Function: {r.function})")

if __name__ == "__main__":
    asyncio.run(inspect_arq())

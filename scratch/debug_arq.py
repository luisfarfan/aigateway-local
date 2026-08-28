import asyncio
from arq.connections import create_pool
from src.core.redis import get_arq_redis_settings

async def check():
    pool = await create_pool(get_arq_redis_settings())
    # Arq doesn't have a simple "list all" but we can check the queues
    queues = ["high", "normal", "low"]
    for q in queues:
        # Arq stores jobs in a list named arq:queue:{name}
        # Wait, arq documentation says it's a Redis list.
        # Let's check the length of those keys directly.
        pass

    # Actually let's use the low-level redis to check all keys
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    for q in queues:
        key = f"arq:queue:{q}"
        print(f"{key}: {r.llen(key)}")
        if r.llen(key) > 0:
            print(f"Contents: {r.lrange(key, 0, -1)}")

if __name__ == "__main__":
    asyncio.run(check())

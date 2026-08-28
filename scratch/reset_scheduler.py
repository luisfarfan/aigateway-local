import asyncio
from src.core.redis import get_redis
from src.modules.queue.scheduler import ModalityScheduler

async def reset_scheduler():
    redis = get_redis()
    s = ModalityScheduler(redis)
    usage = await s.current_usage()
    print(f"Current Usage: {usage}")
    
    # Check if pipeline is blocked
    if usage.get('pipeline', {}).get('current', 0) >= usage.get('pipeline', {}).get('limit', 0):
        print("Pipeline is BLOCKED. Resetting counters...")
        # Delete all sema keys
        keys = await redis.keys("sema:modality:*")
        if keys:
            await redis.delete(*keys)
            print(f"Deleted keys: {keys}")
    else:
        print("Pipeline is not blocked at scheduler level.")

if __name__ == "__main__":
    asyncio.run(reset_scheduler())

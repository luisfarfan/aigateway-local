import redis
import os
from dotenv import load_dotenv

load_dotenv()

r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
keys = r.keys("arq:queue:*")
print(f"Keys: {keys}")
for key in keys:
    length = r.llen(key)
    print(f"Queue {key}: {length}")

# Also check for the job result keys
job_keys = r.keys("arq:job:*")
print(f"Job result keys: {job_keys}")

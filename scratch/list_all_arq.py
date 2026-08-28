import redis
r = redis.Redis(host='localhost', port=6379, db=0)
keys = r.keys("arq:*")
print(f"Total arq keys: {len(keys)}")
for k in keys:
    print(f"{k.decode()} -> {r.type(k).decode()}")

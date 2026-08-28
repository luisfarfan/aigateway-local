from arq.worker import Worker
from workers.main import WorkerSettings

w = Worker(functions=WorkerSettings.functions, queues=WorkerSettings.queues)
print(f"Worker queues: {w.queues}")

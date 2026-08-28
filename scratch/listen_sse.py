import asyncio
import httpx

async def listen():
    url = "http://localhost:8000/api/v1/events?api_key=your-secret-api-key-here"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", url) as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                if line:
                    print(f"Event: {line}")

if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        pass


import asyncio
import os
from src.core.storage import storage

async def diag():
    try:
        # Check buckets
        async with storage._client_ctx() as client:
            buckets = await client.list_buckets()
            print(f"Buckets encontrados: {[b['Name'] for b in buckets['Buckets']]}")
            
            bucket = os.getenv('MINIO_BUCKET', 'local-ai-gateway')
            print(f"Buscando en bucket: {bucket}")
            
            objs = await client.list_objects_v2(Bucket=bucket)
            if 'Contents' in objs:
                print(f"¡ÉXITO! Se encontraron {len(objs['Contents'])} archivos:")
                for o in objs['Contents']:
                    print(f" - {o['Key']} ({o['Size']} bytes)")
            else:
                print("El bucket está vacío.")
    except Exception as e:
        print(f"Error accediendo a MinIO: {e}")

if __name__ == "__main__":
    asyncio.run(diag())

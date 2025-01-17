from chunkr_ai import Chunkr, ChunkrAsync
import asyncio

chunkr = ChunkrAsync()

async def main():
    url = 'https://sweetspot-gov-data-bucket-dev.s3.amazonaws.com/attachments/IHS%2BIEE%2BRepresentation%2BForm.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZBJUQBAJTQ4AHWHH%2F20250117%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250117T002457Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=6bf217cc1ec95408199a645b694a56fd46382fd934b4731d3ceee4ebceb06389'
    task = await chunkr.upload(url)
    print(task)

if __name__ == "__main__":
    asyncio.run(main())

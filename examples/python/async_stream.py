import asyncio

from myvista import AsyncMyVista


async def main() -> None:
    async with AsyncMyVista() as client:
        stream = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        async for chunk in stream:
            print(chunk.delta, end="")
        print()


asyncio.run(main())

import aiohttp
import asyncio

async def register_user(session, i):
    user = f"first_name=first_name_{i}&last_name=last_name_{i}&username=username_{i}&password=password_{i}"
    await session.post("/wrk2-api/user/register", data=user)


async def register_movie(session, i):
    movie = f"title=title_{i}&movie_id=movie_id_{i}"
    await session.post("/wrk2-api/movie/register", data=movie)


async def register_users_and_movies(server_address):
    conn = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(server_address, connector=conn) as session:
        tasks = []
        for i in range(1, 1001, 1):
            task = asyncio.ensure_future(register_user(session, i))
            tasks.append(task)
            task = asyncio.ensure_future(register_movie(session, i))
            tasks.append(task)
            if i % 50 == 0:
                print(f"{i} movies and users finished")
                await asyncio.gather(*tasks)

def main(server_address="http://localhost:30445"):
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(register_users_and_movies(server_address))
    loop.run_until_complete(future)
import ray


ray.init()


@ray.remote
def f():
    return 5


futures = []
for i in range(10):
    futures.append(f.remote())

done, futures = ray.wait(futures, num_returns=1)

for future in done:
    result = ray.get(future)
    print(result)

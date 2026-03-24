# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import time

import ray


ray.init()


@ray.remote
def f():
    time.sleep(1)
    return ray.get_runtime_context().get_worker_id()


futures = []
for _ in range(10):
    futures.append(f.remote())

while futures:
    done, futures = ray.wait(futures, num_returns=1)

    for future in done:
        result = ray.get(future)
        print(result)

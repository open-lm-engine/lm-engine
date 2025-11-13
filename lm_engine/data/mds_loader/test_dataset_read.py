from torch.utils.data import DataLoader
import time
from streaming.base import StreamingDataset

local_dir = '~/data/lhc/temp'
remote_dir = 'gs://nemocc'

dataset = StreamingDataset(local=local_dir, remote=remote_dir, split=None, shuffle=True)

epoch_times = []
for i in range(30):
    start_time = time.time()
    batch_len = 0
    for j in range(10000):
        sample = dataset[j + i * 10000 + 100000]
        batch_len += len(sample['array'])
    print(batch_len)
    end_time = time.time()
    epoch_times.append(end_time - start_time)
    # print(f"Time taken: {end_time - start_time} seconds")

print(epoch_times)
# print(f"Average time taken: {np.mean(epoch_times)} seconds")
from streaming.base import MDSWriter, StreamingDataset
import logging
from argparse import ArgumentParser, Namespace
import os

from tqdm import tqdm

from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset



def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument("--path-folder", type=str, required=True, help="Path to binary file without suffix")

    args = parser.parse_args()

    return args


output_dir = os.path.expanduser("~/data/lhc/mds/")
columns = {"array" : "ndarray:int32"}

# out_tuple = (output_dir, 'gs://nemocc')
with MDSWriter(out=output_dir, columns=columns, exist_ok=True) as out:

    args = get_args()
    folder_path = args.path_folder

    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith('.bin'):
            path = os.path.join(folder_path, filename[0:-4])
            print(path)
            dataset = MMapIndexedDataset(path)

            tokens = 0
            i = 0
            for document in tqdm(dataset):
                #each doc is ndarray 
                out.write({"array": document})
                i += 1

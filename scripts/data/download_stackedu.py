import pyarrow.parquet as pq
import os
from tqdm import tqdm
from queue import Queue
import threading


def split_streaming_prefetch(source_path, output_dir, target_size_mb=1024, prefetch_count=4):
    """
    Overlap disk reads with writes using a prefetch queue.
    While writing row group N, we're already reading row groups N+1, N+2, etc.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pf = pq.ParquetFile(source_path, memory_map=True)
    schema = pf.schema_arrow
    num_groups = pf.num_row_groups
    target_size_bytes = target_size_mb * 1024 * 1024
    
    print(f"Total Row Groups: {num_groups}, Prefetch: {prefetch_count}")
    
    # Prefetch queue: (index, table_chunk, group_size) or None for end
    prefetch_queue = Queue(maxsize=prefetch_count)
    read_error = [None]  # Mutable container to capture errors from reader thread
    
    def prefetch_worker():
        """Background thread that reads row groups ahead of time"""
        try:
            for i in range(num_groups):
                meta = pf.metadata.row_group(i)
                group_size = meta.total_byte_size
                # This is the slow disk read - happens in background
                table_chunk = pf.read_row_group(i, use_threads=True)
                prefetch_queue.put((i, table_chunk, group_size))
            prefetch_queue.put(None)  # Signal end
        except Exception as e:
            read_error[0] = e
            prefetch_queue.put(None)
    
    # Start prefetch thread
    reader_thread = threading.Thread(target=prefetch_worker, daemon=True)
    reader_thread.start()
    
    file_counter = 0
    current_size = 0
    writer = None
    
    try:
        with tqdm(total=num_groups, unit='group', desc="Splitting") as pbar:
            while True:
                item = prefetch_queue.get()
                
                if item is None:
                    # Check for error BEFORE breaking
                    if read_error[0]:
                        raise read_error[0]
                    break
                    
                i, table_chunk, group_size = item
                
                # Start new file if needed
                if writer is None:
                    output_path = os.path.join(output_dir, f"part_{file_counter:05d}.parquet")
                    writer = pq.ParquetWriter(
                        output_path, 
                        schema,
                        compression='snappy',
                        write_statistics=False,
                    )
                    current_size = 0
                
                # Write (while next read is happening in background)
                writer.write_table(table_chunk)
                current_size += group_size
                
                del table_chunk
                pbar.update(1)
                
                # Close file if target reached
                if current_size >= target_size_bytes:
                    writer.close()
                    writer = None
                    file_counter += 1
    finally:
        # Ensure writer is closed even on error
        if writer is not None:
            writer.close()
        reader_thread.join()


# CONFIGURATION
SOURCE_FILE = "/local-ssd/tmpttbf7g2b/tmp-hans/the-stack-v2-dedup-expanded/data/Jupyter_Notebook/train-00000-of-00001.parquet"
OUTPUT_FOLDER = "/local-ssd-fast/stack-v2"

# prefetch_count controls how many row groups to read ahead (tune based on RAM)
split_streaming_prefetch(SOURCE_FILE, OUTPUT_FOLDER, target_size_mb=1024, prefetch_count=4)

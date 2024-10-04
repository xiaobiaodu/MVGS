
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time



data_dir="data/360_v2"
scenes = ["bicycle", "bonsai", "garden", "stump",  "kitchen", "counter", "room"]

factors = [-1]*len(scenes)

MV_normal = [48, 24, 12,8]
MV_largest = [1, 1, 1, 3]

excluded_gpus = set([])

output_dir = "benchmark_360v2_ours"

dry_run = False

jobs = list(zip(scenes, factors))


def train_scene(gpu, scene, factor):

    if scene  in  [ "garden", "stump", "bicycle" ]:
        mvs = MV_largest

    else:
        mvs = MV_normal

    if mvs[0] != 1:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
               f" python train.py -s {data_dir}/{scene} -m {output_dir}/{scene} --eval --white_background"
               f" --mv  {mvs[0]} --iterations 3000 -r 8  "
               f" --port {6009 + int(gpu)} ")
        print(cmd)
        if not dry_run:
            os.system(cmd)

    if mvs[1] != 1:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
               f" python train.py -s {data_dir}/{scene} -m {output_dir}/{scene} --eval --white_background"
               f" --mv  {mvs[1]} --iterations 3000 -r 4 --start_checkpoint {output_dir}/{scene}/chkpnt3000_mv{mvs[0]}.pth "
               f" --port {6009 + int(gpu)} ")
        print(cmd)
        if not dry_run:
            os.system(cmd)

    if mvs[2] != 1:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
               f" python train.py -s {data_dir}/{scene} -m {output_dir}/{scene} --eval --white_background"
               f" --mv  {mvs[2]} --iterations 3000 -r 2 --start_checkpoint {output_dir}/{scene}/chkpnt3000_mv{mvs[1]}.pth "
               f" --port {6009 + int(gpu)} ")
        print(cmd)
        if not dry_run:
            os.system(cmd)

        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
               f" python train.py -s {data_dir}/{scene} -m {output_dir}/{scene} --eval --white_background"
               f" --mv  {mvs[3]} -r {factor} --start_checkpoint {output_dir}/{scene}/chkpnt3000_mv{mvs[2]}.pth "
               f" --port {6009 + int(gpu)} ")
        print(cmd)
        if not dry_run:
            os.system(cmd)


    else:
        cmd = (f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu}"
               f" python train.py -s {data_dir}/{scene} -m {output_dir}/{scene} --eval --white_background"
               f" --mv  {mvs[3]} -r {factor}  "
               f" --port {6009 + int(gpu)} ")
        print(cmd)
        if not dry_run:
            os.system(cmd)


    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --data_device cpu --skip_train"
    print(cmd)
    if not dry_run:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}"
    print(cmd)
    if not dry_run:
        os.system(cmd)
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)


import time
import subprocess
import os
from datetime import datetime
import GPUtil

QUEUE_FILE = "voice/Ganomaly_based/trad_ganomaly/lib/job_queue.txt"
GPU_ID = 0
MAX_GPU_UTIL = 0.5       # 50 %
CHECK_INTERVAL = 10      # seconds between checks
REQUIRED_FREE_TIME = 60  # seconds of idle before launch


def is_gpu_free(gpu_id=GPU_ID, max_util=MAX_GPU_UTIL) -> bool:
    try:
        gpu = GPUtil.getGPUs()[gpu_id]
        return gpu.load < max_util
    except Exception:
        return False


def read_first_job() -> str | None:
    if not os.path.exists(QUEUE_FILE):
        return None
    with open(QUEUE_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[0] if lines else None


def pop_first_job() -> str | None:
    with open(QUEUE_FILE, "r+") as f:
        lines = [line for line in f if line.strip()]
        if not lines:
            return None
        first = lines.pop(0).strip()
        f.seek(0)
        f.truncate()
        f.writelines(line + "\n" for line in lines)
    return first


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main():
    current_proc = None

    while True:
        job = read_first_job()
        if job is None:
            log("Queue empty — sleeping...")
            time.sleep(CHECK_INTERVAL)
            continue

        log(f"Waiting to start: {job}")
        free_time = 0

        # wait until GPU free for REQUIRED_FREE_TIME and no job running
        while True:
            if current_proc is not None and current_proc.poll() is None:
                free_time = 0
            elif is_gpu_free():
                free_time += CHECK_INTERVAL
            else:
                free_time = 0

            if free_time >= REQUIRED_FREE_TIME:
                break
            time.sleep(CHECK_INTERVAL)

        # pop and run job
        confirmed = pop_first_job()
        if confirmed != job:
            log(f"Skipping mismatched job: {confirmed}")
            continue

        log(f"Spawning: {job}")
        current_proc = subprocess.Popen(job, shell=True)
        log(f"Started PID={current_proc.pid}")

if __name__ == "__main__":
    main()

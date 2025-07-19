#!/usr/bin/env python3

import time
import subprocess
import os
import sys
import argparse
from datetime import datetime
import fcntl
import GPUtil
import threading
import termios
import tty

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple GPU-aware job queue scheduler"
    )
    parser.add_argument(
        "--queue-file", "-q",
        default="job_queue.txt",
        help="Path to the job queue file (one command per line)"
    )
    parser.add_argument(
        "--gpu-id", "-g",
        type=int,
        default=0,
        help="GPU index to monitor"
    )
    parser.add_argument(
        "--max-util", "-u",
        type=float,
        default=0.5,
        help="Max GPU load (0–1) to consider it free"
    )
    parser.add_argument(
        "--check-interval", "-i",
        type=int,
        default=10,
        help="Seconds between GPU/util checks"
    )
    parser.add_argument(
        "--free-time", "-f",
        type=int,
        default=120,
        help="Consecutive seconds GPU must stay under max-util"
    )
    parser.add_argument(
        "--log-file", "-l",
        default="gpu_scheduler.log",
        help="Path to the log file"
    )
    return parser.parse_args()

def lock_and_read_first_line(path):
    """Open file with shared lock and peek first nonempty line."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        for line in f:
            text = line.strip()
            if text:
                fcntl.flock(f, fcntl.LOCK_UN)
                return text
        fcntl.flock(f, fcntl.LOCK_UN)
    return None

def lock_and_pop_first_line(path):
    """Open file with exclusive lock, remove first nonempty line, and write back."""
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = [l.strip() for l in f if l.strip()]
        if not lines:
            fcntl.flock(f, fcntl.LOCK_UN)
            return None
        first = lines.pop(0)
        f.seek(0)
        f.truncate()
        for l in lines:
            f.write(l + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return first

def log(msg, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    if log_file:
        with open(log_file, "a") as f:
            f.write(line)
    else:
        print(line, flush=True)

def is_gpu_free(gpu_id, max_util):
    try:
        gpu = GPUtil.getGPUs()[gpu_id]
        return gpu.load < max_util
    except Exception:
        return False

def terminal_control(paused_flag, spawn_now_flag, kill_flag):
    """Thread: Listen for 'p' (pause/resume), 'n' (next), 'k' (kill current) keypresses."""
    print("Controls: [p]ause/resume, [n]ext job immediately, [k]ill current job")
    while True:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "p":
                paused_flag[0] = not paused_flag[0]
                print(f"\n{'Paused' if paused_flag[0] else 'Resumed'} job spawning.")
            elif ch == "n":
                spawn_now_flag[0] = True
                print("\nWill spawn next job immediately.")
            elif ch == "k":
                kill_flag[0] = True
                print("\nKill signal sent to current job.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    args = parse_args()
    queue_file = os.path.abspath(args.queue_file)
    log_file = os.path.abspath(args.log_file)
    current_proc = None

    paused_flag = [False]  # mutable flag for pause/resume
    spawn_now_flag = [False]  # mutable flag for immediate spawn
    kill_flag = [False]  # mutable flag for killing current process

    # Start terminal control thread
    t = threading.Thread(target=terminal_control, args=(paused_flag, spawn_now_flag, kill_flag), daemon=True)
    t.start()

    while True:
        job = lock_and_read_first_line(queue_file)
        if not job:
            log(f"Queue empty - sleeping for {args.check_interval}s", log_file)
            time.sleep(args.check_interval)
            continue

        log(f"Next job: {job}", log_file)
        idle_counter = 0

        while True:
            # Pause logic
            if paused_flag[0]:
                log("Paused. Waiting...", log_file)
                while paused_flag[0]:
                    time.sleep(1)
                log("Resumed.", log_file)

            # Kill logic
            if kill_flag[0] and current_proc is not None and current_proc.poll() is None:
                log("Killing current job via keyboard interrupt.", log_file)
                try:
                    current_proc.send_signal(subprocess.signal.SIGINT)
                except Exception as e:
                    log(f"Failed to send SIGINT: {e}", log_file)
                kill_flag[0] = False

            # if previous job still running, reset idle timer
            if current_proc is not None and current_proc.poll() is None:
                log("Previous job still running - reset idle counter", log_file)
                idle_counter = 0
            else:
                if is_gpu_free(args.gpu_id, args.max_util):
                    idle_counter += args.check_interval
                    log(f"GPU idle for {idle_counter}/{args.free_time}s", log_file)
                else:
                    if idle_counter > 0:
                        log("GPU busy - reset idle counter", log_file)
                    idle_counter = 0

            # detect if user removed or reordered this job
            first_now = lock_and_read_first_line(queue_file)
            if first_now != job:
                log(f"Job changed or removed - skipping '{job}'", log_file)
                job = None
                break

            # Immediate spawn logic
            if spawn_now_flag[0]:
                log("Immediate spawn requested via terminal.", log_file)
                spawn_now_flag[0] = False
                break

            if idle_counter >= args.free_time:
                break

            time.sleep(args.check_interval)

        if not job:
            continue

        confirmed = lock_and_pop_first_line(queue_file)
        if confirmed != job:
            log(f"Queue pop mismatch (got '{confirmed}') - skipping", log_file)
            continue

        log(f"Launching: {job}", log_file)
        # ensure you spawn within the same Python interpreter (venv)
        with open(log_file, "a") as lf:
            current_proc = subprocess.Popen(
                job,
                shell=True,
                env=os.environ,
                stdout=lf,
                stderr=lf
            )
            log(f"Started PID={current_proc.pid}", log_file)
            while True:
                # Check for kill signal during job execution
                if kill_flag[0] and current_proc.poll() is None:
                    log("Killing current job via keyboard interrupt.", log_file)
                    try:
                        current_proc.send_signal(subprocess.signal.SIGINT)
                    except Exception as e:
                        log(f"Failed to send SIGINT: {e}", log_file)
                    kill_flag[0] = False
                if current_proc.poll() is not None:
                    break
                time.sleep(1)
            log(f"Job PID={current_proc.pid} finished with return code {current_proc.returncode}", log_file)

if __name__ == "__main__":
    main()

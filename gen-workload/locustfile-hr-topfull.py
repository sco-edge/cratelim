#!/usr/bin/python3

from locust import HttpUser, LoadTestShape, task, events
import locust.stats
import random
import logging
import time
import json
import numpy as np
from pathlib import Path

mean_iat = 1 
request_log_file = open('request.log', 'a')

class HRUser(HttpUser):
    proxies = {
        "http": "http://localhost:8091",
        "https": "http://localhost:8091"
    }

    def on_start(self):
        self.client.proxies = self.proxies

    # return wait time in second
    def wait_time(self):
        global mean_iat
        return random.expovariate(lambd=1/mean_iat)

    @events.request.add_listener
    def on_request(response_time, context, **kwargs):
        response_code = kwargs.get("response").status_code if kwargs.get("response") else None
        request_log_file.write(json.dumps({
            'time': time.perf_counter(),
            'latency': response_time / 1e3,
            'context': context,
            'response_code': response_code,
        }) + '\n')

    # @task
    # def search_hotel(self):
    #     in_date = random.randint(9, 23)
    #     out_date = random.randint(in_date+1, 24)

    #     in_date = "2015-04-23"
    #     out_date = "2015-04-25"
    #     lat = "37.8000"
    #     lon = "-122.4000"

    #     path = '/hotels?inDate=' + in_date + '&outDate=' + out_date + \
    #         '&lat=' + str(lat) + "&lon=" + str(lon)

    #     headers = {"api-context": "hotels"}

    #     self.client.get(path, name='search_hotel',
    #         context={'type': 'search_hotel'},
    #         headers=headers)
        
    @task()
    def reservation(self):
        in_date = "2015-04-23"
        out_date = "2015-04-25"
        lat = "37.8000"
        lon = "-122.4000"
        hotel_id = "6"
        customer_name = "Cornell_37"
        username = "Cornell_37"
        password = "7777777777"
        number = "1"

        path = '/reservation?inDate=' + in_date + '&outDate=' + out_date + \
            '&lat=' + str(lat) + "&lon=" + str(lon) + \
            '&hotelId=' + hotel_id + '&customerName=' + customer_name + \
            '&username=' + username + '&password=' + password + '&number=' + number
        headers = {"api-context": "reservation"}

        self.client.post(path, name='reservation',
            context={'type': 'reservation'},
            headers=headers)


# logging.basicConfig(level=logging.INFO)
trace_path = "traces/bursty-short-test.txt"
with open(trace_path, "r") as file:
    RPS = [int(line.strip()) * 5 for line in file]

class CustomShape(LoadTestShape):
    time_limit = len(RPS)
    spawn_rate = 100

    def tick(self):
        run_time = self.get_run_time()
        if run_time < self.time_limit:
            user_count = RPS[int(run_time)]
            return (user_count, self.spawn_rate)
        return None
#!/bin/python3

import re
import sys
import asyncio
import aiohttp
import json
import time
import logging
import argparse
import subprocess
import numpy as np

response_times = []
failed_responses = 0

async def send_http_request(session, port, parameters, headers):
    global failed_responses
    try:
        start_time = time.perf_counter()
        async with session.get(f"http://localhost:{port}/hotels", params=parameters, headers=headers) as response:
            logging.debug(response)
            body = await response.text()
            logging.debug(body)
            if response.status == 200:
                end_time = time.perf_counter()
                response_time = end_time - start_time
                response_times.append(response_time)    
            else:
                failed_responses += 1

    except Exception as e:
        logging.error(f"Error sending request: {e}")
        failed_responses += 1

async def http_test_with_rps(rps_list, duration, port, params, headers):
    logging.debug(f"Starting load test with dynamic RPS for {duration} seconds")
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=12000)
    async with aiohttp.ClientSession(connector=connector) as session:
        current_second = 0

        while current_second < len(rps_list) and time.time() - start_time < duration:
            rps = rps_list[current_second]
            interval = 1 / rps if rps > 0 else 1
            tasks = []

            logging.info(f"Sending {rps} requests for second {current_second}")
            for _ in range(rps):
                task = asyncio.create_task(send_http_request(session, port, params, headers))
                tasks.append(task)
                await asyncio.sleep(interval)

            await asyncio.gather(*tasks)
            current_second += 1

    if response_times:
        logging.info(f"Load test completed")
        print(f"{len(response_times)} {failed_responses} {np.mean(response_times)} {np.percentile(response_times, 50)}",
              f"{np.percentile(response_times, 90)} {np.percentile(response_times, 95)}",
              f"{np.percentile(response_times, 99)} {np.percentile(response_times, 99.9)}")
    else:
        logging.warning("No successful responses were collected during the load test.")

    logging.info("Load test completed")

async def http_test(rps, duration, port, params, headers):
    logging.debug(f"Starting load test with varying RPS.")
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=12000)
    async with aiohttp.ClientSession(connector=connector) as session:
        next_request_time = start_time
        tasks = []

        logging.info(f"Sending {rps[0]} requests for second")
        while time.time() - start_time < duration:
            now = time.time()
            elapsed_second =  int(now - start_time)
            sleep_time = max(0, next_request_time - now)
            await asyncio.sleep(sleep_time)

            task = asyncio.create_task(send_http_request(session, port, params, headers))
            tasks.append(task)

            interval = 1 / rps[elapsed_second]
            next_request_time += interval

        await asyncio.gather(*tasks)

    if response_times:
        logging.debug(f"Load test completed")
        # print(f"{len(response_times)} {failed_responses} {np.mean(response_times)} {np.percentile(response_times, 50)}",
        #       f"{np.percentile(response_times, 90)} {np.percentile(response_times, 95)}",
        #       f"{np.percentile(response_times, 99)} {np.percentile(response_times, 99.9)}")
        print(f"{len(response_times)} {failed_responses} {np.mean(response_times)*1000:.3f} {np.percentile(response_times, 50)*1000:.3f}",
              f"{np.percentile(response_times, 90)*1000:.3f} {np.percentile(response_times, 95)*1000:.3f}",
              f"{np.percentile(response_times, 99)*1000:.3f} {np.percentile(response_times, 99.9)*1000:.3f}")
        
    else:
        logging.warning("No successful responses were collected during the load test.")

    logging.debug("Load test completed")


# Example usage:
# rps_list = [10, 20, 30, 40, 50]  # RPS for each second
# asyncio.run(http_test_with_rps(rps_list, len(rps_list), '127.0.0.1', 8080, {}, {}))

parser = argparse.ArgumentParser()
parser.add_argument('--log', '-l', default="warning", help="critical|error|warning|info|debug")
args = parser.parse_args()

log_level = getattr(logging, args.log.upper(), None)
if log_level is None:
    print(f"{args.log} is not an available log level. Available: critical, error, warning, info, debug")
    sys.exit(1)
logging.basicConfig(level=log_level)

# logging.basicConfig(level=logging.INFO)
trace_path = "traces/bursty-short-test.txt"
# trace_path = "traces/bursty-short-test2.txt"
with open(trace_path, "r") as file:
    # rps = [int(line.strip()) * 5 for line in file]
    rps = [int(line.strip()) for line in file]
    # rps = [30]

print(np.sum(rps))
params = {"inDate": "2015-04-23", "outDate": "2015-04-25", "lat": "37.8000", "lon": "-122.4000"}
headers = {"api-context": "hotels"}
# asyncio.run(http_test_with_rps(rps, len(rps), "8080", params, headers))
asyncio.run(http_test(rps, 5, "8080", params, headers))
#!/usr/bin/python3
import os
import re
import time
import json
import logging
import datetime
import argparse
import subprocess
import numpy as np

def enable_lua_logging(microservice):
    try:
        logging.info("enable_lua_logging(): fetching all pods.")
        max_retries = 5
        retries = 0

        while retries < max_retries:
            cmd = ["kubectl", "get", "pods", "-A"]
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            output_lines = result.stdout.splitlines()
            for line in output_lines[1:]:
                tokens = line.split()
                if len(tokens) > 3 and tokens[1].startswith(microservice):
                    if tokens[3] == "Running":
                        cmd = ["istioctl", "proxy-config", "log", f"{tokens[1]}.default", "--level", "lua=info"]
                        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

                        cmd1 = ["istioctl", "proxy-config", "log", f"{tokens[1]}.default"]
                        cmd2 = ["grep", "lua"]
                        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, text=True)
                        p2 = subprocess.Popen(cmd2, stdin=p1.stdout, stdout=subprocess.PIPE, text=True)

                        output, error = p2.communicate()
                        if p2.returncode == 0 and "info" in output:
                            retries = 5
                        else:
                            logging.error(f"enable_lua_logging(): {error} retries: {retries}")
            retries += 1

    except subprocess.CalledProcessError as e:
        logging.error("enable_lua_logging(): an error occurred while executing kubectl commands.")
        logging.error(e.stderr)

def parse_loki_log(rate, duration, microservice, api, arrivals, departures, range_from, range_to):
    data = []
    backlogs = [0] * (len(arrivals) + 1)
    max_backlogs = [0] * (len(arrivals) + 1)
    start_timestamp = datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
    end_timestamp = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    try:
        target = f"{microservice}.{api}.{rate}"
        logging.info(f'Parse loki log {target}')

        max_retries = 10
        retries = 0
        while retries < max_retries:
            time.sleep(0.5)

            cmd = ["logcli", "query", f'{{app="{microservice}"}}', f'--from={range_from}', f'--to={range_to}', '--limit=0', '--forward']
            logging.info(" ".join(cmd))
            with open(f"{target}.log", "w") as log_file:
                result = subprocess.run(cmd, text=True, capture_output=True, check=True, cwd=profiler_cwd)
                log_file.write(result.stdout)
        
            pattern = (
                r"^(?P<first_timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"
                r".*?"
                r"(?P<second_timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"
                r"\s+info\s+envoy\s+lua.*?script log:\s+"
                r"(?P<direction>[IO])\s+"
                r"(?P<type>[A-Z]+)"
                r"(?:\s+(?P<status>\d{3}))?"
                r"(?:\s+(?P<path>/[^\s]+))?"
                r"\s+(?P<api_context>[^\s]+)"
                r"\s+(?P<progress_context>[^\s]+)"
                r".*thread=(?P<thread_id>\d+)"
            )

            with open(f"{target}.log", "r") as log_file:
                measured_timestamps = []
                for line in log_file:
                    match = re.match(pattern, line)
                    if match:
                        timestamp, interval_measured = calculate_backlog(arrivals, departures, data, backlogs, max_backlogs, match.groupdict())

                        if start_timestamp > timestamp:
                            start_timestamp = timestamp
                        if end_timestamp < timestamp:
                            end_timestamp = timestamp

                        if interval_measured:
                            measured_timestamps.append(timestamp)
                        
            # We allow duration * 0.9
            logging.debug(f"Time window: {abs(end_timestamp - start_timestamp)} {start_timestamp} {end_timestamp}")
            if abs(end_timestamp - start_timestamp) >= datetime.timedelta(milliseconds=(duration * 0.9)):
                break
            else:
                logging.debug(f"Not sufficient: {abs(end_timestamp - start_timestamp)} {start_timestamp} {end_timestamp}")
                retries += 1

        if len(measured_timestamps) > 1:
            intervals = []
            for i in range(1, len(measured_timestamps)):
                interval = (measured_timestamps[i] - measured_timestamps[i-1]).total_seconds()
                intervals.append(interval)
            print(np.average(intervals), np.std(intervals), 1/rate)
            return max_backlogs, np.average(intervals)
        logging.error("parse_loki_log() has no measured timestamps")

    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)

def calculate_backlog(arrivals, departures, data, backlogs, max_backlogs, parsed_line):
    interval_measured = False
    for i, condition in enumerate(arrivals):
        if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
            backlogs[0] += 1
            backlogs[i+1] += 1
            data.append((parsed_line['second_timestamp'], tuple(backlogs)))
            if backlogs[i+1] > max_backlogs[i+1]:
                max_backlogs[i+1] = backlogs[i+1]
            if backlogs[0] > max_backlogs[0]:
                max_backlogs[0] = backlogs[0]

            if parsed_line['direction'] == 'I' and parsed_line['type'] == 'REQ' and parsed_line['progress_context'] == 'no-progress':
                interval_measured = True
            
    for i, condition in enumerate(departures):
        if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
            backlogs[0] -= 1
            backlogs[i+1] -= 1
            data.append((parsed_line['second_timestamp'], tuple(backlogs)))

    return datetime.datetime.fromisoformat(parsed_line['second_timestamp'].replace("Z", "+00:00")), interval_measured
def clear_envoyfilter():
    try:
        logging.info(f"Clear envoyfilters")
        result = subprocess.run(
            ["kubectl", "get", "envoyfilter"],
            text=True,
            capture_output=True,
            check=True
        )
        output_lines = result.stdout.splitlines()
        target_envoyfilters = []
        for line in output_lines[1:]:
            parts = line.split()
            if len(parts) > 1:
                target_envoyfilters.append(parts[0])

        logging.info(f"Found {len(target_envoyfilters)} filters")
        for envoyfilter in target_envoyfilters:
            try:
                logging.info(f"Deleting {envoyfilter}")
                subprocess.run(
                    ["kubectl", "delete", "envoyfilter", envoyfilter],
                    check=True,
                    text=True,
                    stdout=subprocess.DEVNULL
                )
                logging.info(f"Successfully deleted envoyfilter: {envoyfilter}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to delete envoyfilter: {envoyfilter}")
                logging.error(e.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)

def apply_envoyfilter(target_app, target_microservice, target_api):
    spans = None
    with open(os.path.join(profiler_cwd, f'{target_app}/spec.json'), 'r') as file:
        data = json.load(file)
        
    if not target_app == data.get("app"):
        logging.error("application ids are different.")

    microservices = data.get("microservices", [])
    for microservice in microservices:
        if microservice.get("id") == target_microservice:
            apis = microservice.get("apis", [])
            for api in apis:  
                if api.get("id") == target_api:
                    if microservice.get("gateway") == True:
                        port = 5000
                    else:
                        port = microservice.get("port")
                    spans = api.get("spans")

    if port is None or spans is None:
        raise ValueError(f"Unsupported target: microservice: {target_microservice} api: {target_api}")

    inbound_envoyfilter = f"""
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: rs.{target_app}.{target_microservice}.{target_api}.inbound
spec:
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
      listener:
        portNumber: {port}
        filterChain:
          filter:
            name: "envoy.filters.network.http_connection_manager"
            subFilter:
              name: "envoy.filters.http.router"
    patch:
      operation: INSERT_BEFORE
      value: 
       name: envoy.lua
       typed_config:
          "@type": "type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua"
          default_source_code:
            inline_string: |
              function envoy_on_request(request_handle)
                local path = request_handle:headers():get(":path") or "no-path"
                local api_context = request_handle:headers():get("api-context") or "no-api"
                local progress_context = request_handle:headers():get("progress-context") or "no-progress"
                request_handle:logInfo("I REQ " .. path .. " " .. api_context .. " " .. progress_context)
              end
              function envoy_on_response(response_handle)
                local status = response_handle:headers():get(":status") or "no-status"
                local api_context = response_handle:headers():get("api-context") or "no-api"
                local progress_context = response_handle:headers():get("progress-context") or "no-progress"
                response_handle:logInfo("I RES " .. status .. " " .. api_context .. " " .. progress_context)
              end
"""
    try:
        logging.info(f"Apply envoyfilter: rs.{target_app}.{target_microservice}.{target_api}.inbound")
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=inbound_envoyfilter,
            text=True,
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)
    
    for span in spans:
        for microservice in microservices:
            if microservice.get("id") == span["microservice"]:
                apis = microservice.get("apis", [])
                for api in apis:  
                    if api.get("id") == span["api"]:
                        if microservice.get("gateway") == True:
                            span["port"] = 5000
                        else:
                            span["port"] = microservice.get("port")

    for span in spans:
        outbound_envoyfilter = f"""
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: rs.{target_app}.{target_microservice}.{target_api}.{span["microservice"]}.{span["api"]}
spec:
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_OUTBOUND
      listener:
        portNumber: {span["port"]}
        filterChain:
          filter:
            name: "envoy.filters.network.http_connection_manager"
            subFilter:
              name: "envoy.filters.http.router"
    patch:
      operation: INSERT_BEFORE
      value: 
       name: envoy.lua
       typed_config:
          "@type": "type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua"
          default_source_code:
            inline_string: |
              function envoy_on_request(request_handle)
                local path = request_handle:headers():get(":path") or "no-path"
                local api_context = request_handle:headers():get("api-context") or "no-api"
                local progress_context = request_handle:headers():get("progress-context") or "no-progress"
                request_handle:logInfo("O REQ " .. path .. " " .. api_context .. " " .. progress_context)
              end
              function envoy_on_response(response_handle)
                local status = response_handle:headers():get(":status") or "no-status"
                local api_context = response_handle:headers():get("api-context") or "no-api"
                local progress_context = response_handle:headers():get("progress-context") or "no-progress"
                response_handle:logInfo("O RES " .. status .. " " .. api_context .. " " .. progress_context)
              end
"""
        try:
            logging.info(f'Apply envoyfilter: rs.{target_app}.{target_microservice}.{target_api}.{span["microservice"]}.{span["api"]}')
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=outbound_envoyfilter,
                text=True,
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(e.stderr)

if __name__ == "__main__":
    script_cwd = os.getcwd()
    profiler_cwd = os.path.join(script_cwd, "../profiler")

    parser = argparse.ArgumentParser()
    parser.add_argument('app')
    parser.add_argument('--log', '-l', default="warning")
    args = parser.parse_args()

    if args.log == "critical":
        logging.basicConfig(level=logging.CRITICAL)
    elif args.log == "error":
        logging.basicConfig(level=logging.ERROR)
    elif args.log == "warning":
        logging.basicConfig(level=logging.WARNING)
    elif args.log == "info":
        logging.basicConfig(level=logging.INFO)
    elif args.log == "debug":
        logging.basicConfig(level=logging.DEBUG)
    else:
        print(f"{args.log} is not an available log level. Available: critical, error, warning, info, debug")
        exit()

    rate = 10
    duration = 2000
    microservice = "frontend"
    api = "frontend-hotels"

    arrivals = []
    departures = []
    arrivals.append(('I', 'REQ', 'no-progress'))
    
    clear_envoyfilter()
    apply_envoyfilter(args.app, microservice, api)
    enable_lua_logging(microservice)
    range_from = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    time.sleep(5)

    range_to = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    parse_loki_log(rate, duration, microservice, api, arrivals, departures, range_from, range_to)
    clear_envoyfilter()
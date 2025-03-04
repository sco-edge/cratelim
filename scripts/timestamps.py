#!/usr/bin/python3
import os
import re
import sys
import time
import json
import signal
import logging
import datetime
import argparse
import subprocess
import numpy as np

models = {"closed", "constant", "bursty", "random"}

def bg_port_forward():
    bg_processes = []
    bg_processes.append(subprocess.Popen(
        ["kubectl", "port-forward", "-n", "loki", "svc/loki", "3100:3100"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    ))
    bg_processes.append(subprocess.Popen(
        ["kubectl", "port-forward", "svc/hr-gateway-istio", "8080:80"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    ))

    return bg_processes

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
                        timestamp = None
                        parsed_line = match.groupdict()
                        for i, condition in enumerate(arrivals):
                            if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
                                timestamp = datetime.datetime.fromisoformat(parsed_line['second_timestamp'].replace("Z", "+00:00"))
                                data.append(timestamp)

                                if parsed_line['direction'] == 'I' and parsed_line['type'] == 'REQ' and parsed_line['progress_context'] == 'no-progress':
                                    interval_measured = True
                        
                        if not timestamp:
                            continue
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
            return data
        logging.error("parse_loki_log() has no measured timestamps")

    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)

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

def run(model, trace):
    if model == "closed":
        try:
            cmd = ["locust", '--headless', '-f=closed-hr.py', '-H=http://localhost:8080', f'--log={args.log}', f'--trace={trace}']
            logging.info(" ".join(cmd))
            subprocess.run(cmd, text=True, capture_output=True, check=True, cwd=workload_gen_cwd)
        except subprocess.CalledProcessError as e:
            logging.error(e.stderr)

    elif model == "constant":
        try:
            cmd = ["constant-hr.py", f'--log={args.log}', f'--trace={trace}']
            logging.info(" ".join(cmd))
            result = subprocess.run(cmd, text=True, capture_output=True, check=True, cwd=workload_gen_cwd)
        except subprocess.CalledProcessError as e:
            logging.error(e.stderr)

    elif model == "bursty":
        try:
            cmd = ["bursty-hr.py"]
            logging.info(" ".join(cmd))
            subprocess.run(cmd, text=True, capture_output=True, check=True, cwd=workload_gen_cwd)
        except subprocess.CalledProcessError as e:
            logging.error(e.stderr)


if __name__ == "__main__":
    script_cwd = os.getcwd()
    profiler_cwd = os.path.join(script_cwd, "../profiler")
    workload_gen_cwd = os.path.join(script_cwd, "../gen-workload")

    parser = argparse.ArgumentParser()
    parser.add_argument('--app', '-a', default="hr")
    parser.add_argument('--model', '-m', default="closed", help="closed|constant|bursty|random")
    parser.add_argument('--trace', '-t', default="bursty-short")
    parser.add_argument('--log', '-l', default="warning", help="critical|error|warning|info|debug")
    args = parser.parse_args()

    if not args.model in models:
        print(f"{args.model} is not an available model. Available: closed, constant, bursty, random")
        sys.exit(1)
    model = args.model
    
    if not os.path.exists(os.path.join(workload_gen_cwd, f"traces/{args.trace}.txt")):
        print(f"{args.trace} is not an available trace. Check {workload_gen_cwd}/traces directory.")
        sys.exit(1)
    trace = args.trace

    log_level = getattr(logging, args.log.upper(), None)
    if log_level is None:
        print(f"{args.log} is not an available log level. Available: critical, error, warning, info, debug")
        sys.exit(1)
    logging.basicConfig(level=log_level)

    rate = 10
    duration = 2000
    microservice = "frontend"
    api = "frontend-hotels"

    arrivals = []
    departures = []
    arrivals.append(('I', 'REQ', 'no-progress'))

    try:
        apply_envoyfilter(args.app, microservice, api)
        enable_lua_logging(microservice)
        bg_processes = bg_port_forward()
        time.sleep(0.5)
        range_from = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        run(model, trace)
        range_to = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        data = parse_loki_log(rate, duration, microservice, api, arrivals, departures, range_from, range_to)
    finally:
        clear_envoyfilter()
        if bg_processes:
            for bg_process in bg_processes:
                bg_process.terminate()
                bg_process.wait()

    if data:
        with open(f"{model}.{trace}.server.log", "w") as timestamp_file:
            for timestamp in data:
                timestamp_file.write(str(timestamp) + '\n')
    else:
        logging.error("No server-side data to write.")

    with open(f"{model}.{trace}.client.log", 'w') as client_file, open(os.path.join(workload_gen_cwd, f"{model}.log")) as file:
        for line in file:
            response = json.loads(line.strip())
            line = f'{response["time"]} {response["latency"]} {response["response_code"]}\n'
            client_file.write(line)

            

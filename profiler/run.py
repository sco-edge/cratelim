#!/bin/python3
import re
import os
import sys
import time
import json
import datetime
import logging
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as mp

def get_profiling_targets(spec):
    targets = []

    microservices = spec.get("microservices", [])
    for microservice in microservices:
        apis = microservice.get("apis", [])
        for api in apis:  
            targets.append((microservice.get("id"), api.get("id"), microservice.get("gateway") is True))

    return targets

def get_spans_dependencies(spec, target_microservice, target_api):
    microservices = spec.get("microservices", [])
    for microservice in microservices:
        if microservice.get("id") == target_microservice:
            apis = microservice.get("apis", [])
            for api in apis:  
                if api.get("id") == target_api:
                    return api.get("spans"), api.get("dependencies")

    logging.error("get_spans_dependencies() failed")

def parse_response_times(data):
    # Extract response_times from JSON
    response_times = data[0]["response_times"]

    # Create a list of all response times with their occurrences
    response_data = []
    for response_time, count in response_times.items():
        response_time = float(response_time)  # Ensure response time is float
        response_data.extend([response_time] * count)  # Add each response time `count` times

    # Convert to NumPy array for statistical calculations
    response_array = np.array(response_data)

    # Calculate statistics
    average = np.mean(response_array)
    p50 = np.percentile(response_array, 50)  # Median
    p90 = np.percentile(response_array, 90)
    p99 = np.percentile(response_array, 99)
    p999 = np.percentile(response_array, 99.9)

    # Create histogram data
    histogram, bins = np.histogram(response_array, bins='auto')

    # Return results
    return {
        "average": average,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "p99.9": p999,
        "histogram": {
            "bins": bins.tolist(),
            "counts": histogram.tolist()
        }
    }

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

def parse_log_line(log_line):
    log_pattern = (
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"
        r".*?"
        r"info\s+envoy\s+lua.*?script log:\s+(?P<direction>[IO])\s+(?P<type>[A-Z]+)"  # Direction and type (I/O, REQ/RES)
        r"(?:\s+(?P<status>\d{3}))?"
        r"(?:\s+(?P<path>/[^\s]+))?"
        r"\s+(?P<api_context>[^\s]+)"
        r"\s+(?P<progress_context>[^\s]+)"
        r".*thread=(\d+)" # thread ID
    )
    match = re.search(log_pattern, log_line)
    if match:
        return match.groupdict()
    return None

def plot(data, file_name, len_span):
    timestamps = [datetime.datetime.fromisoformat(item[0][:23]) for item in data]
    values = [item[1] for item in data]

    mp.figure(figsize=(12, 6))

    total_col = [v[0] for v in values]
    mp.plot(timestamps, total_col, label='Total')

    for i in range(len_span):
        col = [v[i+1] for v in values]
        mp.plot(timestamps, col, label=i+1)

    mp.xlabel('Time')
    mp.ylabel('Backlogs')
    mp.title('Backlogs Over Time')
    mp.legend()
    mp.grid(True)
    mp.xticks(rotation=45)
    mp.tight_layout()

    # start_time = datetime.fromisoformat("2025-01-12T13:08:23.643984571Z"[:23])
    # end_time = datetime.fromisoformat("2025-01-12T13:08:33.643984571Z"[:23])
    # mp.xlim(left=start_time, right=end_time)

    output_path = f"{file_name}.png"
    mp.savefig(output_path, dpi=300, bbox_inches='tight')
    mp.close()

def parse_log(arrivals, departures, results, logs):
    for log in logs:
        # max_backlog = 0
        # backlog = 0
        data = []
        backlogs = [0] * (len(arrivals) + 1)
        max_backlogs = [0] * (len(arrivals) + 1)

        with open(f"{log}", 'r') as file:
            for line in file:
                parsed_line = parse_log_line(line.strip())
                if parsed_line:
                    # if is_arrival(parsed_line):
                    #     backlog += 1
                    # else:
                    #     backlog -= 1
                    # if backlog > max_backlog:
                    #     max_backlog = backlog
                    calculate_backlog(arrivals, departures, data, backlogs, max_backlogs, parsed_line)

        # for datum in data:
        #     print(datum)
        plot(data, log, len(arrivals))
        results['max_backlogs'] = max_backlogs
        # print(f'{log} max backlog: {max_backlogs}')

def run_wrk2(rps):
    wrk2_command = [
    "wrk2",
    "-t5",
    "-c50",
    "-d10s",
    f"-R{rps}",
    "--script=hr-script.lua",
    "http://localhost:8080",
    ]

    try:
        result = subprocess.run(wrk2_command, check=True, text=True, capture_output=True)
        logging.info("wrk2 execution completed successfully.")
        logging.debug(result.stdout)
        # return parse_response_times(json_data)
    except subprocess.CalledProcessError as e:
        logging.error("wrk2 execution failed:")
        logging.error(e.stderr)

def run_constant_interval(rps, duration, app, microservice, api):
    command = ["./constant-interval-hr.py", "--rps", str(rps), "--duration", str(duration), "-a", app, "-m", microservice, "-i", api]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=os.path.join(profiler_cwd, app))
        logging.info("Load script execution completed successfully.")
        responses = result.stdout
    except subprocess.CalledProcessError as e:
        logging.error("Load script execution failed:")
        logging.error(e.stderr)
    
    return responses.split()

def run_locust(rps):
    # if rps < 10:
    #     logging.warning("The lower bound of rps is 10.")
    #     rps = 10
    
    locust_command = [
        "locust",
        "--headless",
        "--json",
        "--processes", "10",
        "-f", "locustfile-hr.py",
        "-u", str(rps * 10),
        "-r", str(rps),
        "-H", "http://localhost:8080",
        "-t", "10"
    ]

    try:
        result = subprocess.run(locust_command, check=True, text=True, capture_output=True)
        logging.info("Locust execution completed successfully.")
        logging.debug(result.stdout)
        valid_json_lines = []
        for line in result.stdout.splitlines():
            if line.strip().startswith("[]"):
                continue
            valid_json_lines.append(line)
        json_data = json.loads("".join(valid_json_lines))
        return parse_response_times(json_data)
    except subprocess.CalledProcessError as e:
        logging.error("Locust execution failed:")
        logging.error(e.stderr)

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

def init_pods(microservice, dependencies):
    try:
        logging.info("Fetching all pods...")
        result = subprocess.run(
            ["kubectl", "get", "pods", "-A"],
            check=True,
            text=True,
            capture_output=True
        )

        logging.info("Parsing pod information...")
        output_lines = result.stdout.splitlines()
        target_pods = []
        for line in output_lines[1:]:
            parts = line.split()
            if len(parts) > 1 and (
                parts[1].startswith(microservice) or
                any(parts[1].startswith(dependency) for dependency in dependencies)
            ):
                namespace = parts[0]
                pod_name = parts[1]
                target_pods.append((namespace, pod_name))

        if not target_pods:
            logging.error(f"No '{microservice}' pods found.")
            return

        logging.info(f"Found {len(target_pods)} '{microservice}' and its dependencies {dependencies} pods")
        for namespace, pod_name in target_pods:
            try:
                logging.info(f"Deleting pod: {pod_name} in namespace: {namespace}")
                subprocess.run(
                    ["kubectl", "delete", "pod", pod_name, "-n", namespace],
                    check=True,
                    text=True,
                    stdout=subprocess.DEVNULL
                )
                logging.info(f"Successfully deleted pod: {pod_name}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to delete pod: {pod_name}")
                logging.error(e.stderr)

        max_retries = 10
        interval = 1
        retries = 0

        while retries < max_retries:
            logging.info(f"Checking if pods are recreated... (Attempt {retries + 1}/{max_retries})")
            result = subprocess.run(
                ["kubectl", "get", "pods", "-A"],
                check=True,
                text=True,
                capture_output=True
            )
            output_lines = result.stdout.splitlines()
            recreated_pods = []
            for line in output_lines[1:]:
                parts = line.split()
                if (
                    len(parts) > 3 and
                    parts[1].startswith(microservice) and
                    parts[3] == "Running"
                ):
                    result = subprocess.run(
                        ["istioctl", "proxy-config", "log", f"{parts[1]}.default", "--level", "lua=info"],
                        check=True,
                        text=True,
                        capture_output=True
                    )
                    recreated_pods.append(parts[1])

            if recreated_pods:
                logging.info(f"Recreated '{microservice}' pods: {recreated_pods}")
                time.sleep(5)
                return recreated_pods

            logging.warning(f"No '{microservice}' pods recreated yet. Retrying in {interval} seconds...")
            retries += 1
            time.sleep(interval)

        logging.error(f"Timeout reached. No '{microservice}' pods were recreated.")

    except subprocess.CalledProcessError as e:
        logging.error("An error occurred while executing kubectl commands.")
        logging.error(e.stderr)

def gen_log(pods, rate):
    global run

    i = 0
    log_files = []
    for pod in pods:
        log_file = f"{run}.{rate}.{i}.log"
        command = ["kubectl", "logs", pod, "-c", "istio-proxy"]

        with open(f"{log_file}", "w") as f:
            subprocess.run(command, check=True, text=True, stdout=f)
        
        i += 1
        log_files.append(log_file)

    return log_files

def construct_convex_hull(coefficients):
    def is_internal(l1, l2, l3):
        # Check if the (l1, l2) intersection is larger than the (l2, l3) intersection
        return (l1[0] - l2[0]) * (l3[1] - l2[1]) <= (l2[0] - l3[0]) * (l2[1] - l1[1])
    
    dimension = len(coefficients[0][1])
    result = [[] for _ in range(dimension)]
    line_segments_per_dims = [[(slope, intercepts[i]) for slope, intercepts in coefficients if i < len(intercepts)] for i in range(dimension)]
    for i, line_segments in enumerate(line_segments_per_dims):
        hull = []
        intersections = []

        for line in line_segments:
            # While the last two lines and the current line make the second-to-last line redundant, remove it
            while len(hull) >= 2 and is_internal(hull[-2], hull[-1], line):
                hull.pop()
                intersections.pop()
            if hull:
                s1, c1 = hull[-1]
                s2, c2 = line
                x_intersection = (c2 - c1) / (s1 - s2)
                intersections.append(x_intersection)
            hull.append(line)

        prev_x = float('-inf')
        for j in range(len(hull)):
            s, c = hull[j]
            x_end = intersections[j] if j < len(intersections) else float('inf')
            result[i].append((s, c, prev_x, x_end))
            prev_x = x_end
    
        return result

def generate_tikz_curve(rates, backlogs):
    # We skip the first (total) backlog
    dimension = len(backlogs) - 1
    line_segments = [[] for _ in range(dimension)]
    tikz_code = """
\\documentclass{article}
\\usepackage[active,tightpage]{preview}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\begin{document}
\\begin{preview}
\\begin{tikzpicture}
\\begin{axis}[
    axis lines=middle,
    xlabel={$t$},
    ylabel={Value},
    xmin=0, xmax=1,
    ymin=0, ymax=2000,
    domain=0:10,
    samples=200,
    legend pos=north west,
    grid=both,
]
"""

    # Add each function to the plot
    for i, slope in enumerate(rates):
        for j in range(dimension):
            tikz_code += f"\\addplot[blue, dashed] {{{slope}*x - {backlogs[j+1][i]}}};\n"
            line_segments[j].append((slope, backlogs[j+1][i]))
    
    for i in range(dimension):
        max_expression = "max(" + ", ".join([f"{slope}*x - {intercept}" for slope, intercept in line_segments[i]]) + ")"
        tikz_code += f"\\addplot[black, thick] {{{max_expression}}};\n"
    tikz_code += "\\end{axis}\n\\end{tikzpicture}\n\\end{preview}\n\\end{document}"
            
    return tikz_code

def plot_tikz_image(rates, backlogs, microservice, api):
    profile = f"{microservice}.{api}"
    tikz_code = generate_tikz_curve(rates, backlogs)
    with open(f"{profile}.tex", 'w') as file:
        file.write(tikz_code)

    try:
        subprocess.run(["pdflatex", f"{profile}.tex", "-output-directory=.."], check=True,
                       stdin=subprocess.PIPE,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        logging.info(f"Compilation successful. PDF generated for {profile}.")
        for tmp_file in [f"{profile}.aux", f"{profile}.log"]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during compilation: {e}")
    except FileNotFoundError:
        logging.error("pdflatex not found. Make sure it is installed and in your PATH.")

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

def init_experiment_path():
    global run

    if not os.path.exists(os.path.join(profiler_cwd, 'output')):
        os.mkdir(os.path.join(profiler_cwd, 'output'))
    os.chdir(os.path.join(profiler_cwd, 'output'))

    while os.path.exists(os.path.join(profiler_cwd, 'output', run)):
        trial = int(run) + 1
        run = f"{trial:03}"

    os.mkdir(os.path.join(profiler_cwd, 'output', run))
    os.chdir(os.path.join(profiler_cwd, 'output', run))

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
        plot(data, target, len(arrivals))

        if len(measured_timestamps) > 1:
            intervals = []
            for i in range(1, len(measured_timestamps)):
                interval = (measured_timestamps[i] - measured_timestamps[i-1]).total_seconds()
                intervals.append(interval)
            # print(np.average(intervals), np.std(intervals), 1/rate)
            return max_backlogs, np.average(intervals)
        logging.error("parse_loki_log() has no measured timestamps")

    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)

def is_internal(l1, l2, l3):
    # Check if the (l1, l2) intersection is larger than the (l2, l3) intersection
    # Note that the second elements (intercepts) are positive
    return (l1[0] - l2[0]) * (l2[1] - l3[1]) <= (l2[0] - l3[0]) * (l1[1] - l2[1])

def update_hulls(hulls, rates, backlogs, x_threshold):
    is_updated = False

    for i in range(len(backlogs)):
        lines = [(rates[j], backlogs[i][j]) for j in range(len(rates))]
        lines.sort(key=lambda x: (x[0], -x[1]))
        
        new_hull = []

        for line in lines:
            # While the last two lines and the current line make the second-to-last line redundant, remove it
            while len(new_hull) >= 2 and is_internal(new_hull[-2], new_hull[-1], line):
                new_hull.pop()
            if len(new_hull) >= 1:
                if (new_hull[-1][1] - line[1]) / (new_hull[-1][0] - line[0]) * 1000 < x_threshold:
                    new_hull.append(line)
            else:
                new_hull.append(line)

        if not new_hull == hulls[i]:
            # print(i, new_hull, hulls[i])
            hulls[i] = new_hull
            is_updated = True

    return is_updated

def find_max_segments(lines):
    # Sort lines by slope first (s), and then by intercept (-c) in descending order
    lines.sort(key=lambda x: (x[0], -x[1]))

    # List to store the selected lines forming the upper envelope
    hull = []
    intersections = []

    for line in lines:
        # While the last two lines and the current line make the second-to-last line redundant, remove it
        while len(hull) >= 2 and is_internal(hull[-2], hull[-1], line):
            hull.pop()
            intersections.pop()
        if hull:
            s1, c1 = hull[-1]
            s2, c2 = line
            x_intersection = (c1 - c2) / (s1 - s2)
            intersections.append(x_intersection)
        hull.append(line)

    result = []
    prev_x = float('-inf')
    for i in range(len(hull)):
        s, a = hull[i]
        x_end = intersections[i] if i < len(intersections) else float('inf')
        result.append((s, a, prev_x, x_end))
        prev_x = x_end
    
    return result

def plot_tikz_image_by_hull(segments_per_api, name):
    tikz_code = """
\\documentclass{article}
\\usepackage[active,tightpage]{preview}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\begin{document}
\\begin{preview}
\\begin{tikzpicture}
\\begin{axis}[
    axis lines=middle,
    xlabel={$t$},
    ylabel={Value},
    xmin=0, xmax=1,
    ymin=0, ymax=2000,
    domain=-10:10,
    samples=200,
    legend pos=north west,
    grid=both,
]
"""

    for segment in segments_per_api:
        for s, c, x_start, x_end in segment:
            if x_start == float('-inf'):
                x_start = -10
            if x_end == float('inf'):
                x_end = 10
            tikz_code += f"\\addplot[domain={x_start}:{x_end}, samples=2] {{{s}*x - {c}}};\n"

    tikz_code += "\\end{axis}\n\\end{tikzpicture}\n\\end{preview}\n\\end{document}"

    with open(f"{name}.tex", 'w') as file:
        file.write(tikz_code)

    try:
        subprocess.run(["pdflatex", f"{name}.tex"], check=True,
                       stdin=subprocess.PIPE,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        logging.info(f"Compilation successful. PDF generated for {name}.")
        for tmp_file in [f"{name}.aux", f"{name}.log"]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during compilation: {e}")
    except FileNotFoundError:
        logging.error("pdflatex not found. Make sure it is installed and in your PATH.")

if __name__ == "__main__":
    profiler_cwd = os.getcwd()
    run = f"000"
    init_experiment_path()

    parser = argparse.ArgumentParser()
    parser.add_argument('app')
    parser.add_argument('--duration', '-d', default=2000)
    parser.add_argument('--threshold', default=1000)
    # parser.add_argument('--app', '-a')
    parser.add_argument('--microservice', '-m')
    parser.add_argument('--api', '-i')
    parser.add_argument('--log', '-l', default="warning")
    args = parser.parse_args()

    duration = int(args.duration)
    threshold = int(args.threshold)

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
    
    with open(os.path.join(profiler_cwd, f'{args.app}/spec.json'), 'r') as spec_file:
        spec = json.load(spec_file)
    
    profiles = []
    if args.microservice != None and args.api != None:
        targets = [(args.microservice, args.api)]
    else:
        targets = get_profiling_targets(spec)
    print(f"{run} {args.app}", " ".join([f"{target[0]}:{target[1]}" for target in targets]))

    for i, target in enumerate(targets):
        microservice = target[0]
        api = target[1]
        print(f"{microservice}:{api} ({i+1}/{len(targets)})")

        spans, dependencies = get_spans_dependencies(spec, microservice, api)

        arrivals = []
        departures = []
        arrivals.append(('I', 'REQ', 'no-progress'))
        for span in spans:
            arrivals.append(('O', 'RES', span["api"]))
            departures.append(('O', 'REQ', span["api"]))
        departures.append(('I', 'RES', 'no-progress'))

        clear_envoyfilter()
        apply_envoyfilter(args.app, microservice, api)

        rates = []
        backlogs = [[] for _ in range(len(spans) + 2)]
        hulls = [[] for _ in range(len(spans) + 2)]

        exponential_growth = True
        current_rate = 25
        
        # As it is multiplied by 2, the first step size is 25
        step_size = 12.5

        enable_lua_logging(microservice)
        
        os.mkdir(f"{microservice}.{api}")
        os.chdir(f"{microservice}.{api}")

        while True:
            logging.info("Checking all pods...")
            max_retries = 20
            retries = 0
            while retries < max_retries:
                cmd = ["kubectl", "get", "pods", "-A"]
                result = subprocess.run(cmd, check=True, text=True, capture_output=True)
                output_lines = result.stdout.splitlines()
                not_running = []
                for line in output_lines[1:]:
                    tokens = line.split()
                    if len(tokens) > 3 and (tokens[1].startswith(microservice) or any(tokens[1].startswith(dependency) for dependency in dependencies)):
                        if not tokens[3] == "Running":
                            not_running.append(tokens[1])
                        
                if len(not_running) > 0:
                    logging.warning(f"Pod {not_running} are not running")
                    retries += 1
                    time.sleep(10)
                else:
                    break

            logging.info("Start scanning.")
            range_from = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
            responses = run_constant_interval(current_rate, duration, args.app, microservice, api)
            range_to = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

            if not responses:
                logging.error(f"No responses from a constant_interval scan: {current_rate}")
                os.chdir(f"..")
                break

            results = {"num_responses": int(responses[0]),
                    "num_failures": int(responses[1]),
                    "mean": float(responses[2]),
                    "p50": float(responses[3]),
                    "p90": float(responses[4]),
                    "p99": float(responses[5]),
                    "p99.9": float(responses[6])
                    }
            
            results["max_backlogs"], results["mean_interval"] = parse_loki_log(current_rate, duration, microservice, api, arrivals, departures, range_from, range_to)
            # logs = gen_log(pods, current_rate)
            # parse_log(arrivals, departures, results, logs)

            print(f'  {current_rate} {results["num_responses"]} {results["num_failures"]} {results["mean"]:.3f}',
                f'{results["p50"]:.3f} {results["p90"]:.3f} {results["p99"]:.3f} {results["p99.9"]:.3f} {results["max_backlogs"]}')
        
            break_signal = False
            # We do not update this run
            fail_ratio = results["num_failures"] / results["num_responses"]
            interval_ratio = abs(current_rate * results["mean_interval"] - 1)
            if fail_ratio < 0.04 and interval_ratio < 0.2 and results["p99.9"] * 1000 < threshold:
                rates.append(current_rate)
                for j, max_backlog in enumerate(results["max_backlogs"]):
                    backlogs[j].append(results["max_backlogs"][j])

                updated = update_hulls(hulls, rates, backlogs, threshold)
                if not updated:
                    logging.debug(f'ex: {exponential_growth} updated: {updated}')
                    break_signal = True
            else:
                logging.debug(f'ex: {exponential_growth} fail: {fail_ratio:.2f} interval: {interval_ratio:.2f} p99.9: {results["p99.9"]:.3f}')
                break_signal = True

                # restart the pod
                if fail_ratio >= 0.04:
                    cmd = ["kubectl", "get", "pods", "-A"]
                    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
                    output_lines = result.stdout.splitlines()
                    pods_to_restart = []
                    for line in output_lines[1:]:
                        tokens = line.split()
                        if len(tokens) > 3 and (tokens[1].startswith(microservice) or any(tokens[1].startswith(dependency) for dependency in dependencies)):
                            pods_to_restart.append(tokens[1])
                            
                    if len(pods_to_restart) > 0:
                        logging.warning(f"Restart Pods {pods_to_restart}")
                        for pod in pods_to_restart:
                            cmd = ["kubectl", "delete", "pod", pod]
                            result = subprocess.run(cmd, check=True, text=True, capture_output=True)

                        # enable lua logs at the new pod
                        enable_lua_logging(microservice)

            if break_signal:
                if exponential_growth:
                    exponential_growth = False
                    current_rate = int(current_rate / 2)
                    step_size = int(current_rate * 0.2)
                    current_rate += step_size
                else:
                    os.chdir(f"..")
                    break
            else:
                if exponential_growth:
                    step_size *= 2
                current_rate += int(step_size)

            time.sleep(results["p99.9"])

        plot_tikz_image(rates, backlogs, microservice, api)

        profiles.append({"microservice": microservice, "api": api, "rates": rates, "backlogs": backlogs})
        clear_envoyfilter()

    json_profiles = {}

    for profile in profiles:
        logging.debug(profile["microservice"], profile["api"])

        rates = profile["rates"]
        backlogs = profile["backlogs"]
        hulls = [[(rate, backlogs[i][j]) for j, rate in enumerate(rates)] for i in range(len(backlogs))]
        
        segments_per_api = []
        for i, hull in enumerate(hulls):
            segments = find_max_segments(hull)
            for s, c, x_start, x_end in segments:
                logging.debug(f"{i}: {s}x - {c}, Range: [{x_start}, {x_end}]")
            segments_per_api.append(segments)
        
        print(segments_per_api)
        plot_tikz_image_by_hull(segments_per_api[1:], f'{profile["microservice"]}.{profile["api"]}.hull')

        segments_per_api_sanitized = []
        for segments in segments_per_api:
            segments_sanitized = []
            for s, c, x_start, x_end in segments:
                if x_start == float('-inf'):
                    x_start = -1e308
                if x_end == float('inf'):
                    x_end = 1e308
                segments_sanitized.append((s, c, x_start, x_end))
            segments_per_api_sanitized.append(segments_sanitized)
        profile["segments_per_api"] = segments_per_api_sanitized

    # Write the profile in a json file
    profile_output = f"{args.app}.json"
    with open(profile_output, "w") as file:
        json.dump(profiles, file, indent=4)
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

def get_spans_dependencies(target_app, target_microservice, target_api):
    with open(f'../../{target_app}/{target_app}-spec.json', 'r') as file:
        data = json.load(file)
        
    if not target_app == data.get("app"):
        logging.error("application ids are different.")

    print(target_app, target_microservice, target_api)
    microservices = data.get("microservices", [])
    for microservice in microservices:
        if microservice.get("id") == target_microservice:
            apis = microservice.get("apis", [])
            for api in apis:  
                if api.get("id") == target_api:
                    return api.get("spans"), api.get("dependencies")

    logging.error("no such api")

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
    for i, condition in enumerate(arrivals):
        if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
            backlogs[0] += 1
            backlogs[i+1] += 1
            data.append((parsed_line['second_timestamp'], tuple(backlogs)))
            if backlogs[i+1] > max_backlogs[i+1]:
                max_backlogs[i+1] = backlogs[i+1]
            if backlogs[0] > max_backlogs[0]:
                max_backlogs[0] = backlogs[0]
            
    for i, condition in enumerate(departures):
        if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
            backlogs[0] -= 1
            backlogs[i+1] -= 1
            data.append((parsed_line['second_timestamp'], tuple(backlogs)))

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
    command = [
        "./constant-interval-hr.py",
        "--rps", str(rps),
        "--duration", str(duration),
        "-a", app,
        "-m", microservice,
        "-i", api
    ]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=os.path.join("../..", app))
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

def enable_lua_logs(microservice):
    try:
        logging.info("Fetching all pods...")
        cmd = ["kubectl", "get", "pods", "-A"]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        output_lines = result.stdout.splitlines()
        for line in output_lines[1:]:
            parts = line.split()
            if len(parts) > 3 and parts[1].startswith(microservice):
                max_retries = 5
                retries = 0
                while True:
                    if parts[3] == "Running":
                        cmd = ["istioctl", "proxy-config", "log", f"{parts[1]}.default", "--level", "lua=info"]
                        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
                        break
                    else:
                        retries += 1
                        time.sleep(1)

    except subprocess.CalledProcessError as e:
        logging.error("An error occurred while executing kubectl commands.")
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
    ymin=-15, ymax=500,
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

def plog_tikz_image(rates, backlogs):
    global run
    file_path = f"{run}.curve.tex"
    tikz_code = generate_tikz_curve(rates, backlogs)
    with open(file_path, 'w') as file:
        file.write(tikz_code)

    try:
        subprocess.run(["pdflatex", file_path], check=True,
                       stdin=subprocess.PIPE,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        logging.info(f"Compilation successful. PDF generated for {file_path}.")
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
    with open(f'../../{target_app}/{target_app}-spec.json', 'r') as file:
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

    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, 'output')):
        os.mkdir(os.path.join(cwd, 'output'))
    os.chdir(os.path.join(cwd, 'output'))

    while os.path.exists(os.path.join(cwd, 'output', run)):
        (remained, last) = run.rsplit("-", 1)
        trial = int(last) + 1
        run = f"{remained}-{trial:03}"

    os.mkdir(os.path.join(cwd, 'output', run))
    os.chdir(os.path.join(cwd, 'output', run))

def parse_loki_log(rate, microservice, arrivals, departures, range_from, range_to):
    global run

    data = []
    backlogs = [0] * (len(arrivals) + 1)
    max_backlogs = [0] * (len(arrivals) + 1)
    try:        
        logging.info(f'Parse loki log')
        cmd = ["logcli", "query", f'{{app="{microservice}"}}',
               f'--from={range_from}', f'--to={range_to}', '--limit=0', '--forward']
        with open(f"{run}.{microservice}.{rate}.log", "w") as log_file:
            result = subprocess.run(cmd, text=True, capture_output=True, check=True, cwd="../../..")
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

        with open(f"{run}.{rate}.log", "r") as log_file:
            for line in log_file:
                match = re.match(pattern, line)
                if match:
                    calculate_backlog(arrivals, departures, data, backlogs, max_backlogs, match.groupdict())
                    
        plot(data, f"{run}.{microservice}.{rate}", len(arrivals))
        return max_backlogs

    except subprocess.CalledProcessError as e:
        logging.error(e.stderr)

if __name__ == "__main__":
    run = f"run-000"
    init_experiment_path()

    parser = argparse.ArgumentParser()
    parser.add_argument('--app', '-a')
    parser.add_argument('--microservice', '-m')
    parser.add_argument('--api', '-i')
    parser.add_argument('--log', '-l', default="warning")

    global args
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
    
    spans, dependencies = get_spans_dependencies(args.app, args.microservice, args.api)

    arrivals = []
    departures = []

    arrivals.append(('I', 'REQ', 'no-progress'))
    for span in spans:
        arrivals.append(('O', 'RES', span["api"]))
        departures.append(('O', 'REQ', span["api"]))
    departures.append(('I', 'RES', 'no-progress'))

    clear_envoyfilter()
    apply_envoyfilter(args.app, args.microservice, args.api)

    rates = []
    backlogs = [[] for _ in range(len(spans) + 2)]

    exponential_growth = True
    current_rate = 50
    step_size = 100
    threshold = 1

    # pods = init_pods(args.microservice, dependencies)
    enable_lua_logs(args.microservice)

    while True:
        range_from = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        responses = run_constant_interval(current_rate, 3, args.app, args.microservice, args.api)
        range_to = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

        if not responses:
            logging.error(f"No responses from a constant_interval scan: {current_rate}")
            break

        results = {"num_responses": int(responses[0]),
                   "num_failures": int(responses[1]),
                   "mean": float(responses[2]),
                   "p50": float(responses[3]),
                   "p90": float(responses[4]),
                   "p99": float(responses[5]),
                   "p99.9": float(responses[6])
                }
        
        results["max_backlogs"] = parse_loki_log(current_rate, args.microservice, arrivals, departures, range_from, range_to)
        
        # logs = gen_log(pods, current_rate)
        # parse_log(arrivals, departures, results, logs)

        print(f'{current_rate} {results["num_responses"]} {results["num_failures"]} {results["mean"]:.3f}',
              f'{results["p50"]:.3f} {results["p90"]:.3f} {results["p99"]:.3f} {results["p99.9"]:.3f} {results["max_backlogs"]}')
        rates.append(current_rate)
        for i, max_backlog in enumerate(results["max_backlogs"]):
            backlogs[i].append(results["max_backlogs"][i])

        # if current_rate <= 200:
        #     current_rate += 100
        #     continue
        # else:
        #     break

        # If the p99 response time is larger than the threshold or the failure ratio is larger than 0.2,
        # it stops the exponential growth, and backoffs to the last rate + step size
        if results["p99"] >= threshold or results["num_failures"] / results["num_responses"] > 0.2:
            if exponential_growth:
                exponential_growth = False
                step_size = int(current_rate / 2 * 0.2)
                current_rate = int(current_rate / 2 + step_size)
            else:
                break
        else:
            if exponential_growth:
                current_rate *= 2
            else:
                current_rate += step_size
        
        # Sleep sufficiently (we use the p99.9 response time of the last run) to absorb the queued requests
        time.sleep(results["p99.9"])

    plog_tikz_image(rates, backlogs)
    clear_envoyfilter()
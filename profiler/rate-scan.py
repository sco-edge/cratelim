#!/bin/python3
import re
import time
import json
import logging
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as mp
from datetime import datetime

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

def is_arrival(parsed_line):
    arrival_conditions = [
        ('I', 'REQ', 'no-progress'),
        ('O', 'RES', 'near-by'),
        ('O', 'RES', 'check-availability'),
        ('O', 'RES', 'get-profiles'),
    ]
    return (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) in arrival_conditions

def calculate_backlog(data, backlogs, max_backlogs, parsed_line):
    arrival_conditions = [
        ('I', 'REQ', 'no-progress'),
        ('O', 'RES', 'near-by'),
        ('O', 'RES', 'check-availability'),
        ('O', 'RES', 'get-profiles'),
    ]
    departure_conditions = [
        ('O', 'REQ', 'near-by'),
        ('O', 'REQ', 'check-availability'),
        ('O', 'REQ', 'get-profiles'),
        ('I', 'RES', 'no-progress'),
    ]

    for i, condition in enumerate(arrival_conditions):
        if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
            backlogs[0] += 1
            backlogs[i+1] += 1
            data.append((parsed_line['timestamp'], tuple(backlogs)))
            if backlogs[i+1] > max_backlogs[i+1]:
                max_backlogs[i+1] = backlogs[i+1]
            if backlogs[0] > max_backlogs[0]:
                max_backlogs[0] = backlogs[0]
            
    for i, condition in enumerate(departure_conditions):
        if (parsed_line['direction'], parsed_line['type'], parsed_line['progress_context']) == condition:
            backlogs[0] -= 1
            backlogs[i+1] -= 1
            data.append((parsed_line['timestamp'], tuple(backlogs)))

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

def plot(data, file_name):
    timestamps = [datetime.fromisoformat(item[0][:23]) for item in data]
    values = [item[1] for item in data]

    col1 = [v[0] for v in values]
    col2 = [v[1] for v in values]
    col3 = [v[2] for v in values]
    col4 = [v[3] for v in values]
    col5 = [v[4] for v in values]

    mp.figure(figsize=(12, 6))
    mp.plot(timestamps, col1, label='Total')
    mp.plot(timestamps, col2, label='1')
    mp.plot(timestamps, col3, label='2')
    mp.plot(timestamps, col4, label='3')
    mp.plot(timestamps, col5, label='4')

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

    output_path = f"output/{file_name}.png"
    mp.savefig(output_path, dpi=300, bbox_inches='tight')

def parse_log(results, logs):
    for log in logs:
        # max_backlog = 0
        # backlog = 0
        data = []
        backlogs = [0] * 5
        max_backlogs = [0] * 5

        with open(f"output/{log}", 'r') as file:
            for line in file:
                parsed_line = parse_log_line(line.strip())
                if parsed_line:
                    # if is_arrival(parsed_line):
                    #     backlog += 1
                    # else:
                    #     backlog -= 1
                    # if backlog > max_backlog:
                    #     max_backlog = backlog
                    calculate_backlog(data, backlogs, max_backlogs, parsed_line)

        # for datum in data:
        #     print(datum)
        plot(data, log)
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
        result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=app)
        logging.info("Load script execution completed successfully.")
        logging.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error("Load script execution failed:")
        logging.error(e.stderr)

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

def init_pods():
    try:
        # Step 1: Get all pods and delete 'frontend' pods
        logging.info("Fetching all pods...")
        result = subprocess.run(
            ["kubectl", "get", "pods", "-A"],
            check=True,
            text=True,
            capture_output=True
        )

        # Parse output to find 'frontend' pods
        logging.info("Parsing pod information...")
        output_lines = result.stdout.splitlines()
        frontend_pods = []
        for line in output_lines[1:]:  # Skip the header
            parts = line.split()
            if len(parts) > 1 and parts[1].startswith("frontend"):
                namespace = parts[0]  # Namespace
                pod_name = parts[1]  # Pod name
                frontend_pods.append((namespace, pod_name))

        if not frontend_pods:
            logging.error("No 'frontend' pods found.")
            return

        # Delete 'frontend' pods
        logging.info(f"Found {len(frontend_pods)} 'frontend' pods: {[pod for _, pod in frontend_pods]}")
        for namespace, pod_name in frontend_pods:
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

        # Step 2: Check for recreated pods in a loop
        max_retries = 10  # Maximum number of checks
        interval = 2  # Seconds between checks
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
            for line in output_lines[1:]:  # Skip the header
                parts = line.split()
                if (
                    len(parts) > 3 and  # Ensure there are enough parts in the line
                    parts[1].startswith("frontend") and  # Pod name starts with 'frontend'
                    parts[3] == "Running"  # STATUS column is 'Running'
                ):
                    result = subprocess.run(
                        ["istioctl", "proxy-config", "log", f"{parts[1]}.default", "--level", "lua=info"],
                        check=True,
                        text=True,
                        capture_output=True
                    )
                    recreated_pods.append(parts[1])
                    return recreated_pods

            if recreated_pods:
                logging.info(f"Recreated 'frontend' pods: {recreated_pods}")
                return  # Exit the function when pods are recreated

            logging.warning(f"No 'frontend' pods recreated yet. Retrying in {interval} seconds...")
            retries += 1
            time.sleep(interval)

        logging.error("Timeout reached. No 'frontend' pods were recreated.")

    except subprocess.CalledProcessError as e:
        logging.error("An error occurred while executing kubectl commands.")
        logging.error(e.stderr)

def gen_log(pods, run):
    i = 0
    log_files = []
    for pod in pods:
        log_file = f"tmp.{run}.{i}.log"
        command = ["kubectl", "logs", pod, "-c", "istio-proxy"]

        with open(f"output/{log_file}", "w") as f:
            subprocess.run(command, check=True, text=True, stdout=f)
        
        i += 1
        log_files.append(log_file)

    return log_files

def generate_tikz_curve(coefficients):
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
    for i, (slope, intercept) in enumerate(coefficients):
        color = ["blue", "red", "green", "orange", "purple", "cyan"]
        tikz_code += f"\\addplot[{color[i % len(color)]}, dashed, thick] {{{slope}*x - {intercept}}};\n"
    
    max_expression = "max(" + ", ".join([f"{slope}*x - {intercept}" for slope, intercept in coefficients]) + ")"
    tikz_code += f"\\addplot[black, thick] {{{max_expression}}};\n"
    tikz_code += "\\end{axis}\n\\end{tikzpicture}\n\\end{preview}\n\\end{document}"
            
    return tikz_code

def plog_tikz_image(coefficients):
    file_path = "output/curve.tex"
    tikz_code = generate_tikz_curve(coefficients)
    with open(file_path, 'w') as file:
        file.write(tikz_code)

    try:
        subprocess.run(["pdflatex", "-output-directory=output", file_path], check=True)
        print(f"Compilation successful. PDF generated for {file_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e}")
    except FileNotFoundError:
        print("pdflatex not found. Make sure it is installed and in your PATH.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default="0.log")
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
    
    coeffs = []
    rates = [100, 200]
    for rate in rates:
        for run in range(1):
            pods = init_pods()
            run_constant_interval(rate, 3, "hr", "search", "nearby")
            # run_wrk2(rate)
            results = {}
            # results = run_locust(rate)
            logs = gen_log(pods, rate)
            parse_log(results, logs)

            # print(f'rate: {rate} avg: {results["average"]:.3f} p50: {results["p50"]:.3f} '
            #       f'p90: {results["p90"]:.3f} p99: {results["p99"]:.3f} p99.9: {results["p99.9"]:.3f} '
            #       f'max_backlogs: {results["max_backlogs"]}')
            print(f'rate: {rate} max_backlogs: {results["max_backlogs"]}')
            coeffs.append((rate, results["max_backlogs"][1]))

    plog_tikz_image(coeffs)
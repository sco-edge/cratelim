#!/usr/bin/python3
import sys
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


parser = argparse.ArgumentParser()
parser.add_argument('log_path')
args = parser.parse_args()

timestamps = []

# Server
try:
    with open(f"{args.log_path}.server.log", "r") as file:
        for line in file:
            timestamps.append(datetime.datetime.fromisoformat(line.strip()))
except Exception as e:
    print(f"Cannot open {args.log_path}.server.log: {e}")

start_time = timestamps[0]
time_deltas = [(ts - start_time).total_seconds() * 1000 for ts in timestamps]

# ms
bin_size_small = 100
bin_size_large = 1000
max_time = max(time_deltas)

bins_large = np.arange(0, max_time + bin_size_large, bin_size_large)
bins_small = np.arange(0, max_time + bin_size_small, bin_size_small)

hist_large, _ = np.histogram(time_deltas, bins=bins_large)
hist_small, _ = np.histogram(time_deltas, bins=bins_small)

plt.figure(figsize=(10, 5))
plt.bar(bins_large[:-1], hist_large, width=1000, label="1s Bins", color="#41b6c4")
plt.bar(bins_small[:-1], hist_small, width=100, label="100ms Bins", color="#253494")
plt.xlabel("Time (ms since start, binned by 10ms)")
plt.ylabel("Request Count")
plt.title("Requests per 10ms Time Bin")
plt.grid(axis='y', linestyle='--')

output_path = f"{sys.argv[1]}.server.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

# Client
x_values = []
y_values = []
colors = []
try:
    with open(f"{args.log_path}.client.log", "r") as file:
        for line in file:
            tokens = line.split()
            ts = (datetime.datetime.fromisoformat(tokens[0]) - start_time).total_seconds()
            latency = float(tokens[1])
            status = int(tokens[2])

            x_values.append(ts)
            y_values.append(latency)
            colors.append("#253494" if status == 200 else "red")
except Exception as e:
    print(f"Cannot open {args.log_path}.client.log: {e}")

print(f"{len(y_values)} {np.mean(y_values):.3f} {np.percentile(y_values, 50):.3f}",
      f"{np.percentile(y_values, 95):.3f}", f"{np.percentile(y_values, 99):.3f}",
      f"{np.percentile(y_values, 99.9):.3f}") # ms

plt.figure(figsize=(10, 5))
plt.scatter(x_values, y_values, c=colors, label="Latency", s=10)
plt.ylabel("Latency (ms)")
plt.title("Latency Over Time (Black: 200, Red: Other)")
plt.xticks(rotation=45)
plt.grid(True)

output_path = f"{sys.argv[1]}.client.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
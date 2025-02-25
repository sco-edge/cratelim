#!/bin/python3

import matplotlib.pyplot as mp

trace = "bursty-short"
with open(f"traces/{trace}.txt", "r") as file:
    data = [int(line.strip()) for line in file]

mp.figure()
mp.plot(data, linestyle='-')
mp.grid(True)
mp.tight_layout()

mp.savefig(f"{trace}.png", dpi=300, bbox_inches='tight')
mp.close()
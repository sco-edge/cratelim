#!/bin/python3
import json
import math
import os
import subprocess
import numpy as np
import logging
import argparse
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

class Problem:
    def __init__(self, external_apis, apis, chains, index_segments, y_intersections, slopes,
                 intercepts, weights, deadlines):
        self.A_E = external_apis
        self.A = apis
        self.C = chains
        self.L_j = index_segments
        self.b_j_k = y_intersections
        self.s_j_k = slopes
        self.c_j_k = intercepts
        self.w = weights
        self.D = deadlines

        print(y_intersections)

    def solve(self):
        problem = LpProblem(name="milp-optimization", sense=LpMaximize)

        M = 1e6
        epsilon = 1e-6

        # Variables and constraint (10)
        b = LpVariable.dicts("b", [(i, j) for i in self.A_E for j in self.A], lowBound=0)
        B = LpVariable.dicts("B", self.A, lowBound=0)
        z = LpVariable.dicts("z", self.A_E, lowBound=0)
        v = LpVariable.dicts("v", [(j, k) for j in self.A for k in self.L_j[j]], lowBound=0)

        # constraints (21, 22)
        delta_b = LpVariable.dicts("delta",
                                   [(i, j, k) for i in self.A_E for j in self.A
                                    for k in self.L_j[j]], cat=LpBinary)
        lambda_b = LpVariable.dicts("lambda",
                                    [(j, k) for j in self.A for k in self.L_j[j]], cat=LpBinary)

        # objective function (6)
        problem += lpSum(self.w[i] * z[i] for i in self.A_E)

        # # constraint (7)
        # for i in self.A_E:
        #     for j in self.C[i]:
        #         for k in self.L_j[j]:
        #             problem += z[i] <= self.s_j_k[j][k] * delta_b[i, j, k]

        # constraint (7)
        for i in self.A_E:
            problem += z[i] <= lpSum(self.s_j_k[j][k] * delta_b[i, j, k] for j in self.C[i] for k in self.L_j[j])

        # constraints (8, 9)
        for i in self.A_E:
            for j in self.C[i]:
                for k in self.L_j[j]:
                    problem += self.b_j_k[j][k] <= b[i, j] + M * (1 - delta_b[i, j, k])
                    problem += b[i, j] <= self.b_j_k[j][k+1] - epsilon + M * (1 - delta_b[i, j, k])

        # constraint (11)
        for i in self.A_E:
            problem += lpSum((v[j, k] + lambda_b[j, k] * self.c_j_k[j][k]) / self.s_j_k[j][k]
                             for j in self.C[i] for k in self.L_j[j]) <= self.D[i]

        # constraints (12, 13, 14, 15)
        for j in self.A:
            for k in self.L_j[j]:
                problem += v[j, k] <= M * lambda_b[j, k]
                problem += v[j, k] >= -M * lambda_b[j, k]
                problem += v[j, k] <= B[j] + M * (1 - lambda_b[j, k])
                problem += v[j, k] >= B[j] - M * (1 - lambda_b[j, k])

        # constraints (16, 17)
        for j in self.A:
            for k in self.L_j[j]:
                problem += self.b_j_k[j][k] <= B[j] + M * (1 - lambda_b[j, k])
                problem += B[j] <= self.b_j_k[j][k + 1] - epsilon + M * (1 - lambda_b[j, k])

        # constraint (18)
        for j in self.A:
            problem += lpSum(b[i, j] for i in self.A_E) == B[j]

        # constraint (19)
        for i in self.A_E:
            for j in self.C[i]:
                problem += lpSum(delta_b[i, j, k] for k in self.L_j[j]) == 1

        # constraint (20)
        for j in self.A:
            problem += lpSum(lambda_b[j, k] for k in self.L_j[j]) == 1

        problem.solve()

        print(f"Objective Value = {problem.objective.value()}")
        for i in self.A_E:
            print(f"z[i={i}] = {z[i].varValue}")
        for i in self.A_E:
            for j in self.A:
                print(f"b[i={i},j={j}] = {b[i, j].varValue}")
        for j in self.A:
            print(f"B[j={j}] = {B[j].varValue}")
        for j in self.A:
            for k in self.L_j[j]:
                if v[j, k].varValue == 0:
                    continue
                print(f"v[j={j},k={k}] = {v[j, k].varValue} c[j={j},k={k}] = {self.c_j_k[j][k]}")
        for i in self.A_E:
            # for j in self.A:
            for j in self.C[i]:
                for k in self.L_j[j]:
                    if delta_b[i, j, k].varValue == 0:
                        continue
                    print(f"delta[i={i},j={j},k={k}] = {delta_b[i, j, k].varValue}")
        for j in self.A:
            for k in self.L_j[j]:
                if lambda_b[j, k].varValue == 0:
                    continue
                print(f"lambda[j={j},k={k}] = {lambda_b[j, k].varValue}")
        for i in self.A_E:
            budgets = {}
            for j in self.C[i]:
                budget = 0
                for k in self.L_j[j]:
                    budget += (v[j, k].varValue + lambda_b[j, k].varValue * self.c_j_k[j][k]) / self.s_j_k[j][k]
                budgets[j] = budget

            sum = 0
            for budget in budgets.values():
                sum += budget
            print(f"budgets[i={i}] = {budgets}, {sum}")
        return 

def parse_profiles_with_subsidiaries(file_name, latency_slo):
    # hr_call_chains = {
    #     ("frontend", "frontend-hotels"):
    #         [("frontend", "frontend-hotels"),
    #          ("search", "search-near-by"),
    #          ("geo", "geo-near-by"),
    #          ("rate", "rate-get-rates"),
    #          ("reservation", "reservation-check-availability"),
    #          ("profile", "profile-get-profiles")],
    #     ("frontend", "frontend-recommendations"):
    #         [("frontend", "frontend-recommendations"),
    #          ("recommendation", "recommendation-get-recommendations"),
    #          ("profile", "profile-get-profiles")],
    #     ("frontend", "frontend-user"):
    #         [("frontend", "frontend-user"),
    #          ("user", "user-check-user")],
    #     ("frontend", "frontend-reservation"):
    #         [("frontend", "frontend-reservation"),
    #          ("user", "user-check-user"),
    #          ("reservation", "reservation-make-reservation")]
    # }
    hr_call_chains = {
        ("frontend", "frontend-hotels"):
            [("geo", "geo-near-by")]
    }

    with open(file_name, "r") as file:
        profiles = json.load(file)

        # A_E and A
        external_apis = []
        metadata_external_apis = {}
        apis = []
        metadata_apis = {}
        metadata_apis_inverse = {}

        # b_j_k, s_j_k, c_j_k, L_j
        ys = {}
        slopes = {}
        intercepts = {}
        index_segments = {}
        
        sanitized_ys = {}
        sanitized_slopes = {}
        sanitized_intercepts = {}
        sanitized_index_segments = {}
        
        for j, profile in enumerate(profiles):
            print(f'{profile["microservice"]}-{profile["api"]}')
            if (profile["microservice"], profile["api"]) in hr_call_chains:
                external_apis.append(j)
                metadata_external_apis[f'{profile["microservice"]}-{profile["api"]}'] = j

            apis.append(j)
            metadata_apis[f'{profile["microservice"]}-{profile["api"]}'] = j
            metadata_apis_inverse[j] = f'{profile["microservice"]}-{profile["api"]}'

            y_full = {}
            slopes_full = {}
            intercepts_full = {}
            for i, segments in enumerate(profile["segments_per_api"][1:]):
                y_full[i] = []
                slopes_full[i] = []
                intercepts_full[i] = []
                for segment in segments:
                    # Skip the segments on the left of y-axis
                    if segment[3] <= 0:
                        continue

                    # If the segment START's y-value is negative and END's y-value is positive, it
                    # is the first meaningful segment. Otherwise, (1) both START and END are less
                    # than zero or (2) both START and END are larger than zero
                    if segment[0] * segment[2] - segment[1] < 0:
                        if segment[0] * segment[3] - segment[1] > 0:
                            y_full[i].append(0)
                        else:
                            # Both are less than zero. Skip it.
                            continue
                    else:
                        if math.isinf(segment[0] * segment[2] - segment[1]):
                            y_full[i].append(1e308)
                        else:
                            y_full[i].append(segment[0] * segment[2] - segment[1])

                    slopes_full[i].append(segment[0])
                    intercepts_full[i].append(segment[1])

                    # If the segment END infinite, break
                    if segment[3] == float('inf'):
                        # if math.isinf(segment[0] * latency_slo - segment[1]):
                        #     y_full[i].append(1e308)
                        # else:
                        #     y_full[i].append(segment[0] * latency_slo - segment[1])
                        break

            print(f'{profile["microservice"]}-{profile["api"]} y: {y_full}')
            print(f'{profile["microservice"]}-{profile["api"]} s: {slopes_full}')
            print(f'{profile["microservice"]}-{profile["api"]} c: {intercepts_full}')

            # Gather all breakpoints from every span
            y_for_ms = sorted(set(value for values in y_full.values() for value in values))
            slopes_for_ms = {}
            intercepts_for_ms = {}
            for k, bp in enumerate(y_for_ms):
                # if k == len(y_for_ms) - 1:
                #     break

                segments_to_add = [0 for _ in range(len(y_full))]
                for i in range(len(y_full)):
                    segment_index = segments_to_add[i]
                    while True:
                        if segment_index + 1 >= len(y_full[i]):
                            break

                        if bp >= y_full[i][segment_index + 1]:
                            segment_index += 1
                        else:
                            break
                    segments_to_add[i] = segment_index
                    # for k, insertection in enumerate(y_full[i]):
                    #     if bp < insertection:
                    #         segments_to_add.append(k-1)
                    #         break
                print(f"bp: {bp} sta:", segments_to_add)

                if len(segments_to_add) == 0:
                    continue

                slope_inverse_sum = 0
                frac_intercept_sum = 0
                for i, index in enumerate(segments_to_add):
                    if index >= len(slopes_full[i]) and index >= len(intercepts_full[i]):
                        slope_inverse_sum += 1 / slopes_full[i][index-1]
                        frac_intercept_sum += intercepts_full[i][index-1] / slopes_full[i][index-1]
                    else:
                        slope_inverse_sum += 1 / slopes_full[i][index]
                        frac_intercept_sum += intercepts_full[i][index] / slopes_full[i][index]
                slopes_for_ms[k] = 1 / slope_inverse_sum
                intercepts_for_ms[k] = slopes_for_ms[k] * frac_intercept_sum
            y_for_ms.append(1e308)

            print(f'{profile["microservice"]}-{profile["api"]} bp: {y_for_ms}')
            print(f'{profile["microservice"]}-{profile["api"]} ss: {slopes_for_ms}')
            print(f'{profile["microservice"]}-{profile["api"]} cs: {intercepts_for_ms}')

            ys[j] = y_for_ms
            slopes[j] = slopes_for_ms
            intercepts[j] = intercepts_for_ms
            index_segments[j] = range(len(y_for_ms) - 1)

            # while True:
            #     spans_to_add = []
            #     for i, segments in enumerate(profile["segments_per_api"][1:]):

            # y = [segments[0][2] for segments in profile["segments_per_api"][1:]]
            # print(breakpoints)            

            # for i in range(len(profile["segments_per_api"]) - 1):
            #     y_intersections[j] = []
            #     slopes[j] = []
            #     intercepts[j] = []

        chains = {}
        for m1, a1 in hr_call_chains:
            external_api = f"{m1}-{a1}"
            chain = hr_call_chains[(m1, a1)]

            external_api_index = metadata_external_apis[external_api]
            chain_indices = []
            for m2, a2 in chain:
                chain_indices.append(metadata_apis[f"{m2}-{a2}"])
            chains[external_api_index] = chain_indices

        print("external_apis:", external_apis)
        print("metadata_external_apis:", metadata_external_apis)
        print("apis:", apis)
        print("metadata_apis:", metadata_apis)
        print("chains:", chains)
        for j in range(len(index_segments)):
            print(f"{metadata_apis_inverse[j]} {j}  y: {ys[j]} s: {slopes[j]} c: {intercepts[j]} L: {index_segments[j]}")
        
        for name in metadata_apis:
            j = metadata_apis[name]
            xs = []
            for k in index_segments[j]:
                x = (ys[j][k] + intercepts[j][k]) / slopes[j][k]
                # print(name, j, k, x)
                if x >= latency_slo:
                    break
                xs.append(x)
            
            x_start = []
            x_end = []
            for k, x in enumerate(xs):
                x_start.append(x)
                if k == 0:
                    continue
                x_end.append(x)
            x_end.append(latency_slo)
            # print(x_start, x_end, slopes[j], intercepts[j], index_segments[j])
            plot_tikz_image_by_hull(x_start, x_end, slopes[j], intercepts[j], index_segments[j], f"int.{name}", latency_slo)

            print(j, name, xs, ys[j], slopes[j], intercepts[j], index_segments[j])
            
            sanitized_y = []
            sanitized_slope = {}
            sanitized_intercept = {}
            for k, x in enumerate(xs):
                sanitized_y.append(ys[j][k])
                sanitized_slope[k] = slopes[j][k]
                sanitized_intercept[k] = intercepts[j][k]

            sanitized_ys[j] = sanitized_y
            sanitized_slopes[j] = sanitized_slope
            sanitized_intercepts[j] = sanitized_intercept
            sanitized_index_segments[j] = range(k)

    weights = {}
    deadlines = {}
    for external_api in external_apis:
        weights[external_api] = 1/len(external_apis)
        deadlines[external_api] = latency_slo

    return Problem(external_apis, apis, chains, sanitized_index_segments, sanitized_ys, sanitized_slopes, sanitized_intercepts,
                   weights, deadlines)
    # return Problem(external_apis, apis, chains, index_segments, ys, slopes, intercepts,
    #                weights, deadlines)

def parse_profiles(file_name, latency_slo):
    hr_call_chains = {
        ("frontend", "frontend-hotels"):
            [("search", "search-near-by"),
             ("geo", "geo-near-by"),
             ("rate", "rate-get-rates"),
             ("reservation", "reservation-check-availability"),
             ("profile", "profile-get-profiles")],
        ("frontend", "frontend-recommendations"):
            [("recommendation", "recommendation-get-recommendations"),
            ("profile", "profile-get-profiles")],
        ("frontend", "frontend-user"):
            [("user", "user-check-user")],
        ("frontend", "frontend-reservation"):
            [("user", "user-check-user"),
             ("reservation", "reservation-make-reservation")]
    }

    with open(file_name, "r") as file:
        profiles = json.load(file)

        # A_E and A
        external_apis = []
        metadata_external_apis = {}
        apis = []
        metadata_apis = {}

        # b_j_k, s_j_k, c_j_k, L_j
        y_intersections = {}
        slopes = {}
        intercepts = {}
        index_segments = {}
        
        j = 0
        for profile in profiles:
            for i in range(len(profile["segments_per_api"]) - 1):
                print(f'{profile["microservice"]}-{profile["api"]}-{i}')
                if profile["microservice"] == "frontend" and i == 0:
                    external_apis.append(j)
                    metadata_external_apis[f'{profile["microservice"]}-{profile["api"]}-{i}'] = j

                apis.append(j)
                metadata_apis[f'{profile["microservice"]}-{profile["api"]}-{i}'] = j
                # index_segments[j] = list(range(len(profile["segments_per_api"][i+1])))

                y_intersections[j] = []
                slopes[j] = []
                intercepts[j] = []

                for segment in profile["segments_per_api"][i+1]:
                    if segment[3] <= 0:
                        continue

                    if segment[0] * segment[2] - segment[1] < 0:
                        if segment[0] * segment[3] - segment[1] > 0:
                            y_intersections[j].append(0)
                        else:
                            continue
                    else:
                        if math.isinf(segment[0] * segment[2] - segment[1]):
                            y_intersections[j].append(1e308)
                        else:
                            y_intersections[j].append(segment[0] * segment[2] - segment[1])

                    slopes[j].append(segment[0])
                    intercepts[j].append(segment[1])

                    if segment[3] > latency_slo:
                        if math.isinf(segment[0] * segment[3] - segment[1]):
                            y_intersections[j].append(1e308)
                        else:
                            y_intersections[j].append(segment[0] * segment[3] - segment[1])
                        break
                        
                index_segments[j] = list(range(len(slopes[j])))
                j += 1

        chains = {}
        for m1, a1 in hr_call_chains:
            external_api = f"{m1}-{a1}-0"
            chain = hr_call_chains[(m1, a1)]

            external_api_index = metadata_external_apis[external_api]
            for m2, a2 in chain:
                i = 0
                chain_indices = []
                while f"{m2}-{a2}-{i}" in metadata_apis:
                    chain_indices.append(metadata_apis[f"{m2}-{a2}-{i}"])
                    i += 1
            chains[external_api_index] = chain_indices


        print("external_apis:", external_apis)
        print("metadata_external_apis:", metadata_external_apis)
        print("apis:", apis)
        print("metadata_apis:", metadata_apis)
        print("chains:", chains)
        for j in range(len(index_segments)):
            print(f"{j}  y: {y_intersections[j]} s: {slopes[j]} c: {intercepts[j]} L: {index_segments[j]}")
        
    weights = {}
    deadlines = {}
    for external_api in external_apis:
        weights[external_api] = 1/len(external_apis)
        deadlines[external_api] = latency_slo

    return Problem(external_apis, apis, chains, index_segments, y_intersections, slopes, intercepts, weights, deadlines)

def plot_tikz_image_by_hull(x_start, x_end, s, c, indices, name, x_cap):
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
    domain=-10:10,
    samples=200,
    legend pos=north west,
    grid=both,
]
"""

    for index in range(len(x_start)):
        tikz_code += f"\\addplot[domain={x_start[index]}:{x_end[index]}, samples=2] {{{s[index]}*x - {c[index]}}};\n"
    
    if x_end[index] < x_cap:
        tikz_code += f"\\addplot[domain={x_end[index]}:{x_cap}, samples=2] {{{s[index]}*x - {c[index]}}};\n"

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
    parser = argparse.ArgumentParser()
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

    problem = parse_profiles_with_subsidiaries("output/040/040.hr.json", 2)
    parameters = problem.solve()
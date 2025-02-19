#!/bin/python3
import json
import math
import os
import subprocess
import numpy as np
import logging
import argparse
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus
# from pulp import PULP_CBC_CMD
from pulp import GUROBI_CMD

class Problem:
    def __init__(self, external_apis, apis, chains, index_segments, y_intersections, slopes,
                 intercepts, weights, deadlines):
        self.A_E = external_apis
        self.A = apis
        self.C = chains
        self.L_j = index_segments
        self.y_j_k = y_intersections
        self.s_j_k = slopes
        self.c_j_k = intercepts
        self.w = weights
        self.D = deadlines
        print(index_segments)

    def solve(self):
        problem = LpProblem(name="milp-optimization", sense=LpMaximize)

        M = 1e4
        epsilon = 0.1

        # Variables and constraint (11)
        b = LpVariable.dicts("b", [(i, j) for i in self.A_E for j in self.A], lowBound=0)
        B = LpVariable.dicts("B", self.A, lowBound=0)
        z = LpVariable.dicts("z", self.A_E, lowBound=0)
        v = LpVariable.dicts("v", [(j, k) for j in self.A for k in self.L_j[j]], lowBound=0)

        # constraints (22, 23)
        delta_b = LpVariable.dicts("delta",
                                   [(i, j, k) for i in self.A_E for j in self.A
                                    for k in self.L_j[j]], cat=LpBinary)
        lambda_b = LpVariable.dicts("lambda",
                                    [(j, k) for j in self.A for k in self.L_j[j]], cat=LpBinary)

        # objective function (7)
        problem += lpSum(self.w[i] * (z[i] + b[i, j]) for i in self.A_E for j in self.C[i])
        # problem += lpSum(self.w[i] * z[i] for i in self.A_E)

        # # constraint (8)
        # for i in self.A_E:
        #     for j in self.C[i]:
        #         for k in self.L_j[j]:
        #             problem += z[i] <= self.s_j_k[j][k] * delta_b[i, j, k]

        # constraint (8)
        for i in self.A_E:
            for j in self.C[i]:
                problem += z[i] <= lpSum(self.s_j_k[j][k] * delta_b[i, j, k] for k in self.L_j[j])

        # constraints (9, 10)
        for i in self.A_E:
            for j in self.C[i]:
                for k in self.L_j[j]:
                    problem += self.y_j_k[j][k] <= b[i, j] + M * (1 - delta_b[i, j, k])
                    problem += b[i, j] <= self.y_j_k[j][k+1] - epsilon + M * (1 - delta_b[i, j, k])

        # constraint (12)
        for i in self.A_E:
            problem += lpSum((v[j, k] + lambda_b[j, k] * self.c_j_k[j][k]) / self.s_j_k[j][k]
                             for j in self.C[i] for k in self.L_j[j]) <= self.D[i]

        # constraints (13, 14, 15, 16)
        for j in self.A:
            for k in self.L_j[j]:
                problem += v[j, k] <= M * lambda_b[j, k]
                problem += v[j, k] >= -M * lambda_b[j, k]
                problem += v[j, k] <= B[j] + M * (1 - lambda_b[j, k])
                problem += v[j, k] >= B[j] - M * (1 - lambda_b[j, k])

        # constraints (17, 18)
        for j in self.A:
            for k in self.L_j[j]:
                problem += self.y_j_k[j][k] <= B[j] + M * (1 - lambda_b[j, k])
                problem += B[j] <= self.y_j_k[j][k + 1] - epsilon + M * (1 - lambda_b[j, k])

        # constraint (19)
        for j in self.A:
            problem += lpSum(b[i, j] for i in self.A_E) == B[j]

        # constraint (20)
        for i in self.A_E:
            for j in self.C[i]:
                problem += lpSum(delta_b[i, j, k] for k in self.L_j[j]) == 1

        # constraint (21)
        # for j in self.A:
        for i in self.A_E:
            for j in self.C[i]:
                problem += lpSum(lambda_b[j, k] for k in self.L_j[j]) == 1

        # Additional constraint: stability of token buckets 
        for i in self.A_E:
            for j in self.C[i]:
                problem += b[i, j] >= lpSum(self.s_j_k[j][k] * delta_b[i, j, k] * 1/20 for k in self.L_j[j])

        problem.solve(GUROBI_CMD(options=[
            ("Presolve", 0),
            ("FeasibilityTol", 1e-9),
            ("IntFeasTol", 1e-9),("MIPGap", 1e-9), ("TimeLimit", 600)]))
        print("Solver Status:", LpStatus[problem.status])
        if LpStatus[problem.status] == "Not Solved":
            return

        print(f"Objective Value = {problem.objective.value()}")
        # for j in self.A:
        #     for k in self.L_j[j]:
        #         if v[j, k].varValue == 0:
        #             continue
        #         print(f"v[j={j},k={k}] = {v[j, k].varValue} c[j={j},k={k}] = {self.c_j_k[j][k]}")
        
        budgets_per_eapi = {}
        for i in self.A_E:
            budgets = {}
            for j in self.C[i]:
                budget = 0
                for k in self.L_j[j]:
                    budget += (v[j, k].varValue + lambda_b[j, k].varValue * self.c_j_k[j][k]) / self.s_j_k[j][k]
                budgets[j] = budget

            # print(f"budget[i={i:2}]")
            # sum = 0
            # for j, budget in budgets.items():
            #     print(f"  {j:2} = {budget:.3f}")
            #     sum += budget
            # print(f" sum = {sum:.3f}")
            budgets_per_eapi[i] = budgets

        for i in self.A_E:
            # for j in self.A:
            for j in self.C[i]:
                for k in self.L_j[j]:
                    if delta_b[i, j, k].varValue <= 1e-4:
                        continue
                    print(f"delta[i={i:2},j={j:2},k={k:2}]",
                          f"s[j={j:2}][k={k:2}] = {self.s_j_k[j][k]:8.3f}",
                          f"b[i={i:2},j={j:2}] = {b[i, j].varValue:8.3f}",
                          f"B[j={j:2}] = {B[j].varValue:8.3f}",
                          f"z[i={i:2}] = {z[i].varValue:8.3f}",
                          f"budgets[i={i:2},j={j:2}] = {budgets_per_eapi[i][j]:.3f}")
            print(f"D[i={i:2}] = {self.D[i]} sum_partitions = {sum(budgets_per_eapi[i].values()):.3f}")
                    
        # for j in self.A:
        #     for k in self.L_j[j]:
        #         if lambda_b[j, k].varValue == 0:
        #             continue
        #         print(f"lambda[j={j},k={k}] = {lambda_b[j, k].varValue}")


        for constraint_name, constraint in problem.constraints.items():
            lhs_value = constraint.value()
            rhs_value = constraint.constant

            # if constraint.sense == -1:  # (LHS <= RHS)
            #     slack = rhs_value - lhs_value
            # elif constraint.sense == 1:  # (LHS >= RHS)
            #     slack = lhs_value - rhs_value
            # else:  # (LHS = RHS)
            #     slack = abs(lhs_value - rhs_value)

            # print(f"{constraint_name} {constraint} ({constraint.sense}): lhs = {lhs_value}, rhs = {rhs_value}, slack = {slack}")
            # if "10000" in f"{constraint}":
            #     continue
            # slack = -lhs_value if constraint.sense == -1 else lhs_value
            # print(f"{constraint_name} {constraint} ({constraint.sense}) lhs: {lhs_value} slack: {slack}")

            # if abs(slack) < 1e-6:
            #     print(f"  🔥 Constraint {constraint_name} is Tight! (slack = {slack})")
        return 

def parse_profiles_with_subsidiaries(file_name, latency_slos):
    max_latency_slo = np.max(latency_slos)
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
            [("geo", "geo-near-by"),
             ("rate", "rate-get-rates"),
             ("reservation", "reservation-check-availability"),
             ("profile", "profile-get-profiles"),
             ("user", "user-check-user")],
        ("frontend", "frontend-recommendations"):
            [("frontend", "frontend-recommendations"),
             ("geo", "geo-near-by"),
             ("recommendation", "recommendation-get-recommendations"),
             ("reservation", "reservation-check-availability"),
             ("profile", "profile-get-profiles"),
             ("user", "user-check-user")],
        ("frontend", "frontend-user"):
            [("frontend", "frontend-user"),
             ("user", "user-check-user")],
        ("frontend", "frontend-reservation"):
            [("frontend", "frontend-reservation"),
             ("user", "user-check-user"),
             ("reservation", "reservation-make-reservation")]
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
                    slope, intercept, x_start, x_end = segment

                    # Skip the segments on the left of y-axis
                    if x_end <= 0:
                        continue

                    start_y = slope * x_start - intercept
                    end_y = slope * x_end - intercept

                    # If the segment x-START's y-value is negative and x-END's y-value is positive,
                    # it is the first meaningful segment. Otherwise,
                    # (1) both x-START and x-END are leq to zero or
                    # (2) both x-START and x-END are geq to zero
                    if start_y < 0:
                        if end_y > 0:
                            y_full[i].append(0)
                        else:
                            continue
                    else:
                        if math.isinf(start_y):
                            y_full[i].append(1e308)
                        else:
                            y_full[i].append(start_y)

                    slopes_full[i].append(slope)
                    intercepts_full[i].append(intercept)

                    # If the segment x-END is inf, break
                    if x_end == float('inf'):
                        # if math.isinf(segment[0] * latency_slo - segment[1]):
                        #     y_full[i].append(1e308)
                        # else:
                        #     y_full[i].append(segment[0] * latency_slo - segment[1])
                        break

            # Merge sub-service curves
            merged_y = sorted(set(value for values in y_full.values() for value in values))
            merged_slopes = {}
            merged_intercepts = {}
            for k, bp in enumerate(merged_y):
                num_spans = len(y_full)
                segment_indices_to_add = [0 for _ in range(num_spans)]
                for i in range(num_spans):
                    # Start at index 0, find the segment in i-th span that contains this breakpoint
                    segment_index = segment_indices_to_add[i]
                    while True:
                        if segment_index + 1 >= len(y_full[i]):
                            break

                        if bp >= y_full[i][segment_index+1]:
                            segment_index += 1
                        else:
                            break
                    segment_indices_to_add[i] = segment_index

                # Calculate the slope (M) and intercept (C) of the merged curve
                # M is 1/(1/s_1 + 1/s_2 + ...) (the harmonic mean of slopes)
                # C is M * (c_1/s_1 + c_2/s_2 + ...)
                slope_inverse_sum = 0
                frac_intercept_sum = 0
                for i, index in enumerate(segment_indices_to_add):
                    slope_inverse_sum += 1 / slopes_full[i][index]
                    frac_intercept_sum += intercepts_full[i][index] / slopes_full[i][index]
                merged_slopes[k] = 1 / slope_inverse_sum
                merged_intercepts[k] = merged_slopes[k] * frac_intercept_sum
            merged_y.append(1e308)

            # print(f'{profile["microservice"]}-{profile["api"]} bp: {merged_y}')
            # print(f'{profile["microservice"]}-{profile["api"]} ss: {merged_slopes}')
            # print(f'{profile["microservice"]}-{profile["api"]} cs: {merged_intercepts}')

            ys[j] = merged_y
            slopes[j] = merged_slopes
            intercepts[j] = merged_intercepts
            index_segments[j] = range(len(merged_y) - 1)

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
        
        for name in metadata_apis:
            sanitized_y = []
            sanitized_slope = {}
            sanitized_intercept = {}

            j = metadata_apis[name]
            xs = []
            truncated = False
            # print(f"{name} ({j}):")
            for k in index_segments[j]:
                x = (ys[j][k] + intercepts[j][k]) / slopes[j][k]
                # print(f"  k={k} x={x} y={ys[j][k]} s={slopes[j][k]} c={intercepts[j][k]}")
                if x >= max_latency_slo:
                    truncated = True
                    break
                xs.append(x)
                sanitized_y.append(ys[j][k])
                sanitized_slope[k] = slopes[j][k]
                sanitized_intercept[k] = intercepts[j][k]

            xs.append(max_latency_slo)
            if truncated:
                sanitized_y.append(slopes[j][k-1] * max_latency_slo - intercepts[j][k-1])
                print(f"{slopes[j][k-1] * max_latency_slo - intercepts[j][k-1]} = {slopes[j][k-1]} * {max_latency_slo} - {intercepts[j][k-1]}")
            else:
                sanitized_y.append(slopes[j][k] * max_latency_slo - intercepts[j][k])
                print(f"{slopes[j][k] * max_latency_slo - intercepts[j][k]} = {slopes[j][k]} * {max_latency_slo} - {intercepts[j][k]}")
            
            sanitized_ys[j] = sanitized_y
            sanitized_slopes[j] = sanitized_slope
            sanitized_intercepts[j] = sanitized_intercept
            sanitized_index_segments[j] = range(len(sanitized_y) - 1)
            print(f"{metadata_apis_inverse[j]} {j}")
            print(f"  x: {xs}")
            print(f"  y: {sanitized_ys[j]}")
            print(f"  s: {sanitized_slopes[j]}")
            print(f"  c: {sanitized_intercepts[j]}")
            print(f"  L: {sanitized_index_segments[j]}")
            
            if args.plot:
                x_start = []
                x_end = []
                for k in sanitized_index_segments[j]:
                    x_start.append(xs[k])
                    x_end.append(xs[k+1])
                plot_tikz_image_by_hull(x_start, x_end, slopes[j], intercepts[j], index_segments[j], f"int.{name}", max_latency_slo)

    weights = {}
    deadlines = {}
    for i, external_api in enumerate(external_apis):
        weights[external_api] = 1 / len(external_apis)
        deadlines[external_api] = latency_slos[i]

    return Problem(external_apis, apis, chains, sanitized_index_segments, sanitized_ys, sanitized_slopes, sanitized_intercepts,
                   weights, deadlines)

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
    parser.add_argument('--plot', '-p', action='store_true')
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

    latency_slos = [0.6, 1, 1.9, 1.5]
    problem = parse_profiles_with_subsidiaries("output/040/040.hr.json", latency_slos)
    parameters = problem.solve()
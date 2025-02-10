#!/bin/python3
import json
import math
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

class Problem:
    def __init__(self, external_apis, apis, chains, index_segments, y_intersections, slopes, intercepts, weights, deadline):
        self.A_E = external_apis
        self.A = apis
        self.C = chains
        self.L_j = index_segments
        self.b_j_k = y_intersections
        self.s_j_k = slopes
        self.c_j_k = intercepts
        self.w = weights
        self.D = deadline

    def solve(self):
        # T = slo  # Total constraint
        # M = 100  # Large constant for "Big M" method

        # Problem definition
        problem = LpProblem(name="milp-optimization", sense=LpMaximize)

        # A_E = [0, 1, 2]
        # A = [0, 1, 2, 3]
        # C = {0: [0, 1], 1: [2, 3], 2: [1, 3]}
        # L_j = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}
        # b_j_k = {0: [1, 5, 10], 1: [2, 6, 10], 2: [3, 7, 10], 3: [4, 8, 10]}
        # s_j_k = {0: [3, 7], 1: [5, 10], 2: [6, 12], 3: [4, 8]}
        # c_j_k = {0: [0, 2], 1: [1, 3], 2: [2, 4], 3: [3, 5]} 

        M = 1000
        epsilon = 1e-6

        # Variables and constraint (10)
        b = LpVariable.dicts("b", [(i, j) for i in self.A_E for j in self.A], lowBound=0)
        B = LpVariable.dicts("B", self.A, lowBound=0)
        z = LpVariable.dicts("z", self.A_E, lowBound=0)
        v = LpVariable.dicts("v", [(j, k) for j in self.A for k in self.L_j[j]], lowBound=0)

        # constraints (21, 22)
        delta_b = LpVariable.dicts("delta", [(i, j, k) for i in self.A_E for j in self.A for k in self.L_j[j]], cat=LpBinary)
        lambda_b = LpVariable.dicts("lambda", [(j, k) for j in self.A for k in self.L_j[j]], cat=LpBinary)

        # objective function (6)
        problem += lpSum(self.w[i] * z[i] for i in self.A_E)

        # constraint (7)
        for i in self.A_E:
            for j in self.C[i]:
                for k in self.L_j[j]:
                    problem += z[i] <= self.s_j_k[j][k] * delta_b[i, j, k]

        # constraints (8, 9)
        for i in self.A_E:
            for j in self.C[i]:
                for k in self.L_j[j]:
                    problem += self.b_j_k[j][k] <= b[i, j] + M * (1 - delta_b[i, j, k])
                    problem += b[i, j] <= self.b_j_k[j][k + 1] - epsilon + M * (1 - delta_b[i, j, k])

        # constraint (11)
        for i in self.A_E:
            problem += lpSum((v[j, k] - lambda_b[j, k] * self.c_j_k[j][k]) / self.s_j_k[j][k] for j in self.C[i] for k in self.L_j[j]) <= self.D[i]

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
                print(f"v[j={j},k={k}] = {v[j, k].varValue}")
        for i in self.A_E:
            for j in self.A:
                for k in self.L_j[j]:
                    print(f"delta[i={i},j={j},k={k}] = {delta_b[i, j, k].varValue}")
        for j in self.A:
            for k in self.L_j[j]:
                print(f"lambda[j={j},k={k}] = {lambda_b[j, k].varValue}")

def parse_profiles_with_subsidiaries(file_name, latency_slo):
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
        
        for j, profile in enumerate(profiles):
            print(f'{profile["microservice"]}-{profile["api"]}')
            if profile["microservice"] == "frontend":
                external_apis.append(j)
                metadata_external_apis[f'{profile["microservice"]}-{profile["api"]}'] = j

            apis.append(j)
            metadata_apis[f'{profile["microservice"]}-{profile["api"]}'] = j

            y_intersection_full = {}
            slopes_full = {}
            intercepts_full = {}
            for i, segments in enumerate(profile["segments_per_api"][1:]):
                y_intersection_full[i] = []
                slopes_full[i] = []
                intercepts_full[i] = []
                for segment in segments:
                    if segment[3] <= 0:
                        continue

                    if segment[0] * segment[2] - segment[1] < 0:
                        if segment[0] * segment[3] - segment[1] > 0:
                            y_intersection_full[i].append(0)
                        else:
                            continue
                    else:
                        if math.isinf(segment[0] * segment[2] - segment[1]):
                            y_intersection_full[i].append(1e308)
                        else:
                            y_intersection_full[i].append(segment[0] * segment[2] - segment[1])

                    slopes_full[i].append(segment[0])
                    intercepts_full[i].append(segment[1])

                    if segment[3] > latency_slo:
                        if math.isinf(segment[0] * segment[3] - segment[1]):
                            y_intersection_full[i].append(1e308)
                        else:
                            y_intersection_full[i].append(segment[0] * segment[3] - segment[1])
                        break

            print(f'{profile["microservice"]}-{profile["api"]} y: {y_intersection_full}')
            print(f'{profile["microservice"]}-{profile["api"]} s: {slopes_full}')
            print(f'{profile["microservice"]}-{profile["api"]} c: {intercepts_full}')

            breakpoints = sorted(set(value for values in y_intersection_full.values() for value in values))
            for k, bp in enumerate(breakpoints):
                segments_to_add = []
                for i in range(len(y_intersection_full)):
                    for k, insertection in enumerate(y_intersection_full[i]):
                        if bp < insertection:
                            segments_to_add.append(k-1)
                            break
                print(f"bp: {bp} sta:", segments_to_add)

                if len(segments_to_add) == 0:
                    continue
                
                slope_inverse_sum = 0
                frac_intercept_sum = 0
                for i, index in enumerate(segments_to_add):
                    slope_inverse_sum += 1 / slopes_full[i][index]
                    frac_intercept_sum += intercepts_full[i][index] / slopes_full[i][index]

                print(slope_inverse_sum)

                slopes[k] = 1 / slope_inverse_sum
                intercepts[k] = frac_intercept_sum

            print(f'{profile["microservice"]}-{profile["api"]} bp: {breakpoints}')
            print(f'{profile["microservice"]}-{profile["api"]} ss: {slopes}')
            print(f'{profile["microservice"]}-{profile["api"]} cs: {intercepts}')
            # while True:
            #     spans_to_add = []
            #     for i, segments in enumerate(profile["segments_per_api"][1:]):

            # breakpoints = [segments[0][2] for segments in profile["segments_per_api"][1:]]
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
            for m2, a2 in chain:
                chain_indices = []
                chain_indices.append(metadata_apis[f"{m2}-{a2}"])
            chains[external_api_index] = chain_indices

        # print("external_apis:", external_apis)
        # print("metadata_external_apis:", metadata_external_apis)
        # print("apis:", apis)
        # print("metadata_apis:", metadata_apis)
        # print("chains:", chains)
        # for j in range(len(index_segments)):
        #     print(f"{j}  y: {y_intersections[j]} s: {slopes[j]} c: {intercepts[j]} L: {index_segments[j]}")
        
    weights = {}
    deadlines = {}
    for external_api in external_apis:
        weights[external_api] = 1/len(external_apis)
        deadlines[external_api] = latency_slo

    return Problem(external_apis, apis, chains, index_segments, y_intersections, slopes, intercepts, weights, deadlines)

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

if __name__ == "__main__":
    problem = parse_profiles_with_subsidiaries("output/040/040.hr.json", 1)
    # parameters = problem.solve()
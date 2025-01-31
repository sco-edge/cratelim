#!/bin/python3
import json
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

def solve_mip_burst(apis, metadata_apis, slo):
    T = slo  # Total constraint
    M = 100000000   # Large constant for "Big M" method

    # Problem definition
    model = LpProblem(name="burst-mip-optimization", sense=LpMaximize)

    # Variables
    y = [LpVariable(f"y_{i}", lowBound=0) for i in range(len(apis))]
    t = LpVariable("t", lowBound=0)
    x = [[[LpVariable(f"x_{i}_{j}_{k}", lowBound=0) for k in range(len(segments))] for j, segments in enumerate(spans)] for i, spans in enumerate(apis)]
    z = [[[LpVariable(f"z_{i}_{j}_{k}", cat="Binary") for k in range(len(segments))] for j, segments in enumerate(spans)] for i, spans in enumerate(apis)]

    # Objective function: Maximize t
    model += t

    # Constraints
    for i, spans in enumerate(apis):
        for j, segments in enumerate(spans):
            # Ensure y[i] is in exactly one segment (3)
            model += lpSum(z[i][j]) == 1

            for k, (slope, intercept, start, end) in enumerate(segments):
                # Big-M constraints to enforce segment activation
                model += y[i] >= slope * x[i][j][k] - intercept - M * (1 - z[i][j][k])
                model += y[i] <= slope * x[i][j][k] - intercept + M * (1 - z[i][j][k])

                # Ensure x[i][j][k] is only active in its segment (4)
                model += x[i][j][k] >= start * z[i][j][k]
                model += x[i][j][k] <= end * z[i][j][k]
                # model += y[i] >= start * z[i][j]
                # model += y[i] <= end * z[i][j]

                # Ensure t <= slope for the active segment (1)
                # model += t <= slope * z[i][j]
                
        model += lpSum(y[i]) >= t
    # Global constraint on horizontal distance
    g = lpSum(x[i][j][k] for i, spans in enumerate(apis) for j, segments in enumerate(spans) for k in range(len(segments)))
    model += g <= T

    # Solve
    model.solve(PULP_CBC_CMD(msg=False))
    print("Status:", model.status)
    print("t:", t.value())
    y_values = [var.value() for var in y]
    print("y values:", y_values)
    slopes = []
    for i, spans in enumerate(apis):
        for j, segments in enumerate(spans):
            for k, (slope, intercept, start, end) in enumerate(segments):
                if z[i][j][k].value() == 1:
                    print(f"z[{i}][{j}][{k}]: {x[i][j][k].value()} {apis[i][j][k]} {metadata_apis[i][j]}")
                    slopes.append(slope * z[i][j][k].value())
    
    print(slopes)
                    
    # for i, spans in enumerate(apis):
    #     for j, segments in enumerate(spans):
    #         for k in range(len(segments)):
    #             if not x[i][j][k].value() == 0:
    #                 print(f"x[{i}][{j}][{k}]: {x[i][j][k].value()} {apis[i][j][k]} {metadata_apis[i][j]}")
                    

    print("g_i values:", [sum(x[i][j][k].value() for i, spans in enumerate(apis) for j, segments in enumerate(spans) for k in range(len(segments)))])
    # h_values = [max(segments[j][0] * z[i][j].value() for j in range(len(segments))) for i, segments in enumerate(spans)]
    h_values = [min([max(segments[k][0] * z[i][j][k].value() for k in range(len(segments))) for j, segments in enumerate(spans)]) for i, spans in enumerate(apis)]
    print("h values:", h_values)

    return model.status, [(h_values[i], y_values[i]) for i in range(len(y_values))]



def solve_mip_network(apis, metadata_apis, slo):
    T = slo  # Total constraint
    M = 100000000   # Large constant for "Big M" method

    # Problem definition
    model = LpProblem(name="mip-optimization", sense=LpMaximize)

    # Variables
    y = [LpVariable(f"y_{i}", lowBound=0) for i in range(len(apis))]
    t = LpVariable("t", lowBound=0)
    x = [[[LpVariable(f"x_{i}_{j}_{k}", lowBound=0) for k in range(len(segments))] for j, segments in enumerate(spans)] for i, spans in enumerate(apis)]
    z = [[[LpVariable(f"z_{i}_{j}_{k}", cat="Binary") for k in range(len(segments))] for j, segments in enumerate(spans)] for i, spans in enumerate(apis)]

    # Objective function: Maximize t
    # model += t + lpSum(x[i][j][k] for i, spans in enumerate(apis) for j, segments in enumerate(spans) for k in range(len(segments)))
    model += t

    # Constraints
    for i, spans in enumerate(apis):
        # y threshold
        model += y[i] >= 120

        for j, segments in enumerate(spans):
            # Ensure y[i] is in exactly one segment (3)
            model += lpSum(z[i][j]) == 1
            s_sum = []

            for k, (slope, intercept, start, end) in enumerate(segments):
                # Big-M constraints to enforce segment activation
                model += y[i] >= slope * x[i][j][k] - intercept - M * (1 - z[i][j][k])
                model += y[i] <= slope * x[i][j][k] - intercept + M * (1 - z[i][j][k])

                # Ensure x[i][j][k] is only active in its segment (4)
                model += x[i][j][k] >= start * z[i][j][k]
                model += x[i][j][k] <= end * z[i][j][k]
                # model += y[i] >= start * z[i][j]
                # model += y[i] <= end * z[i][j]

                # Ensure t <= slope for the active segment (1)
                # model += t <= slope * z[i][j]
                
                s_sum.append(slope * z[i][j][k])
            model += lpSum(s_sum) >= t
    # Global constraint on horizontal distance
    g = lpSum(x[i][j][k] for i, spans in enumerate(apis) for j, segments in enumerate(spans) for k in range(len(segments)))
    model += g <= T

    # Solve
    model.solve(PULP_CBC_CMD(msg=False))
    print("Status:", model.status)
    print("t:", t.value())
    y_values = [var.value() for var in y]
    print("y values:", y_values)
    slopes = []
    for i, spans in enumerate(apis):
        for j, segments in enumerate(spans):
            for k, (slope, intercept, start, end) in enumerate(segments):
                if z[i][j][k].value() == 1:
                    print(f"z[{i}][{j}][{k}]: {x[i][j][k].value()} {apis[i][j][k]} {metadata_apis[i][j]}")
                    slopes.append(slope * z[i][j][k].value())
    
    print(slopes)
                    
    # for i, spans in enumerate(apis):
    #     for j, segments in enumerate(spans):
    #         for k in range(len(segments)):
    #             if not x[i][j][k].value() == 0:
    #                 print(f"x[{i}][{j}][{k}]: {x[i][j][k].value()} {apis[i][j][k]} {metadata_apis[i][j]}")
                    

    print("g_i values:", [sum(x[i][j][k].value() for i, spans in enumerate(apis) for j, segments in enumerate(spans) for k in range(len(segments)))])
    # h_values = [max(segments[j][0] * z[i][j].value() for j in range(len(segments))) for i, segments in enumerate(spans)]
    h_values = [min([max(segments[k][0] * z[i][j][k].value() for k in range(len(segments))) for j, segments in enumerate(spans)]) for i, spans in enumerate(apis)]
    print("h values:", h_values)

    return model.status, [(h_values[i], y_values[i]) for i in range(len(y_values))]

def solve_mip_unique(segments_per_api, slo):
    T = slo  # Total constraint
    M = 10000   # Large constant for "Big M" method

    # Problem definition
    model = LpProblem(name="convex-hull-optimization", sense=LpMaximize)

    # Variables
    y = [LpVariable(f"y_{i}", lowBound=0) for i in range(len(segments_per_api))]
    t = LpVariable("t", lowBound=0)
    x = [[LpVariable(f"x_{i}_{j}", lowBound=0) for j in range(len(hull))] for i, hull in enumerate(segments_per_api)]
    z = [[LpVariable(f"z_{i}_{j}", cat="Binary") for j in range(len(hull))] for i, hull in enumerate(segments_per_api)]

    # Objective function: Maximize t
    model += t

    # Constraints
    for i, hull in enumerate(segments_per_api):
        # Ensure y[i] is in exactly one segment (3)
        model += lpSum(z[i]) == 1

        s_sum = []
        for j, (slope, intercept, start, end) in enumerate(hull):
            # Big-M constraints to enforce segment activation
            model += y[i] >= slope * x[i][j] + intercept - M * (1 - z[i][j])
            model += y[i] <= slope * x[i][j] + intercept + M * (1 - z[i][j])

            # Ensure x[i][j] is only active in its segment (4)
            model += x[i][j] >= start * z[i][j]
            model += x[i][j] <= end * z[i][j]
            # model += y[i] >= start * z[i][j]
            # model += y[i] <= end * z[i][j]

            # Ensure t <= slope for the active segment (1)
            # model += t <= slope * z[i][j]
            
            s_sum.append(slope * z[i][j])

    model += lpSum(s_sum) >= t
    # Global constraint on horizontal distance
    g = lpSum(x[i][j] for i in range(len(segments_per_api)) for j in range(len(segments_per_api[i])))
    model += g <= T

    # Solve
    model.solve()
    print("Status:", model.status)
    print("t:", t.value())
    y_values = [var.value() for var in y]
    print("y values:", y_values)
    print("g_i values:", [sum(x[i][j].value() for j in range(len(hull))) for i, hull in enumerate(segments_per_api)])
    h_values = [max(hull[j][0] * z[i][j].value() for j in range(len(hull))) for i, hull in enumerate(segments_per_api)]
    print("h values:", h_values)

    return [(h_values[i], y_values[i]) for i in range(len(y_values))]

# convex_hulls = [
#     [(1, 0, 0, 10), (2, 10, 10, 20)],  # [(slope, intercept, start, end)]
#     [(0.5, 5, 0, 15), (1.5, 15, 15, 25)]
# ]
# T = 100  # Total constraint

# model = LpProblem(name="convex-hull-optimization", sense=LpMaximize)

# y = [LpVariable(f"y_{i}", lowBound=0) for i in range(len(convex_hulls))]
# t = LpVariable("t", lowBound=0)

# model += t

# for i, hull in enumerate(convex_hulls):
#     z = [LpVariable(f"z_{i}_{j}", cat="Binary") for j in range(len(hull))]
#     model += lpSum(z) == 1
#     model += t <= lpSum(hull[j][0] * z[j] for j in range(len(hull)))  # t <= slope
#     for j, (slope, intercept, start, end) in enumerate(hull):
#         model += start * z[j] <= y[i]
#         model += y[i] <= end * z[j]

# g = [lpSum((y[i] - hull[j][1]) / hull[j][0] * z[j] for j in range(len(hull))) for i, hull in enumerate(convex_hulls)]
# model += lpSum(g) <= T

# model.solve()
# print("Status:", model.status)
# print("t:", t.value())
# print("y:", [var.value() for var in y])


# timestamps = []
# intervals = []

latency_slo = 0.8
    
def parse_profiles(file_name, latency_slo):
    call_chain = [("frontend", "hotels"), ("search", "near-by"), ("geo", "near-by"), ("rate", "get-rates"), ("reservation", "check-availability"), ("profile", "get-profiles")]

    with open(file_name, "r") as file:
        apis = []
        metadata_microservices = []
        metadata_apis = []
        metadata_mappings = []
        profiles = json.load(file)

        # Preserve the API order of the call chain
        for api in call_chain:
            for profile in profiles:
                if not (profile["microservice"], profile["api"]) == api:
                    continue

                # print(f'{profile["microservice"]}-{profile["api"]}')
                segments = profile["segments_per_api"]
                sanitized_segments = []
                metadata_spans = []
                for i, spans in enumerate(segments):
                    # Skip the max backlog
                    if i == 0:
                        continue

                    spans_sanitized = []
                    for s, c, x_start, x_end in spans:
                        # if x_start == -1e+308:
                        if x_start <= 0.0:
                            x_start = 0.0
                        elif x_start >= latency_slo:
                            x_start = latency_slo
                        # else:
                        #     x_start = x_start * 1000
                        # if x_end >=- 1e+308:
                        if x_end >= latency_slo:
                            x_end = latency_slo
                        # else:
                        #     x_end = x_end * 1000
                        # print(i, (s, c, x_start, x_end))
                        x_len = x_end - x_start
                        if x_len >= 0.2:
                            partitioned_spans = []
                            num_partition = int(x_len / 0.2) + 1
                            step_size = x_len / num_partition

                            partitioned_spans.append((s, c, x_start, x_start + step_size))
                            for i in range(num_partition - 2):
                                partitioned_spans.append((s, c, x_start + step_size * (i+1), x_start + step_size * (i+2)))
                            partitioned_spans.append((s, c, x_start + step_size * (num_partition - 1), x_end))
                            print("P", profile["microservice"], profile["api"], x_len, num_partition, step_size, partitioned_spans)
                            spans_sanitized.extend(partitioned_spans)
                        else:
                            spans_sanitized.append((s, c, x_start, x_end))

                    sanitized_segments.append(spans_sanitized)
                    metadata_spans.append((profile["microservice"], profile["api"], i-1))

                apis.append(sanitized_segments)
                metadata_microservices.append(profile["microservice"])
                metadata_apis.append(metadata_spans)

    for i, api in enumerate(apis):
        for j, segments in enumerate(api):
            for k, segment in enumerate(segments):
                print(i, j, metadata_apis[i][j], segment)

    return apis, metadata_apis

# # Example call_graph data: [(slope, intercept, start, end)]
# call_graph = [
#     [[(100, 100, 0, 50), (200, 5000, 50, 100)]],
#     [[(40, 100, 0, 75), (120, 6000, 75, 100)]],
# ]

# call_graph = [
#     [[(100, 0, 0, 50), (200, 5000, 50, 100)], [(100, 0, 0, 50), (200, 5000, 50, 100)]],
#     [[(40, 0, 0, 75), (120, 6000, 75, 100)]],
# ]

apis, metadata_apis = parse_profiles("output/014/014.hr.json", latency_slo)

parameters = solve_mip_network(apis, metadata_apis, latency_slo)
print(parameters)

parameters = solve_mip_burst(apis, metadata_apis, latency_slo)
print(parameters)
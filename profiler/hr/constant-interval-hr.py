#!/bin/python3

import re
import aiohttp
import asyncio
import grpc
import time
import json
import argparse
import logging
import subprocess
import profile_pb2
import profile_pb2_grpc
import search_pb2
import search_pb2_grpc
import geo_pb2
import geo_pb2_grpc
import rate_pb2
import rate_pb2_grpc
import recommendation_pb2
import recommendation_pb2_grpc
import user_pb2
import user_pb2_grpc
import reservation_pb2
import reservation_pb2_grpc
from dataclasses import dataclass
import numpy as np

response_times = []
failed_responses = 0

@dataclass
class HttpAPI:
    is_gateway: bool
    port: int
    path: str
    api_type: str
    params: dict
    headers: dict

@dataclass
class GrpcAPI:
    is_gateway: bool
    port: int
    api_type: str
    params: list
    headers: dict

def read_profile_spec(target_app, target_microservice, target_api):
    with open(f'./spec.json', 'r') as file:
        data = json.load(file)

    if not target_app == data.get("app"):
        logging.error("application ids are different.")

    microservices = data.get("microservices", [])
    for microservice in microservices:
        if microservice.get("id") == target_microservice:
            apis = microservice.get("apis", [])
            for api in apis:  
                if api.get("id") == target_api:
                    api_type = api.get("type")
                    if api_type == "http":
                        return HttpAPI(
                            is_gateway=bool(microservice.get("gateway")),
                            port=int(microservice.get("port")),
                            path=api.get("path"),
                            api_type=api_type,
                            params=cast_http_params(api.get("params")),
                            headers={"api-context": target_api}
                        )
                    elif api_type == "grpc":
                        return GrpcAPI(
                            is_gateway=bool(microservice.get("gateway")),
                            port=int(microservice.get("port")),
                            api_type=api_type,
                            params=cast_grpc_params(api.get("params")),
                            headers={"api-context": target_api}
                        )
                    else:
                        raise ValueError(f"Unsupported (microservice, api): ({target_microservice}, {target_api})")

    raise ValueError(f"Unsupported (microservice, api): ({target_microservice}, {target_api})")

def cast_http_params(params):
    ret = {}
    for param in params:
        ret[param["key"]] = param["value"]
    return ret

def cast_grpc_params(params):
    ret = []
    for param in params:
        if param["type"] in ["float", "double"]:
            ret.append(float(param["value"]))
        elif param["type"] in [
            "int32", "int64", "uint32", "uint64", "sint32", "sint64",
            "fixed32", "fixed64", "sfixed32", "sfixed64"
        ]:
            ret.append(int(param["value"]))
        elif param["type"] == "bool":
            ret.append(param["value"].lower() == "true")
        elif param["type"] == "bytes":
            ret.append(bytes(param["value"]).encode("utf-8"))
        elif param["type"] == "string":
            ret.append(param["value"])
        else:
            raise ValueError(f"Unsupported param type: {param['type']}")

    return ret

async def send_grpc_request(stub, target_microservice, target_api, api):
    global failed_responses
    try:
        metadata = [("api-context", api.headers["api-context"])]
        start_time = time.perf_counter()

        if target_microservice == "profile" and target_api == "profile-get-profiles":
            request = profile_pb2.Request(hotelIds=api.params[0], locale=api.params[1])
            response = await stub.GetProfiles(request, metadata=metadata)
        elif target_microservice == "search" and target_api == "search-near-by":
            request = search_pb2.NearbyRequest(lat=float(api.params[0]), lon=float(api.params[1]), inDate=api.params[2], outDate=api.params[3])
            response = await stub.Nearby(request, metadata=metadata)
        elif target_microservice == "geo" and target_api == "geo-near-by":
            request = geo_pb2.Request(lat=float(api.params[0]), lon=float(api.params[1]))
            response = await stub.Nearby(request, metadata=metadata)
        elif target_microservice == "rate" and target_api == "rate-get-rates":
            request = rate_pb2.Request(hotelIds=api.params[0], inDate=api.params[1], outDate=api.params[2])
            response = await stub.GetRates(request, metadata=metadata)
        elif target_microservice == "recommendation" and target_api == "recommendation-get-recommendations":
            request = recommendation_pb2.Request(require=api.params[0], lat=float(api.params[1]), lon=float(api.params[2]))
            response = await stub.GetRecommendations(request, metadata=metadata)
        elif target_microservice == "user" and target_api == "user-check-user":
            request = user_pb2.Request(username=api.params[0], password=api.params[1])
            response = await stub.CheckUser(request, metadata=metadata)
        elif target_microservice == "reservation" and target_api == "reservation-make-reservation":
            request = reservation_pb2.Request(customerName=api.params[0], hotelId=api.params[1], inDate=api.params[2], outDate=api.params[3], roomNumber=int(api.params[4]))
            response = await stub.MakeReservation(request, metadata=metadata)
        elif target_microservice == "reservation" and target_api == "reservation-check-availability":
            request = reservation_pb2.Request(customerName=api.params[0], hotelId=api.params[1], inDate=api.params[2], outDate=api.params[3], roomNumber=int(api.params[4]))
            response = await stub.CheckAvailability(request, metadata=metadata)
        
        logging.debug(response)
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_times.append(response_time)

    except grpc.aio.AioRpcError as e:
        logging.debug(e)
        failed_responses += 1

    except Exception as e:
        logging.error(f"Error sending request: {e}")

async def grpc_test(rps, duration, ip, target_microservice, target_api, api):
    logging.debug(f"Starting load test with RPS: {rps} for {duration} seconds")
    interval = 1 / rps
    start_time = asyncio.get_event_loop().time()

    async with grpc.aio.insecure_channel(f"{ip}:{api.port}") as channel:
        if target_microservice == "profile":
            stub = profile_pb2_grpc.ProfileStub(channel)
        elif target_microservice == "search":
            stub = search_pb2_grpc.SearchStub(channel)
        elif target_microservice == "geo":
            stub = geo_pb2_grpc.GeoStub(channel)
        elif target_microservice == "rate":
            stub = rate_pb2_grpc.RateStub(channel)
        elif target_microservice == "recommendation":
            stub = recommendation_pb2_grpc.RecommendationStub(channel)
        elif target_microservice == "user":
            stub = user_pb2_grpc.UserStub(channel)
        elif target_microservice == "reservation":
            stub = reservation_pb2_grpc.ReservationStub(channel)
        
        next_request_time = start_time
        tasks = []

        while asyncio.get_event_loop().time() - start_time < duration:
            now = asyncio.get_event_loop().time()
            sleep_time = max(0, next_request_time - now)
            await asyncio.sleep(sleep_time)

            task = asyncio.create_task(send_grpc_request(stub, target_microservice, target_api, api))
            tasks.append(task)

            next_request_time += interval

        await asyncio.gather(*tasks)

    if response_times:
        print(f"{len(response_times)} {failed_responses} {np.mean(response_times)} {np.percentile(response_times, 50)}",
              f"{np.percentile(response_times, 90)} {np.percentile(response_times, 95)}",
              f"{np.percentile(response_times, 99)} {np.percentile(response_times, 99.9)}")
    else:
        logging.warning("No successful responses were collected during the load test.")

    logging.debug("Load test completed")

async def send_http_request(session, ip, api):
    global failed_responses
    try:
        start_time = time.perf_counter()
        # async with session.get(f"http://{ip}:{port}/{args.api}", params=parameters, headers=headers) as response:
        async with session.get(f"http://localhost:{api.port}/{api.path}", params=api.params, headers=api.headers) as response:
            logging.debug(response)
            body = await response.text()
            logging.debug(body)
            if response.status == 200:
                end_time = time.perf_counter()
                response_time = end_time - start_time
                response_times.append(response_time)    
            else:
                failed_responses += 1

    except grpc.aio.AioRpcError as e:
        failed_responses += 1

    except Exception as e:
        logging.error(f"Error sending request: {e}")

async def http_test(rps, duration, ip, api):
    logging.debug(f"Starting load test with RPS: {rps} for {duration} seconds")
    interval = 1 / rps
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=12000)
    async with aiohttp.ClientSession(connector=connector) as session:
        next_request_time = start_time
        tasks = []

        while time.time() - start_time < duration:
            now = time.time()
            sleep_time = max(0, next_request_time - now)
            await asyncio.sleep(sleep_time)

            task = asyncio.create_task(send_http_request(session, ip, api))
            tasks.append(task)

            next_request_time += interval

        await asyncio.gather(*tasks)

    if response_times:
        logging.debug(f"Load test completed")
        print(f"{len(response_times)} {failed_responses} {np.mean(response_times)} {np.percentile(response_times, 50)}",
              f"{np.percentile(response_times, 90)} {np.percentile(response_times, 95)}",
              f"{np.percentile(response_times, 99)} {np.percentile(response_times, 99.9)}")
    else:
        logging.warning("No successful responses were collected during the load test.")

    logging.debug("Load test completed")

def get_microservice_ip(app, microservice, is_gateway):
    try:
        # Run 'kubectl get pods -A -o wide' command to get pod details
        result = subprocess.run(
            ["kubectl", "get", "pods", "-A", "-o", "wide"],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output
        lines = result.stdout.splitlines()
        pod_data = lines[1:]

        pod_pattern = re.compile(r'^\S+\s+(\S+)\s+.*?\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+.*$')

        for line in pod_data:
            match = pod_pattern.match(line)
            if match:
                name, ip = match.groups()

                if is_gateway and name.startswith(f"{app}-gateway"):
                    logging.debug(f"Microservice '{microservice}' (gateway) is running at IP: {ip}")
                    return ip
                elif not is_gateway and name.startswith(microservice):
                    logging.debug(f"Microservice '{microservice}' is running at IP: {ip}")
                    return ip

        logging.error(f"No running pod found for microservice '{microservice}'")
        return None

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute kubectl command: {e.stderr}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rps', '-r', default=10)
    parser.add_argument('--duration', '-d', default=2000)
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

    rps = int(args.rps)
    duration = int(args.duration) / 1000
    # is_gateway, port, type, params, headers = read_profile_spec(args.app, args.microservice, args.api)
    api = read_profile_spec(args.app, args.microservice, args.api)
    ip = get_microservice_ip(args.app, args.microservice, api.is_gateway)

    logging.debug(f'Parameters: {api.params}')
    if api.api_type == "http":
        asyncio.run(http_test(rps, duration, ip, api))
    elif api.api_type == "grpc":
        asyncio.run(grpc_test(rps, duration, ip, args.microservice, args.api, api))
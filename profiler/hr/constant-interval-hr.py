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
from search_pb2 import NearbyRequest
from search_pb2_grpc import SearchStub

def read_profile_spec(target_app, target_microservice, target_api):
    with open(f'./{target_app}-spec.json', 'r') as file:
        data = json.load(file)

    print(target_app, target_microservice, target_api)
    if not target_app == data.get("app"):
        logging.error("application ids are different.")

    microservices = data.get("microservices", [])
    for microservice in microservices:
        if microservice.get("id") == target_microservice:
            apis = microservice.get("apis", [])
            for api in apis:  
                if api.get("id") == target_api:
                    return microservice.get("gateway"), microservice.get("port"), api.get("type"), api.get("params"), {"api-context": target_api}

    logging.error("no such api")

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

async def send_grpc_request(stub, lat, lon, inDate, outDate):
    try:
        metadata = [("api-context", "hotels"), ("progress-context", "near-by")]
        request = NearbyRequest(lat=float(lat), lon=float(lon), inDate=inDate, outDate=outDate)
        response = await stub.Nearby(request, metadata=metadata)
        print(f"Response: {response.hotelIds}")
    except Exception as e:
        print(f"Error sending request: {e}")

async def grpc_test(rps, duration, ip, port, params):
    print(f"Starting load test with RPS: {rps} for {duration} seconds")
    interval = 1 / rps
    start_time = asyncio.get_event_loop().time()

    async with grpc.aio.insecure_channel(f"{ip}:{port}") as channel:
        stub = SearchStub(channel)
        next_request_time = start_time
        tasks = []

        while asyncio.get_event_loop().time() - start_time < duration:
            now = asyncio.get_event_loop().time()
            sleep_time = max(0, next_request_time - now)
            await asyncio.sleep(sleep_time)

            task = asyncio.create_task(send_grpc_request(stub, *params))
            tasks.append(task)

            next_request_time += interval

        await asyncio.gather(*tasks)

    print("Load test completed")

async def send_http_request(session, ip, port, parameters, headers):
    try:
        async with session.get(f"http://{ip}:{port}/{args.api}", params=parameters, headers=headers) as response:
            print(f"Request sent. Status code: {response.status}")
    except Exception as e:
        print(f"Error sending request: {e}")

async def http_test(rps, duration, ip, port, params, headers):
    print(f"Starting load test with RPS: {rps} for {duration} seconds")
    interval = 1 / rps
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=3000)
    async with aiohttp.ClientSession(connector=connector) as session:
        next_request_time = start_time
        tasks = []

        while time.time() - start_time < duration:
            now = time.time()
            sleep_time = max(0, next_request_time - now)
            await asyncio.sleep(sleep_time)

            task = asyncio.create_task(send_http_request(session, ip, port, params, headers))
            tasks.append(task)

            next_request_time += interval

        await asyncio.gather(*tasks)

    print("Load test completed")

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
                    print(f"Microservice '{microservice}' (gateway) is running at IP: {ip}")
                    return ip
                elif not is_gateway and name.startswith(microservice):
                    print(f"Microservice '{microservice}' is running at IP: {ip}")
                    return ip

        logging.error(f"No running pod found for microservice '{microservice}'")
        return None

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to execute kubectl command: {e.stderr}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rps', '-r', default=10)
    parser.add_argument('--duration', '-d', default=3)
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
    duration = int(args.duration)
    is_gateway, port, type, params, headers = read_profile_spec(args.app, args.microservice, args.api)
    if type == "http":
        params = cast_http_params(params)
    elif type == "grpc":
        params = cast_grpc_params(params)
    else:
        raise ValueError(f"Unsupported api type: type")

    ip = get_microservice_ip(args.app, args.microservice, is_gateway)

    if type == "http":
        asyncio.run(http_test(rps, duration, ip, port, params, headers))
    elif type == "grpc":
        asyncio.run(grpc_test(rps, duration, ip, port, params))
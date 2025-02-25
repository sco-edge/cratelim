#!/bin/bash

# locust --headless -f locustfile-bi.py -u 2 -r 2 -H http://localhost:8080 -t 5
locust --headless -f locustfile-hr-topfull.py -H http://localhost:8080 -t 5
#!/bin/bash

kubectl apply -f resources/bi-ratelimit.yaml
# kubectl apply -f resources/bi-rls.yaml
# kubectl apply -f resources/rl.yaml
kubectl apply -f resources/bi-rls-xds.yaml
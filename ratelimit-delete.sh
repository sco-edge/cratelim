#!/bin/bash

kubectl delete -f resources/bi-ratelimit.yaml
kubectl delete -f resources/bi-rls.yaml
kubectl delete -f resources/rl.yaml
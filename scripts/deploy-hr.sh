#!/bin/bash

pushd ../apps/deathstarbench/hotelReservation
kubectl apply -Rf ./kubernetes
popd

pushd ../resources
kubectl apply -f hr-gateway.yaml
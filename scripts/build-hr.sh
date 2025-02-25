#!/bin/bash

PROJECT="cratelim"
IMAGE_HR="$PROJECT-hr"
TAG="latest"

pushd ../apps/deathstarbench/hotelReservation
docker build -t $IMAGE_HR:$TAG .
docker save $IMAGE_HR:$TAG -o $IMAGE_HR.tar

# For multiple worker nodes, 
# 1. copy the .tar of the image to the worker node
# 2. import the image (using the following command) at the worker node

ctr -n=k8s.io images import $IMAGE_HR.tar

popd
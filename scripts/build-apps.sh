#!/bin/bash

# build hr app (Hotel Reservation from DeathStarBench)
pushd ../apps/deathstarbench
git checkout 6ecb09706140f8730b5385c08f1386c654c3c526
popd
# md

# build bi app (Bookinfo from Istio)
pushd ../apps/istio
git checkout 1.24.2

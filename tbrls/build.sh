#!/bin/bash

# b4127c9b8d78b77423fd25169f05b7476b6ea932
URL="https://github.com/cncf/xds/archive/b4127c9b8d78b77423fd25169f05b7476b6ea932.tar.gz"
curl -sL -O "$URL"
FILENAME=$(basename "$URL")

if [[ "$FILENAME" == *.tar.gz ]]; then
  tar -xzvf "$FILENAME"
  echo "File $FILENAME has been extracted."
else
  echo "Error: File is not a .tar.gz archive."
  exit 1
fi

rm $FILENAME

# 1.0.4
VALIDATE_URL="https://github.com/bufbuild/protoc-gen-validate/archive/refs/tags/v1.0.4.zip"
curl -sL -O "$VALIDATE_URL"
FILENAME=$(basename "$VALIDATE_URL")

if [[ "$FILENAME" == *.zip ]]; then
  unzip "$FILENAME"
  echo "File $FILENAME has been extracted."
else
  echo "Error: File is not a .zip archive."
  exit 1
fi

rm $FILENAME

# Original proto path: envoy/api/envoy/service/ratelimit/v3/rls.proto
# Replaced .go path: (1) go-control-plane/envoy/service/ratelimit/v3/rls.pb.go
#                    (2) go-control-plane/envoy/service/ratelimit/v3/rls_grpc.pb.go
protoc --proto_path=./ \
--proto_path=../envoy/api/ \
--proto_path=./xds-b4127c9b8d78b77423fd25169f05b7476b6ea932/ \
--proto_path=./protoc-gen-validate-1.0.4/ \
--go_out=. \
--go-grpc_out=require_unimplemented_servers=false:. \
rls.proto 

# Original proto path: envoy/api/envoy/extensions/common/ratelimit/v3/ratelimit/proto
# Replaced .go path: go-control-plane/envoy/extensions/common/ratelimit/v3/ratelimit.pb.go
protoc --proto_path=./ \
--proto_path=../envoy/api/ \
--proto_path=./xds-b4127c9b8d78b77423fd25169f05b7476b6ea932/ \
--proto_path=./protoc-gen-validate-1.0.4/ \
--go_out=. \
--go-grpc_out=require_unimplemented_servers=false:. \
ratelimit.proto 

# Original proto path: ratelimit/api/ratelimit/config/ratelimit/v3/rls_conf.proto
# Replaced .go path: go-control-plane/ratelimit/config/ratelimit/v3/rls_conf.pb.go
protoc --go_out=. \
rls_conf.proto
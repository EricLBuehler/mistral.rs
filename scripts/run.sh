#!/bin/sh

set -e
cd $(dirname $0)/..
cargo build --release --features metal
./target/release/mistralrs-server --throughput --port 3001 -n 1000  plain -m microsoft/Phi-3.5-mini-instruct -a phi3

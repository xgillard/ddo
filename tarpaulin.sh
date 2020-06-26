#! /bin/bash
# This will run cargo tarpaulin on any docker equipped system
docker run --security-opt seccomp=unconfined -v "${PWD}:/volume" xd009642/tarpaulin cargo tarpaulin --ignore-tests --out Html

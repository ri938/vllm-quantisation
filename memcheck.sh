#!/bin/bash

/usr/local/cuda/bin/compute-sanitizer --tool memcheck python3 test.py

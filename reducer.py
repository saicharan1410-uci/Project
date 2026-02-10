#!/usr/bin/env python3
import sys

current_vendor = None
total = 0
count = 0

for line in sys.stdin:
    vendor, duration = line.strip().split("\t")
    duration = float(duration)

    if current_vendor == vendor:
        total += duration
        count += 1
    else:
        if current_vendor:
            print(f"{current_vendor}\t{total/count}")
        current_vendor = vendor
        total = duration
        count = 1

if current_vendor:
    print(f"{current_vendor}\t{total/count}")
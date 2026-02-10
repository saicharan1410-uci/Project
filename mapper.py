#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)

header = next(reader)  # skip header

for row in reader:
    try:
        vendor = row[1]
        duration = float(row[-1])
        print(f"{vendor}\t{duration}")
    except:
        pass

#!/bin/bash
output_file="outputs/output.csv"
num_repeats=10
for ((i=1; i<=$num_repeats; i++)); do
    .venv/bin/python main.py >> "$output_file"
done

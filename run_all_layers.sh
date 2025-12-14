#!/bin/bash

# deleting cache may not be needed
rm -rf ~/.iron/cache/*
python3 run_layer_0.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_head_0.py
# rm -rf ~/.iron/cache/*
# python3 run_layer_1_head_1.py
# rm -rf ~/.iron/cache/*
# python3 run_layer_1_head_2.py
# rm -rf ~/.iron/cache/*
# python3 run_layer_1_head_3.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_tail.py
rm -rf ~/.iron/cache/*
python3 run_layer_2-5.py
rm -rf ~/.iron/cache/*

python3 run_layer_6_tail.py
rm -rf ~/.iron/cache/*
python3 run_layer_7-12.py
rm -rf ~/.iron/cache/*
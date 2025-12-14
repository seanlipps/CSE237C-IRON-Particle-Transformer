#!/bin/bash

rm -rf ~/.iron/cache/*
python3 run_layer_0.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_head_0.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_head_1.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_head_2.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_head_3.py
rm -rf ~/.iron/cache/*
python3 run_layer_1_tail.py
rm -rf ~/.iron/cache/*
python3 run_layer_2-5.py
rm -rf ~/.iron/cache/*
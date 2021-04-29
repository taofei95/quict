#!/bin/bash

input_path="/home/shoulifu/QuICT/QuICT/qcda/mapping/cpp_test_data"

gamma=0.8
c=20

with_predictor=0
extended=0
is_generate_data=1

num_of_swap_gates=15
num_of_process=5
threshold_size=150
virtual_loss=0
num_of_iterations=20
num_of_playout=2
bp_mode=0

model_file_path="/home/shoulifu/QuICT/QuICT/qcda/mapping/warmup/output/mcts_model.pt"
device=1
num_of_circuit_process=5
inference_batch_size=8
max_batch_size=128

info=0




echo  "$num_of_process $num_of_iterations $num_of_swap_gates $virtual_loss $gamma $c $num_of_playout $is_data_generate " 
./lib/build/test_data_generator \
$input_path \
$gamma $c \
$with_predictor $extended $is_generate_data \
$num_of_swap_gates \
$num_of_process \
$threshold_size \
$virtual_loss \
$num_of_iterations \
$num_of_playout \
$bp_mode \
$model_file_path \
$device \
$num_of_circuit_process \
$inference_batch_size \
$max_batch_size \
$info \

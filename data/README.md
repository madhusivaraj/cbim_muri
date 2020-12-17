## Critical Decision Points dataset

CUDA_VISIBLE_DEVICES='' python main.py --model GRU --aggregation last

CUDA_VISIBLE_DEVICES='' python main.py --model GRU --aggregation average

CUDA_VISIBLE_DEVICES='' python main.py --model GRU --aggregation max


CUDA_VISIBLE_DEVICES='' python main.py --model LSTM --aggregation last

CUDA_VISIBLE_DEVICES='' python main.py --model LSTM --aggregation average


export CUDA_VISIBLE_DEVICES=1


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data kosarak \
       --data-fname ./Streams/kosarak.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 16KB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 31000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data kosarak \
       --data-fname ./Streams/kosarak.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 32KB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 31000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data kosarak \
       --data-fname ./Streams/kosarak.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 64KB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 31000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data kosarak \
       --data-fname ./Streams/kosarak.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 128KB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 31000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data kosarak \
       --data-fname ./Streams/kosarak.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 256KB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 31000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data kosarak \
       --data-fname ./Streams/kosarak.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 512KB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 31000

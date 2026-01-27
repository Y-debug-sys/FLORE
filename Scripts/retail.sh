export CUDA_VISIBLE_DEVICES=1


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data retail \
       --data-fname ./Streams/retail.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 16KB \
       --break-point 1000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 17000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data retail \
       --data-fname ./Streams/retail.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 32KB \
       --break-point 1000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 17000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data retail \
       --data-fname ./Streams/retail.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 64KB \
       --break-point 1000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 17000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data retail \
       --data-fname ./Streams/retail.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 96KB \
       --break-point 1000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 17000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data retail \
       --data-fname ./Streams/retail.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 128KB \
       --break-point 1000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 17000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data retail \
       --data-fname ./Streams/retail.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 256KB \
       --break-point 1000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 17000

export CUDA_VISIBLE_DEVICES=1


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data webdocs \
       --data-fname ./Streams/webdocs.dat \
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
       --stream-size 130000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data webdocs \
       --data-fname ./Streams/webdocs.dat \
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
       --stream-size 130000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data webdocs \
       --data-fname ./Streams/webdocs.dat \
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
       --stream-size 130000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data webdocs \
       --data-fname ./Streams/webdocs.dat \
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
       --stream-size 130000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data webdocs \
       --data-fname ./Streams/webdocs.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 1MB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 130000


python -u main.py \
       --seed 12345 \
       --itrs 5 \
       --data webdocs \
       --data-fname ./Streams/webdocs.dat \
       --window 128 \
       --interval 5000 \
       --key-size 4 \
       --budget 2MB \
       --break-point 2000000 \
       --bsz 32 \
       --dropout 0.0 \
       --lr 0.001 \
       --stream-model real \
       --epochs 25 \
       --stream-size 130000

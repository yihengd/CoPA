python train.py -e 150 \
                -b 32 \
                --workdir ./workdir \
                --model CoPA \
                --dataset PH2 \
                --datapath ./data/PH2 \
                --unique_name train \
                --save_epoch 50 \
                --seed 304

ssh changxu@spartan.hpc.unimelb.edu.au
password=Xc475329647!

sinteractive -p interactive --time=24:00:00 --mem=32G
sinteractive -p interactive --time=24:00:00 --mem=32G --partition=deeplearn -A punim0478 --gres=gpu:v100:2 -q gpgpudeeplearn

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v2,t4v1,rtx6000,a40
#SBATCH -c 40
#SBATCH -q long
#SBATCH --mem=165G
#SBATCH --job-name=strats
#SBATCH --output=strats_output_%j.log
#SBATCH --error=strats_error_%j.log
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00

module load pytorch1.7.1-cuda11.0-python3.6
source ./bin/activate

run_commands(){
    eval template="$1"
    for train_frac in 0.5 0.4 0.3 0.2 0.1; do
        for ((i=1; i<=10; i++)); do
            run_param="${i}o10"
            eval "$1 --run $run_param --train_frac $train_frac"
        done
    done
}

# python -u main.py --pretrain 1 --dataset mimic_iv --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4 --max_epochs 30

template="python -u main.py --dataset mimic_iv --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 --load_ckpt_path ./outputs/mimic_iv/pretrain/checkpoint_best.bin"
run_commands "\${template}"

# template="python -u main.py --dataset mimic_iv --model_type strats --hid_dim 64 --num_layers 2 --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4"
# run_commands "\${template}"

#partition=‘t4v2,t4v1,rtx6000,a40’
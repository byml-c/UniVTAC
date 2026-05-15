task_name=${1}
setting=${2}
expert_data_num=${3}
train_config=${4:-"pi0_fast_franka_univtac_lora"}

POLICY_ROOT=$(cd "$(dirname "$0")" && pwd)
export OPENPI_DATA_HOME=$POLICY_ROOT/openpi

if [ ! -d "/root/.cache/huggingface/lerobot/univtac" ]; then
    mkdir -p /root/.cache/huggingface/lerobot
    # ln -s $POLICY_ROOT/data ~/.cache/huggingface/lerobot/univtac
    mkdir -p /home/robotwin/.cache/huggingface/lerobot/univtac
    ln -s /home/robotwin/.cache/huggingface/lerobot/univtac ~/.cache/huggingface/lerobot/univtac
fi

export CUDA_VISIBLE_DEVICES=0
python scripts/process_data.py $task_name $setting $expert_data_num \
    --config-name $train_config
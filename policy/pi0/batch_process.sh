task_names=(
    grasp_classify
    insert_tube
    insert_HDMI
    insert_hole
    lift_bottle
    lift_can
    pull_out_key
    put_bottle_in_shelf
)
for i in ${task_names[@]}; do
    task_name=$i
    task_config=demo
    expert_data_num=50
    train_config=pi0_fast_franka_univtac_lora_${task_name}-${task_config}-${expert_data_num}
    
    if [ -d "./data/univtac/$task_name-$task_config-$expert_data_num" ]; then
        echo "Processed data for $task_name already exists. Skipping data processing."
    else
        echo "Processing data for $task_name..."
        bash process_data_pi0.sh $task_name $task_config $expert_data_num $train_config
    fi
done
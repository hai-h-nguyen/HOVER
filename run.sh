# ${ISAACLAB_PATH:?}/isaaclab.sh -p neural_wbc/inference_env/scripts/eval.py \
#     --num_envs 1 \
#     --student_path /home/rtx3/khai/HOVER-VR/logs/student/25_03_21_18-25-20_25_03_23_13-04-35 \
#     --student_checkpoint model_20500.pt \
#     --robot_model g1 \
#     --reference_motion_path /home/rtx3/khai/HOVER-VR/neural_wbc/data/data/motions/upper_test_1.pkl

${ISAACLAB_PATH:?}/isaaclab.sh -p neural_wbc/inference_env/scripts/s2r_player.py \
    --student_path  /home/rtx3/khai/HOVER-VR/logs/student/25_03_21_18-25-20_25_03_23_13-04-35 \
    --student_checkpoint model_20500.pt \
    --reference_motion_path neural_wbc/data/data/motions/upper_test_1.pkl \
    --robot unitree_g1 \
    --max_iterations 5000 \
    --num_envs 1 \
    # --headless
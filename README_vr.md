# HOVER VinRobotics

## Training 

Train Teacher Policy

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 4096 \
    --headless \
    --reference_motion_path neural_wbc/data/data/motions/amass_full_g1_anneal.pkl
```

Train Student Policy

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/train_student_policy.py \
    --num_envs 4096 \
    --headless \
    --reference_motion_path neural_wbc/data/data/motions/amass_full_g1_anneal.pkl \
    --teacher_policy.resume_path  <path> \
    --teacher_policy.checkpoint model_<iteration_number>.pt
```

Resume Policy Training

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/train_teacher_policy.py \
    --num_envs 4096 \
    --headless \
    --reference_motion_path neural_wbc/data/data/motions/amass_full_g1_anneal.pkl \
    --teacher_policy.resume \
    --teacher_policy.resume_path  <path> \
    --teacher_policy.checkpoint model_<iteration_number>.pt
```

## Evaluation

Play Teacher Policy

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/play.py \
    --num_envs 10 \
    --reference_motion_path neural_wbc/data/data/motions/amass_full_g1_anneal.pkl \
    --teacher_policy.resume_path <path> \
    --teacher_policy.checkpoint model_<iteration_number>.pt
```

Play Student Policy

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/rsl_rl/play.py \
    --num_envs 10 \
    --reference_motion_path neural_wbc/data/data/motions/amass_full_g1_anneal.pkl \
    --student_player \
    --student_path <path> \
    --student_checkpoint model_<iteration_number>.pt
```

python lerobot/scripts/train.py \  
--policy.path=/home/qwe/wokonsmall/lerobot_diy/lerobot/smolvla_base \
--dataset.repo_id=lerobot/svla_so101_pickplace \
--batch_size=64  \
--steps=20000  \
--output_dir=outputs/train/my_smolvla  \
--job_name=my_smolvla_training \
--policy.device=cuda  \
--wandb.enable=false \
--dataset.root=/home/qwe/.cache/huggingface/lerobot/622work/1
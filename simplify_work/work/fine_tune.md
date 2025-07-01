# 在lerobot目录下

HF_ENDPOINT=https://hf-mirror.com 
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python lerobot/scripts/train.py --policy.path=models/forsmolvla/smolvla_base \
--dataset.repo_id=lerobot/svla_so101_pickplace \
--batch_size=64  \
--steps=20000  \
--output_dir=outputs/train/my_smolvla  \
--job_name=my_smolvla_training \
--policy.device=cuda  \
--policy.repo_id=lerobot/smolvla_base \
--wandb.enable=false \
--dataset.root=training_dataset/622work/1

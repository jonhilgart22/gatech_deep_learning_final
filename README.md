# Adapticons

- Adapting Transformers for all the things

Get started with
`make init_project`
# Launch GLUE training

- Set env vars
`export TASK_NAME=MNLI`

`export GLUE_DIR=/adapticons/modeling`

-  Launch training
```bash
python adapticons/modeling/glue_training.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/$TASK_NAME \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer
```

# TODO:
- get glue script under adapticons running with an adapter

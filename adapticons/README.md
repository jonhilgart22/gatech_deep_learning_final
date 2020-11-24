# Run language modeling with Adapter

```bash
python -m scripts.run_language_modeling --train_data_file datasets/chemprot/train.txt \
                                        --line_by_line \
                                        --eval_data_file=datasets/chemprot/test.txt \
                                        --output_dir roberta-tapt-chemprot-adapter \
                                        --model_type roberta-base \
                                        --tokenizer_name roberta-base \
                                        --mlm \
                                        --per_gpu_train_batch_size 8 \
                                        --gradient_accumulation_steps 16  \
                                        --model_name_or_path roberta-base \
                                        --do_eval \
                                        --evaluate_during_training  \
                                        --do_train \
                                        --num_train_epochs 100  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50 \
                                        --overwrite_output_dir \
                                        --adapter_name="nli/scitail@ukp"
```

## Run evaluation (fine-tuning on target task) w/wo Adapter

#### Baseline(Roberta) evaluation
Below command assumes the name of dataset is one of [chemprot, rct-20k, rct-sample, citation_intent(ACL_ARC),
sciie(SCIERC), ag(AGNEWS), hyperpartisan_news(HYPERPARTISAN), imdb, amazon(helpfulness)]
```bash
python run_glue_alt.py \
  --do_train \
  --do_eval \
  --data_dir datasets/chemprot/ \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir results/baseline/chemprot/ \
  --task_name chemprot \
  --do_predict \
  --load_best_model_at_end \
  --model_name_or_path roberta-base \
```
#### Roberta with a single adapter evaluation
```bash
python run_glue_alt.py \
  --do_train \
  --do_eval \
  --data_dir datasets/hyperpartisan_news/ \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir results/adapter/hyperpartisan_news/ \
  --task_name HYPER \
  --do_predict \
  --load_best_model_at_end \
  --train_adapter \
  --model_name_or_path cks/roberta-tapt-hyper-adapter/ \
  --adapter_config pfeiffer \
```
For more information about arguments,

https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapter_training.py
https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/training_args.py


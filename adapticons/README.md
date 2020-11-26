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
                                        --adapter_name="chemprot_custom_adapter"
```

## Run evaluation (fine-tuning on target task) w/wo Adapter
Experiments were done with the environmental setting in [adapter-hub](https://github.com/Adapter-Hub/adapter-transformers)
(Python 3.6+ and PyTorch 1.1.0+)
See new_train_requirements.text for specific dependencies.

Alternatively, what I did for setting environment is as follows. First create a new conda env, and then,
```bash
git clone https://github.com/adapter-hub/adapter-transformers.git
cd adapter-transformers
pip install .
```
Additionally, sklearn might be needed for evaluation. (pip install sklearn)

#### Baseline(Roberta) evaluation
Below command assumes the name of folder that contains dataset (each split ends with .jsonl) is one of [chemprot, rct-20k, rct-sample, citation_intent(ACL_ARC), sciie(SCIERC), ag(AGNEWS), hyperpartisan_news(HYPERPARTISAN), imdb, amazon(helpfulness)].

Download link for dataset is [here](https://github.com/allenai/dont-stop-pretraining/blob/master/environments/datasets.py). Append train.jsonl, dev.jsonl, test.jsonl after each url.
```bash
python modeling.new_train.py \
  --do_train \
  --do_eval \
  --data_dir datasets/chemprot/ \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir results/baseline/chemprot/ \
  --task_name chemprot \
  --do_predict \
  --load_best_model_at_end \
  --model_name_or_path roberta-base \
  --metric macro \
```

#### Roberta with a single adapter evaluation
```bash
python modeling.new_train.py \
  --do_train \
  --do_eval \
  --data_dir datasets/hyperpartisan_news/ \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
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
  --metric macro \
```

For argument --task_name, new_train.py expect this the same as the task name of adapter which was used when you performed pra-training. Specifically, when you pre-trained adapter, you add_adapter with a particular name such as "sst-2" as below.

model.add_adapter("sst-2", AdapterType.text_task)

This task_name argument would be the same as the above name.

For more information about arguments,

https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapter_training.py
https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/training_args.py

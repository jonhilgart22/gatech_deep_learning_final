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
                                        --num_train_epochs 10  \
                                        --learning_rate 0.0001 \
                                        --logging_steps 50 \
                                        --overwrite_output_dir \
                                        --adapter_name="nli/scitail@ukp"
```

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
'''
baseline fine tuning
python run_glue_alt.py \
  --do_train \
  --do_eval \
  --data_dir datasets/hyperpartisan_news/ \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --output_dir results/test3/adapter/ \
  --task_name HYPER \
  --do_predict \
  --load_best_model_at_end \
  --model_name_or_path roberta-base \



adapter fine tuning
python run_glue_alt.py \
  --do_train \
  --do_eval \
  --data_dir datasets/hyperpartisan_news/ \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --output_dir results/test3/adapter/ \
  --task_name HYPER \
  --do_predict \
  --load_best_model_at_end \
  --train_adapter \
  --model_name_or_path cks/roberta-tapt-hyper-adapter/ \
  --adapter_config pfeiffer \



'''
import argparse
import random
import pickle
import dataclasses
import logging
import os
import sys
import json
import torch
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Optional

import numpy as np

from transformers import (
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelWithHeads,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    PretrainedConfig,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from datasets import load_dataset, load_metric
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )

    data_dir: Optional[str] = field(default=None, metadata={"help": "The input/output data dir for TFDS."})

    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    '''
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    '''

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    '''
    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        #print('num labels: ', num_labels)
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    '''

    data_dir = data_args.data_dir
    label_to_id_ft = {}
    if 'rct' in data_dir:
        label_to_id_ft = {
            "BACKGROUND": 0,
            "OBJECTIVE": 1,
            "METHODS": 2,
            "RESULTS": 3,
            "CONCLUSIONS": 4
        }
        num_labels = 5

    elif 'hyper' in data_dir:
        label_to_id_ft = {
            "false": 0,
            "true": 1
        }
        num_labels = 2

    output_mode = 'classification'
    #print('---')
    #print(num_labels, output_mode)
    #print('---')
    #output_mode = data_args.output_mode
    #raise



    ################################################ inserted ##################################3 from run_glue.py
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    #datasets2 = load_dataset(
    #    "json", data_files={"train": 'datasets/rct-sample/train.jsonl', "validation": 'datasets/rct-sample/train.jsonl'}
    #)

    #with open('data_sample.txt', 'w') as f:
    #    print(datasets, file=f)
    #print('1: ', datasets)


    #if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        #datasets = load_dataset("glue", data_args.task_name)

    #with open('data_sample2.txt', 'w') as f:
    #    print(datasets, file=f)
    #print('2 :', datasets)
    '''
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
             "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    '''
    '''
    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    '''
    ################################################ inserted ##################################3 from run_glue.py









    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.add_classification_head(data_args.task_name, num_labels=num_labels)
    #with open('model_state_keys.txt', 'w') as f:
    #    print(model.state_dict().keys(), file=f)

    # Setup adapters
    task_name = data_args.task_name
    if False: #adapter_args.train_adapter:
        task_name = data_args.task_name
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    AdapterType.text_task,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, AdapterType.text_task, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                AdapterType.text_lang,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None


    #with open('model_state_1.txt', 'w') as f:
    #    print(model.state_dict(), file = f)


    # Freeze all model weights except of those of this adapter
    if adapter_args.train_adapter:
        model.train_adapter(model.config.adapters.adapter_list(AdapterType.text_lang)[0])###model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        model.set_active_adapters(model.config.adapters.adapter_list(AdapterType.text_lang)[0])
        #if lang_adapter_name:
        #    model.set_active_adapters([lang_adapter_name, task_name])
        #else:
        #    model.set_active_adapters([task_name])

    #with open('model_state_2.txt', 'w') as f:
    #    print(model.state_dict(), file = f)








    ###### inserted from run_glue.py ############################################################
    '''
    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    padding = "max_length"
    max_length = data_args.max_seq_length
    #else:
    #    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    #    padding = False
    #    max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    #print('label to id: ', label_to_id)
    #print('label_list: ', label_list)

    def preprocess_function(examples):
        # Tokenize the texts
        with open('example.txt', 'w') as f:
            print(examples, file = f)
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        #print('argrs: ', args)
        with open('after args.txt', 'w') as f:
            print(args, file = f)
        print(padding, max_length)
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    def preprocess_function_ft(examples):
        # Tokenize the texts
        #with open('example.txt', 'w') as f:
        #    print(examples, file = f)
        text = ((examples['text']))
        label = ((examples['label']))
        #print('argrs: ', args)
        #print(*args)
        result = tokenizer(text, truncation=True, padding=padding, max_length=max_length)

        # Map labels to IDs (not necessary for GLUE tasks)
        result["label"] = [label_to_id_ft[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    #raise
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    print('metric: ', metric)
    raise
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    ### inserted from run_glue.py ##########################################################3
    '''

    # compute_metrics
    # Get datasets      ## have to be modified
    #train_dataset = GlueDataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    #eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev") if training_args.do_eval else None
    #test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="test") if training_args.do_predict else None
    '''
    def compute_metrics(p: EvalPrediction) -> Dict:
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)
    '''

    ### load dataset, define compute_metrics ###
    data_dir = data_args.data_dir
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    test_texts = []
    test_labels = []

    for each in ['train', 'dev', 'test']:
        with open(data_dir + each + ".jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if each == 'train':
                    train_texts.append(json.loads(line)['text'])
                    train_labels.append(label_to_id_ft[json.loads(line)['label']])
                elif each == 'dev':
                    val_texts.append(json.loads(line)['text'])
                    val_labels.append(label_to_id_ft[json.loads(line)['label']])
                else:
                    test_texts.append(json.loads(line)['text'])
                    test_labels.append(label_to_id_ft[json.loads(line)['label']])

    train_encodings = tokenizer(train_texts, padding="max_length", max_length=256, truncation=True)
    val_encodings = tokenizer(val_texts, padding="max_length", max_length=256, truncation=True)
    test_encodings = tokenizer(test_texts, padding="max_length", max_length=256, truncation=True)

    class FTDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = FTDataset(train_encodings, train_labels)
    eval_dataset = FTDataset(val_encodings, val_labels)
    test_dataset = FTDataset(test_encodings, test_labels)

    def compute_metrics_ft(pred):
        labels = pred.label_ids
        #print('labels: ', labels)
        preds = pred.predictions.argmax(-1)
        #print('preds: ', preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        #print('f1: ', f1)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    #############################
































































    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_ft,
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
    )

    # Training
    if training_args.do_train:
        # save model before training
        bsw = {}
        for i in model.state_dict():
            bsw[i] = model.state_dict()[i]
        np.save(training_args.output_dir + 'bsm.npy', bsw)

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        # after finishing traning, save keys
        atw = {}
        for i in model.state_dict():
            atw[i] = model.state_dict()[i]
        np.save(training_args.output_dir + 'atm.npy', atw)

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev"))

        for eval_dataset in eval_datasets:
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            #output_eval_file = os.path.join(
            #    training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            #)
            ######
            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{data_args.task_name}.txt")


            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(data_args.task_name))####eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test"))

        for test_dataset in test_datasets:
            test_eval_result = trainer.evaluate(eval_dataset=test_dataset)
            output_test_eval_file = os.path.join(training_args.output_dir, f"test_results_{data_args.task_name}.txt")

            if trainer.is_world_master():
                with open(output_test_eval_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(data_args.task_name))####eval_dataset.args.task_name))
                    for key, value in test_eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            '''
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{data_args.task_name}.txt" ###test_dataset.args.task_name
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(data_args.task_name))####test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            #print('item: ', item)
                            #print('predictions: ', predictions)
                            #print('?: ', test_dataset.__getitem__(item)['labels'].cpu().numpy())
                            #item = test_dataset.get_labels()[item]
                            item = test_dataset.__getitem__(item)['labels'].cpu().numpy()
                            writer.write("%d\t%s\n" % (index, item))
            '''
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

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
import transformers

transformers.logging.set_verbosity_info()

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
    train_fusion: bool = field(default=False, metadata={"help": "whether train adapter fusion model or not."})
    train_adapter_wop: bool = field(default=False, metadata={"help": "whether train adapter without pretraining."})
    fusion_adapter_path1: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path2: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path2: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path3: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path4: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path5: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path6: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path7: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})
    fusion_adapter_path8: Optional[str] = field(default="", metadata={"help": "adapters for fusion"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the task to train on"},
    )

    data_dir: Optional[str] = field(default=None, metadata={"help": "data dir."})

    metric: Optional[str] = field(default="macro", metadata={"help": "evaluation metric."})

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    sanity_check: bool = field(default=False, metadata={"help": "saved mdoels for sanity check."})

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )


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
    data_dir = data_args.data_dir

    # label to id, set same as in don't stop repo
    label_to_id_ft = {}
    if "chemprot" in data_dir:
        label_to_id_ft = {
            "INHIBITOR": 0,
            "SUBSTRATE": 1,
            "INDIRECT-DOWNREGULATOR": 2,
            "INDIRECT-UPREGULATOR": 3,
            "ACTIVATOR": 4,
            "ANTAGONIST": 5,
            "PRODUCT-OF": 6,
            "AGONIST": 7,
            "DOWNREGULATOR": 8,
            "UPREGULATOR": 9,
            "AGONIST-ACTIVATOR": 10,
            "SUBSTRATE_PRODUCT-OF": 11,  ## test set modify needed
            "AGONIST-INHIBITOR": 12,
        }
        num_labels = 13

    elif "rct" in data_dir:
        label_to_id_ft = {"METHODS": 0, "RESULTS": 1, "CONCLUSIONS": 2, "BACKGROUND": 3, "OBJECTIVE": 4}
        num_labels = 5

    elif "citation" in data_dir or "acl" in data_dir:
        label_to_id_ft = {
            "Background": 0,
            "Uses": 1,
            "CompareOrContrast": 2,
            "Motivation": 3,
            "Extends": 4,
            "Future": 5,
        }
        num_labels = 6

    elif "scierc" in data_dir or "sciie" in data_dir:
        label_to_id_ft = {
            "USED-FOR": 0,
            "CONJUNCTION": 1,
            "EVALUATE-FOR": 2,
            "HYPONYM-OF": 3,
            "PART-OF": 4,
            "FEATURE-OF": 5,
            "COMPARE": 6,
        }
        num_labels = 7

    elif "hyper" in data_dir:
        label_to_id_ft = {"false": 0, "true": 1}
        num_labels = 2

    elif "ag" in data_dir or "agnews" in data_dir:
        label_to_id_ft = {1: 0, 2: 1, 3: 2, 4: 3}
        num_labels = 4

    elif "amazon" in data_dir or "helpful" in data_dir:
        label_to_id_ft = {"helpful": 0, "unhelpful": 1}
        num_labels = 2

    elif "imdb" in data_dir:
        label_to_id_ft = {0: 0, 1: 1}
        num_labels = 2

    else:
        assert False, (
            "Data_dir not in [chemprot, rct-20k, rct-sample, citation_intent(ACL_ARC), "
            "sciie(SCIERC), ag(AGNEWS), hyperpartisan_new (HYPERPARTISAN), imdb, amazon(helpfulness)] "
        )

    output_mode = "classification"

    # Load pretrained model and tokenizer
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
    # if data_args.sanity_check:
    #    bsw = {}
    #    for i in model.state_dict():
    #        bsw[i] = model.state_dict()[i]
    #    np.save('after_model_loaded.npy', bsw)  # Just used for sanity check, (500MB)

    if model_args.train_adapter_wop:
        model.add_adapter(data_args.task_name, AdapterType.text_task)
        model.train_adapter([data_args.task_name])
        model.add_classification_head(data_args.task_name, num_labels=num_labels)
        model.set_active_adapters([[data_args.task_name]])

    elif adapter_args.train_adapter:
        model.train_adapter(model.config.adapters.adapter_list(AdapterType.text_lang)[0])  ###model.train_adapter([task_name])
        model.add_classification_head(model.config.adapters.adapter_list(AdapterType.text_lang)[0], num_labels=num_labels)
        # Set the adapters to be used in every forward pass
        model.set_active_adapters(model.config.adapters.adapter_list(AdapterType.text_lang)[0])

    elif model_args.train_fusion:
        fusion_path = []
        fusion_path.append(model_args.fusion_adapter_path1)
        fusion_path.append(model_args.fusion_adapter_path2)
        fusion_path.append(model_args.fusion_adapter_path3)
        fusion_path.append(model_args.fusion_adapter_path4)
        fusion_path.append(model_args.fusion_adapter_path5)
        fusion_path.append(model_args.fusion_adapter_path6)
        fusion_path.append(model_args.fusion_adapter_path7)
        fusion_path.append(model_args.fusion_adapter_path8)
        while "" in fusion_path:
            fusion_path.remove("")

        from transformers.adapter_config import PfeifferConfig

        for each in fusion_path:
            model.load_adapter(each, "text_lang", config=PfeifferConfig(), with_head=False)

        ADAPTER_SETUP = [
            [
                list(model.config.adapters.adapters.keys())[i]
                for i in range(len(list(model.config.adapters.adapters.keys())))
            ]
        ]

        # Add a fusion layer and tell the model to train fusion
        logger.info(f"Using adapter fusion with the following setup {ADAPTER_SETUP}")
        logger.info(f"Model adapters = {ADAPTER_SETUP}")
        model.add_fusion(ADAPTER_SETUP[0], "dynamic")
        model.train_fusion(ADAPTER_SETUP)
        model.add_classification_head(data_args.task_name, num_labels=num_labels)

    else:
        model.add_classification_head(data_args.task_name, num_labels=num_labels)

    # if data_args.sanity_check:
    #    bsw = {}
    #    for i in model.state_dict():
    #        bsw[i] = model.state_dict()[i]
    #    np.save('after_add_heads.npy', bsw) # Just used for sanity check, (500MB)

    # if data_args.sanity_check:
    #    bsw = {}
    #    for i in model.state_dict():
    #        bsw[i] = model.state_dict()[i]
    #    np.save('after_fusion_merged.npy', bsw)  # Just used for sanity check, (500MB)

    ### load dataset, define compute_metrics ###
    data_dir = data_args.data_dir
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    test_texts = []
    test_labels = []

    if data_dir[-1] != "/":
        data_dir += "/"

    for each in ["train", "dev", "test"]:
        with open(data_dir + each + ".jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if each == "train":
                    train_texts.append(json.loads(line)["text"])
                    train_labels.append(label_to_id_ft[json.loads(line)["label"]])
                elif each == "dev":
                    val_texts.append(json.loads(line)["text"])
                    val_labels.append(label_to_id_ft[json.loads(line)["label"]])
                else:
                    test_texts.append(json.loads(line)["text"])
                    test_labels.append(label_to_id_ft[json.loads(line)["label"]])

    train_encodings = tokenizer(train_texts, padding="max_length", max_length=512, truncation=True)
    val_encodings = tokenizer(val_texts, padding="max_length", max_length=512, truncation=True)
    test_encodings = tokenizer(test_texts, padding="max_length", max_length=512, truncation=True)

    class FTDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = FTDataset(train_encodings, train_labels)
    eval_dataset = FTDataset(val_encodings, val_labels)
    test_dataset = FTDataset(test_encodings, test_labels)

    def compute_metrics_ft(pred):
        # print('pred: ', pred)
        labels = pred.label_ids
        # print('labels: ', labels)
        preds = pred.predictions.argmax(-1)
        # print('preds: ', preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=data_args.metric, labels=[i for i in range(num_labels)]
        )
        # print('f1: ', f1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    #############################

    # Initialize our Trainer
    if not adapter_args.train_adapter and not model_args.train_fusion:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_ft,
            do_save_full_model=True,
        )
    else:
        save_full = False
        if adapter_args.train_adapter:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_ft,
                do_save_full_model=save_full,
                do_save_adapters=adapter_args.train_adapter,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_ft,
                do_save_full_model=save_full,
                do_save_adapter_fusion=True,
            )

    # Training
    if training_args.do_train:
        # save model before training
        if data_args.sanity_check:
            bsw = {}
            for i in model.state_dict():
                bsw[i] = model.state_dict()[i]
            np.save(training_args.output_dir + "before_training.npy", bsw)  # Just used for sanity check, (500MB)

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        # after finishing traning, save keys
        if data_args.sanity_check:
            bsw = {}
            for i in model.state_dict():
                bsw[i] = model.state_dict()[i]
            np.save(training_args.output_dir + "after_training.npy", bsw)  # Just used for sanity check, (500MB)

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_datasets = [eval_dataset]
        for eval_dataset in eval_datasets:
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{data_args.task_name}.txt")

            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info(
                        "***** Eval results {} *****".format(data_args.task_name)
                    )  ####eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            test_eval_result = trainer.evaluate(eval_dataset=test_dataset)
            output_test_eval_file = os.path.join(training_args.output_dir, f"test_results_{data_args.task_name}.txt")

            if trainer.is_world_master():
                with open(output_test_eval_file, "w") as writer:
                    logger.info(
                        "***** Test results {} *****".format(data_args.task_name)
                    )  ####eval_dataset.args.task_name))
                    for key, value in test_eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
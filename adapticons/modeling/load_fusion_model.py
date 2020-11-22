import argparse
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AdapterConfig,
    AdapterType,
)
from transformers.adapter_config import PfeifferConfig
import logging
import os
import torch

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--use_adapter_fusion",
        type=lambda x: (str(x).lower() == "false"),
        default=True,
        help="Train with adapter fusion. Default True.",
    )
    parser.add_argument(
        "--adapter_config_name", type=str, default="pfeiffer", help="adapter config name.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument("--adapter_reduction_factor", type=int, default=12, help="adapter reduction factor.")
    parser.add_argument("--adapter_one", type=str, help="location of first adapter path")
    parser.add_argument("--adapter_two", type=str, help="location of second adapter path")
    parser.add_argument("--adapter_three", type=str, help="location of third adapter path")
    parser.add_argument("--adapter_four", type=str, help="location of fourth adapter path")
    parser.add_argument("--adapter_five", type=str, help="location of fifth adapter path")
    parser.add_argument("--adapter_six", type=str, help="location of six adapter path")
    args = parser.parse_args()
    return args


def save_model(model_to_save, tokenizer, args):
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # save adapter

    for name in list(model_to_save.config.adapters.adapters.keys()):
        model_to_save.save_adapter(args.output_dir, name)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


def main():
    args = parse_args()

    adapter_config = AdapterConfig.load(args.adapter_config_name, reduction_factor=args.adapter_reduction_factor)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    # ~~~~~ Here comes the interesting part of setting up AdapterFusion training ~~~~~

    if not args.adapter_one:
        raise ValueError("You need to pass a directory path to at least one adapter.")
    logger.info("Using adapter fusion")
    # First, load the pre-trained adapters we want to fuse from Hub
    print("loading adapter fusion")
    model.load_adapter(args.adapter_one, "text_lang", config=adapter_config)
    if args.adapter_two is not None:
        model.load_adapter(args.adapter_two, "text_lang", config=adapter_config)
    if args.adapter_three is not None:
        model.load_adapter(args.adapter_three, "text_lang", config=adapter_config)
    if args.adapter_four is not None:
        model.load_adapter(args.adapter_four, "text_lang", config=adapter_config)
    if args.adapter_five is not None:
        model.load_adapter(args.adapter_five, "text_lang", config=adapter_config)
    if args.adapter_six is not None:

        model.load_adapter(args.adapter_six, "text_lang", config=adapter_config)

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

    os.makedirs(args.output_dir, exist_ok=True)

    save_model(model, tokenizer, args)


if __name__ == "__main__":
    main()

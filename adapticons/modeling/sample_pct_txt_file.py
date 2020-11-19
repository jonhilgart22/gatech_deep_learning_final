import argparse
import json
from tqdm import tqdm
import numpy as np


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name")
    parser.add_argument("--sample_pct")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    try:
        sample_pct = float(args.sample_pct)
    except Exception as e:
        print(e)

    if sample_pct > 1:
        sample_pct = sample_pct * 0.01

    print(args)

    new_txt_lines = []
    outF = open(f"{args.file_name.split('.')[0]}_{args.sample_pct.split('.')[-1]}_pct.txt", "w")
    f = open(args.file_name, "r")

    for x in tqdm(f):
        if np.random.rand() <= sample_pct:  # take this line
            outF.write(x)
    outF.close()
    f.close()


if __name__ == "__main__":
    main()

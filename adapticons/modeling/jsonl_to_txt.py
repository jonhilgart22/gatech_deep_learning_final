import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--file_name")
args = parser.parse_args()
print(args)

new_txt_lines = []
outF = open(f"{args.file_name.split('.')[0]}.txt", "w")
f = open(args.file_name, "r")

for x in tqdm(f):
    outF.write(json.loads(x)["text"])
    outF.write("\n")

outF.close()
f.close()

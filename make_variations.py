import random
import shutil
import math
import os
import subprocess
from make_sub_stat import substitution_freq

# FILE = "scop_sf_represeq_lib_latest.fa"
FILE = "data/train.fa"
PERCENT_MOD = 5
NAME = f"train_{PERCENT_MOD}_indel_sub_bert"
MODIFIED_OUT = f"data/{NAME}.fa"
NUM_VARIATIONS = 16


def make_sub(seq):
    pos = random.randrange(len(seq))
    sub_freq = substitution_freq[seq[pos]]
    seq[pos] = sub_freq[random.randrange(len(sub_freq))]
    return seq


def make_deletion(seq):
    pos = random.randrange(len(seq))
    return seq[:pos] + seq[pos + 1 :]

def make_insertions(seq):
    pos = random.randrange(len(seq))
    aa = random.choice(list(substitution_freq.keys()))
    return seq[:pos] + [aa] + seq[pos:]

def write_variations(header, seq, out_f):
    # TODO, one data modifacation could possibly undo another data modifacation
    # deal with old data
    cleaned_seq = ""
    for aa in seq:
        if aa not in "ACDEFGHIKLMNPQRSTVWY":
            aa = "X"
        cleaned_seq += aa
    print(f"{header}|var0", file=out_f)
    print(cleaned_seq, file=out_f)

    for i in range(1, NUM_VARIATIONS):
        var_seq = list(cleaned_seq)
        orig_len = len(var_seq)
        for _ in range(round(orig_len * PERCENT_MOD/300.0)):
            var_seq = make_deletion(var_seq)

        for _ in range(round(orig_len * PERCENT_MOD/300.0)):
            var_seq = make_sub(var_seq)

        for _ in range(round(orig_len * PERCENT_MOD/300.0)):
            var_seq = make_insertions(var_seq)

        # we should probably make sure the final step doesn't make two copies of the same seq
        # if we are running millions of sequences, this could slow down data augmenation though
        # while pair2 == pair1:
        var_seq = "".join(var_seq)
        print(f"{header}|var{i}", file=out_f)
        print(var_seq, file=out_f)


def main():
    header = ""
    seq = ""
    with open(FILE, "r") as in_f, open(MODIFIED_OUT, "w") as out_f:
        for line in in_f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    write_variations(header, seq, out_f)
                header = line
                seq = ""
            else:
                seq += line
        # get the last one
        write_variations(header, seq, out_f)

    try:
        os.mkdir(NAME)
    except FileExistsError:
        shutil.rmtree(NAME)
        os.mkdir(NAME)


    from transformers import BertModel, BertTokenizer
    import re
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    sequence_Example = "A E T C Z A O"
    sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    output = model(**encoded_input)
    # subprocess.run(
    #     [
    #         "python",
    #         "/home/tcoard/w/650_proj/esm/extract.py",
    #         "esm1_t34_670M_UR50S",
    #         f"data/{NAME}.fa",
    #         NAME,
    #         "--repr_layers",
    #         "34",
    #         "--include",
    #         "per_tok",
    #     ]
    # )


if __name__ == "__main__":
    main()

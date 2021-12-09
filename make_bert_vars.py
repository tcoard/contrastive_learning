import random
import torch
import shutil
import math
import os
import subprocess
from make_sub_stat import substitution_freq
from transformers import BertModel, BertTokenizer, pipeline
import re

# FILE = "scop_sf_represeq_lib_latest.fa"
FILE = "data/train.fa"
PERCENT_MOD = 30
NAME = f"train_{PERCENT_MOD}_indel_sub_bert_new"
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

def write_variations(header, seq, out_f, fe):
    # TODO, one data modifacation could possibly undo another data modifacation
    # deal with old data
    cleaned_seq = ""
    for aa in seq:
        if aa not in "ACDEFGHIKLMNPQRSTVWY":
            aa = "X"
        cleaned_seq += aa

    var_seqs = [" ".join(list(cleaned_seq))]

    # emb = fe(" ".join(list(cleaned_seq)))
    # seq_emb = emb[0][1:-1]

    # tensor_emb = torch.empty((len(seq_emb), 1024))
    # for aa_idx, aa in enumerate(seq_emb):
    #     for val_idx, val in enumerate(aa):
    #         tensor_emb[aa_idx][val_idx] = val
    # torch.save(tensor_emb, f"{NAME}/{header[1:]}|var0.pt")

    # encoded_input = tokenizer(" ".join(list(cleaned_seq)), return_tensors='pt')
    # output = model(**encoded_input)
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

        var_seqs.append(" ".join(var_seq))
        # we should probably make sure the final step doesn't make two copies of the same seq
        # if we are running millions of sequences, this could slow down data augmenation though
        # while pair2 == pair1:



        # emb = fe(" ".join(var_seq))
        # seq_emb = emb[0][1:-1]

        # tensor_emb = torch.empty((len(seq_emb), 1024))
        # for aa_idx, aa in enumerate(seq_emb):
        #     for val_idx, val in enumerate(aa):
        #         tensor_emb[aa_idx][val_idx] = val
        # torch.save(tensor_emb, f"{NAME}/{header[1:]}|var{i}.pt")


        var_seq = "".join(var_seq)
        print(f"{header}|var{i}", file=out_f)
        print(var_seq, file=out_f)

    embs = fe(var_seqs)
    for i, emb in enumerate(embs):
        seq_emb = emb[1:-1]
        # tensor_emb = torch.empty((len(seq_emb), 1024))
        # for aa_idx, aa in enumerate(seq_emb):
        #     for val_idx, val in enumerate(aa):
        #         tensor_emb[aa_idx][val_idx] = val
        tensor_emb = torch.tensor(seq_emb)
        torch.save(tensor_emb, f"{NAME}/{header[1:]}|var{i}.pt")


def main():
    header = ""
    seq = ""

    try:
        os.mkdir(NAME)
    except FileExistsError:
        shutil.rmtree(NAME)
        os.mkdir(NAME)

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model.to("cuda")
    model.eval()
    fe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=0)

    with open(FILE, "r") as in_f, open(MODIFIED_OUT, "w") as out_f:
        for line in in_f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    write_variations(header, seq, out_f, fe)
                header = line
                seq = ""
            else:
                seq += line
        # get the last one
        write_variations(header, seq, out_f, fe)



    # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    # model = BertModel.from_pretrained("Rostlab/prot_bert")
    # sequence_Example = "A E T C Z A O"
    # sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
    # encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    # output = model(**encoded_input)
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

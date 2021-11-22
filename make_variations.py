import random
from make_sub_stat import substitution_freq

#FILE = "scop_sf_represeq_lib_latest.fa"
FILE = "data/esm40.fa"
MODIFIED_OUT = "data/modified.fa"

def make_sub(seq):
    pos=random.randrange(len(seq))
    sub_freq = substitution_freq[seq[pos]]
    seq[pos] = sub_freq[random.randrange(len(sub_freq))]
    return seq

def make_deletion(seq):
    pos=random.randrange(len(seq))
    return seq[:pos]+seq[pos+1:]

def write_pairs(header, seq, out_f):
    # TODO, one data modifacation could possibly undo another data modifacation
    # deal with old data
    new_seq = ""
    for aa in seq:
        if aa not in "ACDEFGHIKLMNPQRSTVWY":
            aa = "X"
        new_seq += aa

    pair1 = pair2 = new_seq
    pair1 = list(pair1)
    pair2 = list(pair2)
    for _ in range(2):
        pair1 = make_deletion(pair1)
        pair2 = make_deletion(pair2)

    for _ in range(5):
        pair1 = make_sub(pair1)
        pair2 = make_sub(pair2)
        # we should probably make sure the final step doesn't make two copies of the same seq
        # if we are running millions of sequences, this could slow down data augmenation though
        # while pair2 == pair1:
    pair1 = ''.join(pair1)
    pair2 = ''.join(pair2)
    print(f"{header}|pair1", file=out_f)
    print(pair1, file=out_f)
    print(f"{header}|pair2", file=out_f)
    print(pair2, file=out_f)

def main():
    header = ""
    seq = ""
    with open(FILE, "r") as in_f, open(MODIFIED_OUT, "w") as out_f:
        for line in in_f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    write_pairs(header, seq, out_f)
                header = line
                seq = ""
            else:
                seq += line

        write_pairs(header, seq, out_f)


if __name__ == "__main__":
    main()

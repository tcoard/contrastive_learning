import random
from make_sub_stat import substitution_freq

#FILE = "scop_sf_represeq_lib_latest.fa"
FILE = "data/esm40.fa"
MODIFIED_OUT = "data/modified.fa"
NUM_VARIATIONS = 16

def make_sub(seq):
    pos=random.randrange(len(seq))
    sub_freq = substitution_freq[seq[pos]]
    seq[pos] = sub_freq[random.randrange(len(sub_freq))]
    return seq

def make_deletion(seq):
    pos=random.randrange(len(seq))
    return seq[:pos]+seq[pos+1:]

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
        for _ in range(2):
            var_seq = make_deletion(var_seq)

        for _ in range(5):
            var_seq = make_sub(var_seq)
            # we should probably make sure the final step doesn't make two copies of the same seq
            # if we are running millions of sequences, this could slow down data augmenation though
            # while pair2 == pair1:
        var_seq = ''.join(var_seq)
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

        write_variations(header, seq, out_f)


if __name__ == "__main__":
    main()

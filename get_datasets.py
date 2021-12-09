import pickle
import random
from collections import Counter

MAX_AA_LEN = 1000
BATCH_SIZE = 50
# SAMPLE_PER_SEQ = 2

def get_seq_names(old_seqs=None, run_type="test", batch_size=BATCH_SIZE):
    if old_seqs is None:
        old_seqs = []
    else:
        old_seqs = [j for i in old_seqs for j in i]
        old_seqs = ["|".join(seq.split("|")[0:-1]) for seq in old_seqs]
    seqs = []
    total_types = []
    with open("data/esm40.fa", "r") as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            if line.startswith(">"):
                line = line.strip()
                header = line[1:]
                # this one was too large
                if header == "gi|995950590|BETA-LACTAM":
                    continue
                res = header.split("|")[2]
                if Counter(total_types)[res] < batch_size and header not in old_seqs:
                    # seqs.append([seq.split('/')[1] for seq in glob.glob(f"test_embeddings/{header}*")])
                    # seqs.append(random.sample(glob.glob(f"processed_embeddings/{header}*"), SAMPLE_PER_SEQ))
                    # seqs.append(glob.glob(f"{header}*"))
                    seqs.append(header)
                    total_types.append(res)
    # seqs = random.choices(seqs, k=100)
    with open(f"data/{run_type}.pkl", "wb") as f:
        pickle.dump(seqs, f)
    return seqs

def print_fasta(seqs, run_type):
    with open("data/esm40.fa", "r") as f, open(f"data/{run_type}.fa", 'w') as out:
        get_aa = False
        for line in f:
            if line.startswith(">"):
                line = line.strip()
                header = line[1:]
                if header in seqs:
                    print(line, file=out)
                    get_aa = True
            elif get_aa:
                get_aa = False
                line = line.strip()
                print(line, file=out)
def main():
    train_seqs = []
    train_seqs = get_seq_names(run_type="train")
    if not train_seqs:
        with open("seqs_train.pkl", "rb") as f:
            train_seqs = pickle.load(f)

    print_fasta(train_seqs, "train")

    test_seqs = get_seq_names(train_seqs, run_type="test", batch_size=15)

    print_fasta(test_seqs, "test")

if __name__ == "__main__":
    main()


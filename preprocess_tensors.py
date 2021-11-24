import glob
import math
import torch


class TooLong(Exception):
    pass

MAX_AA_LEN = 1000

def pad(tensor):
    aa_len, _ = tensor.shape

    if aa_len > MAX_AA_LEN:
        raise TooLong

    aa_len_diff = float(MAX_AA_LEN - aa_len)
    pad_amount = (0, 0, math.ceil(aa_len_diff / 2), math.floor(aa_len_diff / 2))
    return torch.nn.functional.pad(tensor, pad_amount, "constant", 0)

def main():
    embedding_paths = glob.glob(f"test_embeddings/*")
    for path in embedding_paths:
        name = path.split('/')[1]
        try:
            emb = torch.load(path)["representations"][34]
            emb = pad(emb)
        except TooLong:
            print(name)
            continue
        torch.save(emb, f"processed_embeddings/{name}")


if __name__ == "__main__":
    main()


import glob
import pickle
import math
from PIL import Image
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
from multiprocessing import Pool


class TooLong(Exception):
    pass

MAX_AA_LEN = 1000

def pad(tensor):
    aa_len, _ = tensor.shape

    if aa_len > MAX_AA_LEN:
        raise TooLong

    aa_len_diff = float(MAX_AA_LEN - aa_len)
    pad_amount = (0, 0, math.ceil(aa_len_diff / 2), math.floor(aa_len_diff / 2))
    b = torch.nn.functional.pad(tensor, pad_amount, "constant", 0)
    return b

def tensor_to_image(path):
    if not path.endswith("var0.pt"):
        return
    name = path.split('/')[1]
    try:
        emb = torch.load(path)["representations"][34]
        emb = torch.from_numpy(minmax_scale(emb))
        emb = pad(emb).numpy()
        emb = (emb * 255).astype(np.uint8)
        # emb = emb[:1000]
        # emb = emb.numpy()
    except TooLong:
        print(name)
        return

    img = Image.fromarray(emb)       # Create a PIL image

    # img = img.convert('RGB')
    drug = name.split('|')[-2]
    name = ''.join(name.split('.')[:-1])
    # img.save(f"new_images/{name}.png", "PNG")
    img.save(f"test_images/{drug}/{name}.png", "PNG")


def main():
    with open("pairs_test.pkl", 'rb') as f:
        embedding_paths = pickle.load(f)
    
    # embedding_paths = glob.glob(f"test_embeddings/*")
    # list(map(tensor_to_image, [embedding_paths[0]]))
    with Pool() as pool:
        pool.map(tensor_to_image, [f"all_embeddings/{j.split('/')[1]}" for i in embedding_paths for j in i])
        # pool.map(tensor_to_image, [j for i in embedding_paths for j in i])
    # list(map(tensor_to_image, [f"all_embeddings/{j.split('/')[1]}" for i in embedding_paths for j in i]))



        # torch.save(emb, f"processed_embeddings/{name}")


if __name__ == "__main__":
    main()


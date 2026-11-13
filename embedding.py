import os
import torch
import argparse
import subprocess
from evo2 import Evo2
from tqdm import tqdm

# Use only the GPU and specify the card number，set GPU (0, 1, 2, 3)
GPU_ID = 0
if not torch.cuda.is_available():
    raise EnvironmentError("GPU not available! This script requires a CUDA device.")
device = f"cuda:{GPU_ID}"
torch.cuda.set_device(device)
print(f" Using device: {device}")

def preprocess_fasta(fasta_path, step=250, window=1024, save_txt=None):
    """
    Use seqkit to process FASTA files and generate fixed‑length sequences
    Args:
        fasta_path: FASTA files path
        step: sliding step size
        window: window size
        save_txt: save sequences to this text file (optional)
    """
    cmd = f"""seqkit sliding -s {step} -W {window} {fasta_path} | seqkit seq -m {window} -M {window} -w {window} | sed '/>/d'"""

    print(f"Running seqkit preprocessing...")

    # Open file for writing if save_txt is provided
    txt_file = None
    if save_txt:
        txt_file = open(save_txt, 'w')
        print(f"Also saving sequences to: {save_txt}")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    seq_count = 0
    for line in process.stdout:
        seq = line.strip()
        if seq:
            seq_count += 1
            # Write to file if save_txt is provided
            if txt_file:
                txt_file.write(seq + '\n')
            yield seq

    process.wait()

    # Close file if opened
    if txt_file:
        txt_file.close()
        print(f"Saved {seq_count} sequences to {save_txt}")

    if process.returncode != 0:
        error = process.stderr.read()
        raise RuntimeError(f"seqkit failed: {error}")

    print(f"Preprocessing complete. Generated {seq_count} sequences.")

def evo2_embeddings(fasta_path, label=None, out_path="./evo2_embeddings.pt", layer="blocks.24", step=1000, window=1024, save_txt=None):
    """
    Extract embeddings from FASTA sequences using Evo2 model
    Args:
        fasta_path: Input FASTA file
        label: Optional label for all sequences (if None, no labels are saved)
        out_path: Output file path
        layer: Model layer to extract embeddings from
        step: Sliding window step size
        window: Window size for sequence processing
        save_txt: save sequences to this text file (optional)
    """
    model = Evo2("evo2_7b")
    emb_list = []  # Store embeddings

    seq_generator = preprocess_fasta(fasta_path, step, window, save_txt=save_txt)

    for i, seq in enumerate(tqdm(seq_generator, desc=f"Extracting on {device}")):
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, emb = model(ids, return_embeddings=True, layer_names=[layer])
            e = emb[layer]

        if e.shape != (1, window, 4096):
            raise ValueError(f"[{i}] shape {e.shape} != (1,{window},4096)")

        pooled = e.mean(dim=1).squeeze(0).cpu()
        if pooled.shape != (4096,):
            raise ValueError(f"[{i}] pooled shape {pooled.shape} != (4096,)")

        emb_list.append(pooled)

        if (i + 1) % 5 == 0:
            print(f"[Check {i+1}] OK {len(emb_list)}x4096")

    embeddings = torch.stack(emb_list)

    # Save with or without labels based on input
    if label is not None:
        # Create labels tensor (same label for all sequences)
        labels = torch.full((embeddings.shape[0],), int(label), dtype=torch.long)
        torch.save({"embeddings": embeddings, "labels": labels}, out_path)
        print(f"Done! Saved {embeddings.shape} with labels={label} to {out_path}")
    else:
        # Save only embeddings (no labels)
        torch.save({"embeddings": embeddings}, out_path)
        print(f"Done! Saved {embeddings.shape} (no labels) to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Evo2 embeddings from FASTA sequences")
    p.add_argument("--fasta", required=True, help="Input FASTA file")
    p.add_argument("--label", type=int, default=None, help="Label for all sequences (optional)")
    p.add_argument("--step", type=int, default=250, help="Sliding window step size")
    p.add_argument("--window", type=int, default=1024, help="Sliding window size")
    p.add_argument("--out", default="./evo2_embeddings.pt", help="Output embeddings file path")
    p.add_argument("--layer", default="blocks.24", help="Model layer to extract embeddings from")
    p.add_argument("--save_txt", default=None, help="Save processed sequences to text file (optional)")
    args = p.parse_args()

    evo2_embeddings(
        args.fasta,
        label=args.label,
        out_path=args.out,
        layer=args.layer,
        step=args.step,
        window=args.window,
        save_txt=args.save_txt
    )

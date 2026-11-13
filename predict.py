import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dcgp.model import dcgp_class
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    matthews_corrcoef, average_precision_score)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EmbeddingDataset(Dataset):
    """
    data_file:
      - "embeddings": float32 Tensor [N, D]
      - "labels": int64 Tensor [N] (0/1) or not labels
    """
    def __init__(self, data_file: str):
        super().__init__()
        data = torch.load(data_file)
        self.emb = data["embeddings"]
        self.lab = data.get("labels")

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx):
        if self.lab is None:
            return self.emb[idx], -1
        return self.emb[idx], self.lab[idx]

def predict(data_file: str, model_path: str, batch_size: int = 32, out: str = "./predict_out"):
    os.makedirs(out, exist_ok=True)
    set_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = EmbeddingDataset(data_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = dcgp_class(input_dim=dataset.emb.shape[1]).to(device)
    state = torch.load(model_path, map_location=device)

    # BF16→FP32
    for k, v in state.items():
        state[k] = v.to(torch.float32)
    model.load_state_dict(state)
    model = model.to(torch.float32)
    model.eval()

    preds_prob, preds_label, trues = [], [], []
    with torch.no_grad():
        for emb, label in loader:
            emb = emb.to(device).to(torch.float32)  # BF16→FP32
            logits = model(emb)
            prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            preds_prob.extend(prob)
            preds_label.extend((prob > 0.5).astype(int))
            trues.extend(label.cpu().tolist())

    result_df = pd.DataFrame({
        "true_label": trues,
        "pred_prob": preds_prob,
        "pred_label": preds_label
    })
    result_df.to_csv(os.path.join(out, "predict_result.csv"), index=False)

    ##  If there are no tables, do not compute any performance metrics
    # if -1 not in trues:
    #     metrics = {
    #         "ACC": accuracy_score(trues, preds_label),
    #         "F1": f1_score(trues, preds_label),
    #         "MCC": matthews_corrcoef(trues, preds_label),
    #         "AUROC": roc_auc_score(trues, preds_prob),
    #         "AUPRC": average_precision_score(trues, preds_prob),
    #     }
    #     pd.DataFrame([metrics]).to_csv(os.path.join(out, "predict_metrics.csv"), index=False)
    #     print("\n Prediction Finished. Metrics:")
    #     print(metrics)
    # else:
    print("\n Prediction Finished")

def parse_args():
    p = argparse.ArgumentParser(description="Predict using trained dcgp model")
    p.add_argument("--data", required=True, help="The data after embedding, in .pt format")
    p.add_argument("--model", required=True, help="The path of trained model .pt")
    p.add_argument("--out", default="./predict_out", help="Output directory")
    p.add_argument("--batch_size", type=int, default=256, help="Data batch size")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    predict(args.data
    , args.model
    , out=args.out
    , batch_size=args.batch_size)

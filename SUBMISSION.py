import os
import gc
import json
import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, logging as hf_logging
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# -------------------------
# Basic config & warnings
# -------------------------
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# -------------------------
# Configuration
# -------------------------
CFG = {
    "MODEL_NAME": "sentence-transformers/all-mpnet-base-v2",
    "MAX_TOKENS": 512,
    "BATCH": 16,
    "EPOCHS": 10,
    "LR_ENCODER": 2e-6,
    "LR_HEAD": 2e-5,
    "FOLDS": 5,
    "SEED": 42,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # GMM oversampling params
    "GMM_TARGET_PER_CLASS": 500,   # desired samples per class after augmentation (tuneable)
    "GMM_MAX_COMPONENTS": 3,       # max mixture components to try for GMM
    "GMM_MAX_COPIES": 10          # if class too small, allow copying each sample up to this many times
}

EMBED_DIM = 768
INTERACT_DIM = EMBED_DIM * 4
NUM_CLASSES = 11

# -------------------------
# Determinism helper
# -------------------------
def fix_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed(CFG["SEED"])
print("Device:", CFG["DEVICE"])

# -------------------------
# Expected score helper
# -------------------------
def expected_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    values = torch.arange(0, NUM_CLASSES, dtype=torch.float).to(probs.device)
    return (probs * values).sum(dim=1)

# -------------------------
# Load & preprocess
# -------------------------
def load_training_files(train_json_path="train_data.json",
                        metric_names_path="metric_names.json",
                        metric_emb_path="metric_name_embeddings.npy") -> Tuple[pd.DataFrame, dict]:
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"{train_json_path} not found.")
    if not os.path.exists(metric_names_path):
        raise FileNotFoundError(f"{metric_names_path} not found.")
    if not os.path.exists(metric_emb_path):
        raise FileNotFoundError(f"{metric_emb_path} not found.")
    with open(train_json_path, "r") as fh:
        train_json = json.load(fh)
    df = pd.DataFrame(train_json)
    with open(metric_names_path, "r") as fh:
        mnames = json.load(fh)
    emb = np.load(metric_emb_path)
    metric_map = dict(zip(mnames, emb))
    return df, metric_map

def prepare_dataframe(df: pd.DataFrame, metric_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_prompt"] = df["user_prompt"].astype(str)
    df["system_prompt"] = df["system_prompt"].fillna("None").astype(str)
    df["response"] = df["response"].astype(str)
    df["full_text"] = (
        "User: " + df["user_prompt"] +
        " | System: " + df["system_prompt"] +
        " | Response: " + df["response"]
    )
    df["metric_embedding"] = df["metric_name"].map(metric_map)
    df["score_float"] = df["score"].astype(float)
    df["score_label"] = df["score_float"].astype(int)
    return df

# -------------------------
# GMM oversampling (metric embeddings only)
# -------------------------
def gmm_augment_dataset(df_all: pd.DataFrame,
                        target_per_class: int,
                        max_components: int = 3,
                        max_copies: int = 10,
                        random_state: int = 0) -> pd.DataFrame:
    """
    Apply GMM-based augmentation + copies to whole dataset (before CV).
    Returns augmented DataFrame.
    """
    np.random.seed(random_state)
    df = df_all.copy().reset_index(drop=True)
    added_rows = []

    for cls in range(NUM_CLASSES):
        cls_df = df[df["score_label"] == cls]
        n_real = len(cls_df)
        if n_real == 0:
            continue
        if n_real >= target_per_class:
            continue

        needed = int(target_per_class - n_real)

        emb_list = cls_df["metric_embedding"].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(EMBED_DIM))
        emb_matrix = np.stack(emb_list.values)

        # If only 1 sample, jitter copies
        if emb_matrix.shape[0] == 1:
            base_row = cls_df.iloc[0]
            copies_done = 0
            while copies_done < needed:
                new_row = base_row.copy()
                if copies_done < max_copies:
                    jitter = np.random.normal(scale=1e-3, size=EMBED_DIM)
                    new_row["metric_embedding"] = emb_matrix[0] + jitter
                else:
                    new_row["metric_embedding"] = emb_matrix[0].copy()
                new_row["score_label"] = cls
                new_row["score_float"] = float(cls)
                added_rows.append(new_row)
                copies_done += 1
            continue

        # Fit GMM (try decreasing components if fails)
        n_components = min(max_components, emb_matrix.shape[0])
        fitted = None
        for comp in range(n_components, 0, -1):
            try:
                gmm = GaussianMixture(n_components=comp, covariance_type="full", random_state=(random_state + cls))
                gmm.fit(emb_matrix)
                fitted = gmm
                break
            except Exception:
                fitted = None
                continue

        if fitted is None:
            # fallback SMOTE-like interpolation
            base = cls_df.reset_index(drop=True)
            for _ in range(needed):
                i = np.random.randint(0, len(base))
                j = np.random.randint(0, len(base))
                if i == j:
                    j = (i + 1) % len(base)
                v1 = base.loc[i, "metric_embedding"]
                v2 = base.loc[j, "metric_embedding"]
                if not isinstance(v1, np.ndarray):
                    v1 = np.zeros(EMBED_DIM)
                if not isinstance(v2, np.ndarray):
                    v2 = np.zeros(EMBED_DIM)
                alpha = np.random.rand()
                v_new = v1 + alpha * (v2 - v1)
                new_row = base.loc[i].copy()
                new_row["metric_embedding"] = v_new
                new_row["score_label"] = cls
                new_row["score_float"] = float(cls)
                added_rows.append(new_row)
            continue

        # Sample from GMM
        samples, _ = fitted.sample(needed)
        real_rows = cls_df.reset_index(drop=True)
        for s_idx in range(needed):
            new_row = real_rows.sample(n=1, random_state=random_state + s_idx).iloc[0].copy()
            new_row["metric_embedding"] = samples[s_idx]
            new_row["score_label"] = cls
            new_row["score_float"] = float(cls)
            added_rows.append(new_row)

        # Additionally add 10 copies per original sample (if requested)
        # Here we add up to max_copies for each real sample (optional)
        for idx in range(len(real_rows)):
            base_row = real_rows.loc[idx].copy()
            for c in range(min(max_copies, needed)):  # careful to not explode counts
                jitter = np.random.normal(scale=1e-3, size=EMBED_DIM)
                base_row_copy = base_row.copy()
                base_row_copy["metric_embedding"] = (base_row["metric_embedding"]
                                                    if isinstance(base_row["metric_embedding"], np.ndarray)
                                                    else np.zeros(EMBED_DIM)) + jitter
                base_row_copy["score_label"] = base_row_copy["score_label"]
                base_row_copy["score_float"] = float(base_row_copy["score_label"])
                added_rows.append(base_row_copy)

    if added_rows:
        aug_df = pd.DataFrame(added_rows)
        result = pd.concat([df, aug_df], ignore_index=True)
        print(f"GMM augmentation (dataset-level): added {len(added_rows)} samples; dataset size {len(df)} -> {len(result)}")
        return result
    else:
        print("GMM augmentation: no samples added (all classes meet target).")
        return df

# -------------------------
# Plotting utilities: TSNE & ISOMAP (clean)
# -------------------------
def plot_tsne_isomap(before_emb: np.ndarray, before_lbl: np.ndarray,
                     after_emb: np.ndarray, after_lbl: np.ndarray,
                     save_dir: str = None):
    """
    Produce two-row figure: [t-SNE before, t-SNE after] and [Isomap before, Isomap after].
    """
    # Standardize embeddings for manifold methods
    scaler = StandardScaler()
    before_scaled = scaler.fit_transform(before_emb)
    after_scaled = scaler.transform(after_emb) if after_emb.shape[0] > 0 else after_emb

    # Fit TSNE separately (keeps comparable style)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=CFG["SEED"])
    before_tsne = tsne.fit_transform(before_scaled)
    tsne2 = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=CFG["SEED"]+1)
    after_tsne = tsne2.fit_transform(after_scaled)

    # Isomap
    iso = Isomap(n_neighbors=10, n_components=2)
    before_iso = iso.fit_transform(before_scaled)
    iso2 = Isomap(n_neighbors=10, n_components=2)
    after_iso = iso2.fit_transform(after_scaled)

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = "viridis"

    # TSNE BEFORE
    ax = axes[0, 0]
    sc = ax.scatter(before_tsne[:, 0], before_tsne[:, 1], c=before_lbl, cmap=cmap, s=18, alpha=0.75, edgecolors='none')
    ax.set_title("t-SNE BEFORE augmentation")
    ax.set_xlabel("TSNE 1"); ax.set_ylabel("TSNE 2")
    fig.colorbar(sc, ax=ax, label="score_label")

    # TSNE AFTER
    ax = axes[0, 1]
    sc = ax.scatter(after_tsne[:, 0], after_tsne[:, 1], c=after_lbl, cmap=cmap, s=10, alpha=0.6, edgecolors='none')
    ax.set_title("t-SNE AFTER augmentation")
    ax.set_xlabel("TSNE 1"); ax.set_ylabel("TSNE 2")
    fig.colorbar(sc, ax=ax, label="score_label")

    # ISOMAP BEFORE
    ax = axes[1, 0]
    sc = ax.scatter(before_iso[:, 0], before_iso[:, 1], c=before_lbl, cmap=cmap, s=18, alpha=0.75, edgecolors='none')
    ax.set_title("Isomap BEFORE augmentation")
    ax.set_xlabel("Isomap 1"); ax.set_ylabel("Isomap 2")
    fig.colorbar(sc, ax=ax, label="score_label")

    # ISOMAP AFTER
    ax = axes[1, 1]
    sc = ax.scatter(after_iso[:, 0], after_iso[:, 1], c=after_lbl, cmap=cmap, s=10, alpha=0.6, edgecolors='none')
    ax.set_title("Isomap AFTER augmentation")
    ax.set_xlabel("Isomap 1"); ax.set_ylabel("Isomap 2")
    fig.colorbar(sc, ax=ax, label="score_label")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, "embeddings_before_after.png")
        fig.savefig(outpath, dpi=150)
        print(f"Saved plot to {outpath}")
    plt.show()

# -------------------------
# Dataset class & Model (unchanged)
# -------------------------
class TextMetricDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.texts = df["full_text"].values
        emb_series = df["metric_embedding"].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(EMBED_DIM))
        self.metric_matrix = np.stack(emb_series.values)
        self.labels = df["score_label"].values
        self.scores = df["score_float"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "metric_embed": torch.tensor(self.metric_matrix[idx], dtype=torch.float),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "score": torch.tensor(float(self.scores[idx]), dtype=torch.float)
        }

class PairInteractionModel(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.encoder = SentenceTransformer(backbone_name)
        self.head = nn.Sequential(
            nn.Linear(INTERACT_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, metric_vec: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_out = self.encoder({"input_ids": input_ids, "attention_mask": attention_mask})
        v2 = text_out["sentence_embedding"]
        v1 = metric_vec
        diff = torch.abs(v1 - v2)
        prod = v1 * v2
        x = torch.cat([v1, v2, diff, prod], dim=1)
        return self.head(x)

# -------------------------
# Training & evaluation (unchanged)
# -------------------------
def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch["metric_embed"].to(device),
                       batch["input_ids"].to(device),
                       batch["attention_mask"].to(device))
        loss = criterion(logits, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds = []
    truths = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["metric_embed"].to(device),
                           batch["input_ids"].to(device),
                           batch["attention_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            total_loss += loss.item()
            es = expected_score_from_logits(logits)
            preds.extend(es.cpu().numpy())
            truths.extend(batch["score"].cpu().numpy())
    rmse = math.sqrt(mean_squared_error(truths, preds))
    return total_loss / len(loader), rmse

# -------------------------
# Run training after dataset-level augmentation
# -------------------------
def run_training(df: pd.DataFrame, metric_map: dict, out_dir: str = "./models_gmm_pre") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CFG["MODEL_NAME"])

    # class weights (inverse-frequency)
    counts = df["score_label"].value_counts().sort_index().reindex(range(NUM_CLASSES), fill_value=0)
    total = len(df)
    class_weights = total / (NUM_CLASSES * (counts + 1))
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(CFG["DEVICE"])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ---------- Perform augmentation ONCE (dataset-level) ----------
    print("Running dataset-level GMM augmentation (before CV)...")
    df_aug = gmm_augment_dataset(
        df,
        target_per_class=CFG["GMM_TARGET_PER_CLASS"],
        max_components=CFG["GMM_MAX_COMPONENTS"],
        max_copies=CFG["GMM_MAX_COPIES"],
        random_state=CFG["SEED"]
    )

    # Produce visualization BEFORE and AFTER augmentation
    before_emb = np.stack(df["metric_embedding"].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(EMBED_DIM)).values)
    before_lbl = df["score_label"].values
    after_emb = np.stack(df_aug["metric_embedding"].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(EMBED_DIM)).values)
    after_lbl = df_aug["score_label"].values

    # Plot and optionally save
    plot_tsne_isomap(before_emb, before_lbl, after_emb, after_lbl, save_dir=out_dir)

    # ---------- Now run StratifiedGroupKFold on augmented dataset ----------
    skf = StratifiedGroupKFold(n_splits=CFG["FOLDS"], shuffle=True, random_state=CFG["SEED"])
    X = df_aug.index.values
    y = df_aug["score_label"].values
    groups = df_aug["metric_name"].values

    saved_paths = []
    oof_rmse = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        print(f"\n=== Fold {fold+1}/{CFG['FOLDS']} ===")
        train_fold = df_aug.iloc[train_idx].reset_index(drop=True)
        val_fold = df_aug.iloc[val_idx].reset_index(drop=True)

        train_ds = TextMetricDataset(train_fold, tokenizer, CFG["MAX_TOKENS"])
        val_ds = TextMetricDataset(val_fold, tokenizer, CFG["MAX_TOKENS"])

        train_loader = DataLoader(train_ds, batch_size=CFG["BATCH"], shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["BATCH"], shuffle=False, num_workers=0)

        print(f"Train samples: {len(train_fold)}, Val samples: {len(val_fold)}")

        model = PairInteractionModel(CFG["MODEL_NAME"]).to(CFG["DEVICE"])
        optimizer = optim.AdamW([
            {"params": model.encoder.parameters(), "lr": CFG["LR_ENCODER"]},
            {"params": model.head.parameters(), "lr": CFG["LR_HEAD"]}
        ])

        best_rmse = float("inf")
        model_path = os.path.join(out_dir, f"fold_{fold+1}.pth")

        for epoch in range(CFG["EPOCHS"]):
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, CFG["DEVICE"])
            v_loss, v_rmse = eval_epoch(model, val_loader, criterion, CFG["DEVICE"])
            print(f"Epoch {epoch+1}: Train Loss={tr_loss:.4f}, Val Loss={v_loss:.4f}, Val RMSE={v_rmse:.4f}")
            if v_rmse < best_rmse:
                best_rmse = v_rmse
                torch.save(model.state_dict(), model_path)
                print(f"  Saved best model -> {model_path}")

        saved_paths.append(model_path)
        oof_rmse.append(best_rmse)

        # cleanup
        del model, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining finished. OOF RMSEs:", oof_rmse)
    return saved_paths

# -------------------------
# Inference / ensemble (unchanged)
# -------------------------
def inference_ensemble(model_paths: List[str], test_json_path: str = "test_data.json", out_csv: str = "submission_gmm_pre.csv"):
    if not os.path.exists(test_json_path):
        print(f"Test file {test_json_path} not found. Skipping inference.")
        return

    with open(test_json_path, "r") as fh:
        test_json = json.load(fh)
    test_df = pd.DataFrame(test_json)
    if "ID" not in test_df.columns:
        test_df["ID"] = range(1, len(test_df) + 1)

    test_df["user_prompt"] = test_df["user_prompt"].astype(str)
    test_df["system_prompt"] = test_df["system_prompt"].fillna("None").astype(str)
    test_df["response"] = test_df["response"].astype(str)
    test_df["full_text"] = (
        "User: " + test_df["user_prompt"] +
        " | System: " + test_df["system_prompt"] +
        " | Response: " + test_df["response"]
    )

    # metric_map load (same as train)
    with open("metric_names.json", "r") as fh:
        metric_names = json.load(fh)
    metric_embs = np.load("metric_name_embeddings.npy")
    metric_map = dict(zip(metric_names, metric_embs))
    test_df["metric_embedding"] = test_df["metric_name"].map(metric_map)
    test_df["score_label"] = 0
    test_df["score_float"] = 0.0

    tokenizer = AutoTokenizer.from_pretrained(CFG["MODEL_NAME"])
    test_ds = TextMetricDataset(test_df, tokenizer, CFG["MAX_TOKENS"])
    test_loader = DataLoader(test_ds, batch_size=CFG["BATCH"] * 2, shuffle=False, num_workers=0)

    device = CFG["DEVICE"]
    model = PairInteractionModel(CFG["MODEL_NAME"]).to(device)

    all_preds = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: model file {path} missing, skipping.")
            continue
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predict {os.path.basename(path)}"):
                logits = model(batch["metric_embed"].to(device), batch["input_ids"].to(device), batch["attention_mask"].to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(probs)
        all_preds.append(np.concatenate(fold_probs, axis=0))

    if not all_preds:
        raise RuntimeError("No model predictions collected. Check model paths.")

    avg_probs = np.mean(all_preds, axis=0)
    scores = avg_probs.dot(np.arange(NUM_CLASSES))
    scores = np.clip(scores, 0, 10)

    submission = pd.DataFrame({"ID": test_df["ID"], "score": scores})
    submission.to_csv(out_csv, index=False)
    print(f"Saved submission: {out_csv}")
    print(submission.head())

# -------------------------
# Entrypoint
# -------------------------
def main():
    # Load train data and metric map
    train_df_raw, metric_map = load_training_files()
    train_df = prepare_dataframe(train_df_raw, metric_map)

    # Run training with dataset-level GMM augmentation
    model_dir = "./models_gmm_pre"
    print('training...')
    model_files = run_training(train_df, metric_map, out_dir=model_dir)

    # Inference (if test_data.json exists)
    inference_ensemble(model_files)

if __name__ == "__main__":
    main()
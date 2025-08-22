from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
from src.utils.logger import get_logger
from src.datasets.loaders import get_dataloaders
from src.models.cnn import ResNet18Binary
from src.rl.policy import PolicyNetwork
from src.utils.metrics import evaluate_classifier
from src.al.strategies import random_sampling, entropy_sampling
from src.utils.plots import plot_curves, save_json
# Additional utilities for future enhancements
# from src.utils.confidence import get_confidence_metrics, calculate_entropy
# from src.utils.evaluation import track_learning_curve, calculate_policy_entropy

PROCESSED_DIR = Path("data/processed/catsdogs_128")
CHECK_DIR = Path("checkpoints/week4")
OUT_DIR = Path("outputs/week4")
LOG_FILE = Path("logs/week4_rl.log")

INIT_LABELED = 1000
ACQ_SIZE = 500
ROUNDS = 8
EPOCHS_PER_ROUND = 3
POLICY_EPOCHS = 5  # REINFORCE outer epochs
BATCH = 64
LR_MODEL = 1e-3
LR_POLICY = 1e-3
SEED = 123

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("week4", LOG_FILE)
    logger.info("Week 4: Final Report & Visualization - RL (REINFORCE) started")
    
    train_loader_full, val_loader, train_ds, val_ds = get_dataloaders(PROCESSED_DIR, img_size=128, batch_size=BATCH)
    N = len(train_ds)
    all_idx = np.arange(N)
    rng.shuffle(all_idx)
    
    # Policy works on extracted features of unlabeled pool
    feat_dim = 512
    policy = PolicyNetwork(in_dim=feat_dim, hidden=256).to(device)
    pol_opt = optim.Adam(policy.parameters(), lr=LR_POLICY)
    
    def extract_feats(model, subset_indices):
        loader = torch.utils.data.DataLoader(Subset(train_ds, subset_indices), batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
        feats_all = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                feats = model.extract_features(x)
                feats_all.append(feats.cpu())
        return torch.cat(feats_all, dim=0)  # (M,512)
    
    rl_curve = []
    
    for pe in range(1, POLICY_EPOCHS + 1):
        labeled = set(all_idx[:INIT_LABELED].tolist())
        unlabeled = set(all_idx[INIT_LABELED:].tolist())
        ep_rewards = []
        
        for rd in range(ROUNDS + 1):
            # Train classifier from scratch on current labeled set
            labeled_list = sorted(list(labeled))
            train_subset = Subset(train_ds, labeled_list)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
            
            model = ResNet18Binary(pretrained=True).to(device)
            crit = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=LR_MODEL)
            
            for _ in range(EPOCHS_PER_ROUND):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward()
                    opt.step()
            
            acc, _ = evaluate_classifier(model, val_loader, device)
            rl_curve.append(acc)
            logger.info(f"PolicyEpoch {pe}/{POLICY_EPOCHS} Round {rd}/{ROUNDS} - Labeled:{len(labeled)} - ValAcc:{acc:.4f}")
            
            if rd == ROUNDS:
                break
            
            # Select next ACQ_SIZE via policy on features
            unlabeled_list = sorted(list(unlabeled))
            
            # Extract features of unlabeled pool using current classifier backbone
            feats = extract_feats(model, unlabeled_list).to(device)  # (U,512)
            logits = policy(feats)  # (U,)
            probs = torch.softmax(logits, dim=0)
            
            # Sample without replacement approximately by top-k of probs
            k = min(ACQ_SIZE, len(unlabeled_list))
            topk = torch.topk(probs, k=k).indices.cpu()
            selected = [unlabeled_list[i] for i in topk.tolist()]
            
            # REINFORCE: define reward as val acc improvement after adding batch
            prev_acc = acc
            new_labeled = set(selected)
            labeled2 = labeled | new_labeled
            
            # Train a quick model after acquisition to measure reward
            train_subset2 = Subset(train_ds, sorted(list(labeled2)))
            train_loader2 = torch.utils.data.DataLoader(train_subset2, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
            
            model2 = ResNet18Binary(pretrained=True).to(device)
            opt2 = optim.Adam(model2.parameters(), lr=LR_MODEL)
            
            for _ in range(EPOCHS_PER_ROUND):
                model2.train()
                for x, y in train_loader2:
                    x, y = x.to(device), y.to(device)
                    opt2.zero_grad()
                    loss2 = crit(model2(x), y)
                    loss2.backward()
                    opt2.step()
            
            acc2, _ = evaluate_classifier(model2, val_loader, device)
            reward = acc2 - prev_acc
            ep_rewards.append(reward)
            
            # Policy loss (maximize reward -> minimize -reward * logprob(selected))
            sel_feats = feats[topk]  # (k,512)
            sel_logits = policy(sel_feats)
            sel_logp = torch.log_softmax(sel_logits, dim=0)
            loss_policy = -(reward * sel_logp.mean())
            
            pol_opt.zero_grad()
            loss_policy.backward()
            pol_opt.step()
            
            # Commit acquisition
            labeled = labeled2
            unlabeled -= new_labeled
        
        CHECK_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), CHECK_DIR / f"policy_epoch{pe}.pth")
    
    # Compare RL vs baselines in one figure using last runs from week3 if present
    curves = {"RL": rl_curve}
    save_json(curves, OUT_DIR / "rl_curve.json")
    plot_curves(curves, "Step", "Val Accuracy", "Week4: RL Curve", OUT_DIR / "plots/rl_curve.png")
    
    logger.info(f"Saved policy checkpoints to {CHECK_DIR} and outputs to {OUT_DIR}")
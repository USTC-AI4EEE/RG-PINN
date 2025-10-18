import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data_loader import BoundaryLayerDataset
from pinn_model import PINNNet
import h5py
from tqdm import tqdm, trange
import csv

# GPU selection (command line argument first, then environment variable, finally cuda:0)
gpu_id = None
if len(sys.argv) > 1:
    gpu_id = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f"[INFO] Using GPU: {gpu_id}")

# Configuration
EPOCHS = 20000
BATCH_SIZE = 128
LR = 1e-3
LAYERS = [3] + [128]*6 + [3]  # Input x,y,t, output u,v,p
REY = 752000  # Boundary layer Reynolds number
BETA = 0.3  # Difficulty weight hyperparameter
VAL_EVAL_EPOCH = 500  # Validation evaluation interval

# Ablation study configuration
USE_REGION_GROWTH = True  # Whether to use region growth strategy, False for normal training
USE_WEAK_CONTINUITY = True  # Whether to use weak continuity loss, False for normal point-wise PDE loss
USE_DIFFICULTY_WEIGHT = False  # Whether to use difficulty weight

# Print ablation study configuration
print(f"[INFO] Ablation Study Configuration:")
print(f"  Region Growth: {'ON' if USE_REGION_GROWTH else 'OFF'}")
print(f"  Weak Continuity: {'ON' if USE_WEAK_CONTINUITY else 'OFF'}")
print(f"  Difficulty Weight: {'ON' if USE_DIFFICULTY_WEIGHT else 'OFF'}")


N_POINTS = 10

# Data paths
TRAIN_PATH = f'../training_validation_sets_{N_POINTS}/training_set.h5'  # Training set: 20 fixed observation stations
VAL_PATH = f'../training_validation_sets_{N_POINTS}/validation_set.h5'  # Validation set: complete grid data

# Non-dimensionalization parameters
L = 3.0  # Characteristic length (m)
U = 3.76  # Characteristic velocity (m/s) 
T = 0.79787  # Characteristic time (s)

# Standardization range (data loader handles normalization automatically, here set network input range)
lb = np.array([0.0, 0.0, 0.0])  # Standardized lower bound
ub = np.array([2.0, 1.0, 2.066])  # Standardized upper bound

# Load all observation points (spatial) for region growth strategy
print(f"[INFO] Loading training data for region growth strategy from: {TRAIN_PATH}")
with h5py.File(TRAIN_PATH, 'r') as f:
    x_all = np.array(f['x']).flatten()
    y_all = np.array(f['y']).flatten()
    t_all = np.array(f['t']).flatten()

# Take only the first N points as spatial coordinates (because data is flattened by time)
N_spatial = N_POINTS  # Number of spatial points
x_spatial = x_all[:N_spatial]
y_spatial = y_all[:N_spatial]

N_pts = N_spatial
center = np.array([0.0, 0.0])  # Center at (0,0)
coords = np.stack([x_spatial, y_spatial], axis=1)
used_idx = []
unused_idx = list(range(N_pts))

# Region growth strategy implementation

def region_grow_strategy_1(coords, center):
    """Strategy 1: Start from center, add all points closest to used set each time"""
    N_pts = coords.shape[0]
    used_idx = []
    unused_idx = list(range(N_pts))
    stage_indices = []
    stage_rects = []
    # 1. Find the point closest to center
    center_dist = np.linalg.norm(coords - center, axis=1)
    first_idx = np.argmin(center_dist)
    used_idx.append(first_idx)
    unused_idx.remove(first_idx)
    # 2. Sequentially select all points closest to used set
    while unused_idx:
        dists = np.min(np.linalg.norm(coords[unused_idx][:,None,:] - coords[used_idx][None,:,:], axis=2), axis=1)
        min_dist = dists.min()
        next_idxs_in_unused = [i for i, d in enumerate(dists) if np.isclose(d, min_dist)]
        next_idxs = [unused_idx[i] for i in next_idxs_in_unused]
        for idx in next_idxs:
            used_idx.append(idx)
        for idx in sorted(next_idxs, reverse=True):
            del unused_idx[unused_idx.index(idx)]
        stage_indices.append(list(used_idx))
        x_min, y_min = coords[used_idx][:,0].min(), coords[used_idx][:,1].min()
        x_max, y_max = coords[used_idx][:,0].max(), coords[used_idx][:,1].max()
        stage_rects.append((x_min, x_max, y_min, y_max))
    # Last stage: full domain
    stage_indices.append(list(range(N_pts)))
    x_min, y_min = coords[:,0].min(), coords[:,1].min()
    x_max, y_max = coords[:,0].max(), coords[:,1].max()
    stage_rects.append((x_min, x_max, y_min, y_max))
    return stage_indices, stage_rects

def region_grow_strategy_2(coords, center):
    # Strategy 2 placeholder
    return [], []

# Strategy selection
if USE_REGION_GROWTH:
    REGION_GROW_STRATEGY = 1  # 1: center nearest expansion, 2: other strategies
    if REGION_GROW_STRATEGY == 1:
        stage_indices, stage_rects = region_grow_strategy_1(coords, center)
    elif REGION_GROW_STRATEGY == 2:
        stage_indices, stage_rects = region_grow_strategy_2(coords, center)
    else:
        raise ValueError('Unknown region grow strategy')
    STAGES = len(stage_indices)
    print(f"[INFO] Using region growth strategy with {STAGES} stages")
else:
    # Normal training: use all points, only one stage
    stage_indices = [list(range(N_pts))]
    stage_rects = [(coords[:,0].min(), coords[:,0].max(), coords[:,1].min(), coords[:,1].max())]
    STAGES = 1
    print(f"[INFO] Using normal training (no region growth) with all {N_pts} points")

# Load full training data (normalized)
print(f"[INFO] Loading full training data from: {TRAIN_PATH}")
train_dataset = BoundaryLayerDataset(
    TRAIN_PATH, 
    normalize=True,
    L=L, U=U, T=T,
    lb=lb, ub=ub,
    u_mean=2.5, v_mean=0.0, u_std=1.5, v_std=0.5
)

# Get normalized data
x = train_dataset.x
y = train_dataset.y
t = train_dataset.t
u = train_dataset.u
v = train_dataset.v
p = train_dataset.p  # Boundary layer data has no pressure field

# Correctly calculate number of spatial and temporal points
N = N_POINTS  # Number of spatial points (fixed observation stations)
T = 100  # Number of temporal points

print(f"[INFO] Training data loaded:")
print(f"  Spatial points: {N}")
print(f"  Time steps: {T}")
print(f"  Total data points: {len(train_dataset)}")
print(f"  Data ranges:")
print(f"    x: [{x.min():.6f}, {x.max():.6f}]")
print(f"    y: [{y.min():.6f}, {y.max():.6f}]")
print(f"    t: [{t.min():.6f}, {t.max():.6f}]")
print(f"    u: [{u.min():.6f}, {u.max():.6f}]")
print(f"    v: [{v.min():.6f}, {v.max():.6f}]")

# Record validation metrics
val_metrics = []

# Learning rate scheduling related variables
best_val_loss = float('inf')
current_lr = LR

# Network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
model = PINNNet(LAYERS, lb, ub).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=1000, 
    verbose=True,
    min_lr=1e-6
)

def rel_l2(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def mape(pred, true):
    mask = (true != 0)
    return (torch.abs((pred - true)[mask] / true[mask])).mean() * 100

def corrcoef(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    pred_mean = pred.mean()
    true_mean = true.mean()
    num = ((pred - pred_mean) * (true - true_mean)).sum()
    denom = torch.sqrt(((pred - pred_mean) ** 2).sum() * ((true - true_mean) ** 2).sum())
    return num / (denom + 1e-8)

# Multi-stage region growth training
EPOCHS_PER_STAGE = EPOCHS // STAGES

for stage in range(STAGES):
    # Reset learning rate to 0.001 at the beginning of each stage
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001
    print(f"[INFO] Stage {stage+1}: Learning rate reset to 0.001")
    print(f"\n[Stage {stage+1}/{STAGES}] Training region: x=[{stage_rects[stage][0]:.2f},{stage_rects[stage][1]:.2f}], y=[{stage_rects[stage][2]:.2f},{stage_rects[stage][3]:.2f}], used points={len(stage_indices[stage])}")
    used_space_idx = np.array(stage_indices[stage])  # Spatial point indices
    
    # Reorganize data: extract all time step data for specified spatial points from flattened data
    # Training data is flattened as (time, space), i.e., [t0_s0, t0_s1, ..., t0_sN, t1_s0, t1_s1, ..., t1_sN, ...]
    x_stage = []
    y_stage = []
    t_stage = []
    u_stage = []
    v_stage = []
    if p is not None:
        p_stage = []
    else:
        p_stage = None
    
    for ti in range(T):
        for si in used_space_idx:
            idx = ti * N + si
            x_stage.append(x[idx])
            y_stage.append(y[idx])
            t_stage.append(t[idx])
            u_stage.append(u[idx])
            v_stage.append(v[idx])
            if p is not None:
                p_stage.append(p[idx])
    
    x_stage = np.array(x_stage)
    y_stage = np.array(y_stage)
    t_stage = np.array(t_stage)
    u_stage = np.array(u_stage)
    v_stage = np.array(v_stage)
    if p is not None:
        p_stage = np.array(p_stage)
    # Build Dataset
    class StageDataset(Dataset):
        def __len__(self):
            return len(x_stage)
        def __getitem__(self, idx):
            if p_stage is not None:
                return (torch.tensor(x_stage[idx], dtype=torch.float32),
                        torch.tensor(y_stage[idx], dtype=torch.float32),
                        torch.tensor(t_stage[idx], dtype=torch.float32),
                        torch.tensor(u_stage[idx], dtype=torch.float32),
                        torch.tensor(v_stage[idx], dtype=torch.float32),
                        torch.tensor(p_stage[idx], dtype=torch.float32))
            else:
                return (torch.tensor(x_stage[idx], dtype=torch.float32),
                        torch.tensor(y_stage[idx], dtype=torch.float32),
                        torch.tensor(t_stage[idx], dtype=torch.float32),
                        torch.tensor(u_stage[idx], dtype=torch.float32),
                        torch.tensor(v_stage[idx], dtype=torch.float32))
    train_loader = DataLoader(StageDataset(), batch_size=BATCH_SIZE, shuffle=True)
    # Train this stage
    # Add extra 10000 epochs for the last stage
    epochs_this_stage = EPOCHS_PER_STAGE + (10000 if stage == STAGES - 1 else 0)
    for epoch in trange(epochs_this_stage, desc=f'Stage {stage+1}', dynamic_ncols=True):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 6:
                x_b, y_b, t_b, u_b, v_b, _ = batch
            else:
                x_b, y_b, t_b, u_b, v_b = batch
            x_b, y_b, t_b, u_b, v_b = [d.to(device) for d in [x_b, y_b, t_b, u_b, v_b]]
            optimizer.zero_grad()
            loss_data = model.loss_data(x_b, y_b, t_b, u_b, v_b)
            idx_eq = torch.randperm(x_b.shape[0])[:BATCH_SIZE]
            x_eq, y_eq, t_eq = x_b[idx_eq], y_b[idx_eq], t_b[idx_eq]
            # Pass epoch, total_epoch, beta and current region
            # Calculate global epoch, considering extra epochs in the last stage
            if stage < STAGES - 1:
                global_epoch = stage * EPOCHS_PER_STAGE + epoch + 1
            else:
                global_epoch = (STAGES - 1) * EPOCHS_PER_STAGE + epoch + 1
            current_region = stage_rects[stage] if stage < STAGES - 1 else None  # Last stage uses full plane
            loss_eq = model.loss_equation(x_eq, y_eq, t_eq, REY, epoch=global_epoch, total_epoch=EPOCHS, beta=BETA, use_difficulty_weight=USE_DIFFICULTY_WEIGHT, current_region=current_region, use_weak_continuity=USE_WEAK_CONTINUITY)
            loss = loss_data + loss_eq
            loss.backward()
            

            
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        # Calculate global epoch, considering extra epochs in the last stage
        if stage < STAGES - 1:
            global_epoch = stage * EPOCHS_PER_STAGE + epoch + 1
        else:
            global_epoch = (STAGES - 1) * EPOCHS_PER_STAGE + epoch + 1
        if (global_epoch) % VAL_EVAL_EPOCH == 0 or (stage == STAGES-1 and epoch == epochs_this_stage-1):
            # Validation
            model.eval()
            val_loss = 0
            rel_u, rel_v = 0, 0
            rmse_u, rmse_v = 0, 0
            mape_u, mape_v = 0, 0
            acc_u, acc_v = 0, 0
            n = 0
            with torch.no_grad():
                for batch in DataLoader(BoundaryLayerDataset(VAL_PATH, normalize=True, L=L, U=U, T=T, lb=lb, ub=ub, u_mean=2.5, v_mean=0.0, u_std=1.5, v_std=0.5), batch_size=1024, shuffle=False):
                    if len(batch) == 6:
                        x_v, y_v, t_v, u_v, v_v, _ = batch
                    else:
                        x_v, y_v, t_v, u_v, v_v = batch
                    x_v, y_v, t_v, u_v, v_v = [d.to(device) for d in [x_v, y_v, t_v, u_v, v_v]]
                    u_pred, v_pred, _ = model.predict_uvp(x_v, y_v, t_v)
                    rel_u += rel_l2(u_pred, u_v).item()
                    rel_v += rel_l2(v_pred, v_v).item()
                    rmse_u += rmse(u_pred, u_v).item()
                    rmse_v += rmse(v_pred, v_v).item()
                    mape_u += mape(u_pred, u_v).item()
                    mape_v += mape(v_pred, v_v).item()
                    acc_u += corrcoef(u_pred, u_v).item()
                    acc_v += corrcoef(v_pred, v_v).item()
                    val_loss += model.loss_data(x_v, y_v, t_v, u_v, v_v).item()
                    n += 1
            val_loss = val_loss / n
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            val_metrics.append([
                global_epoch, train_loss, val_loss,
                rel_u/n, rel_v/n,
                rmse_u/n, rmse_v/n,
                mape_u/n, mape_v/n,
                acc_u/n, acc_v/n
            ])
            tqdm.write(f"Epoch {global_epoch}: train_loss={train_loss:.4e}, val_loss={val_loss:.4e}, lr={current_lr:.2e}")
            # Update learning rate scheduler (based on validation loss)
            scheduler.step(val_loss)
        else:
            tqdm.write(f"Epoch {global_epoch}: train_loss={train_loss:.4e}")


# Determine folder name based on ablation study configuration
folder_parts = [f'checkpoints_N={N_POINTS}']
if USE_DIFFICULTY_WEIGHT:
    folder_parts.append(f'beta={BETA}')
else:
    folder_parts.append('no_difficulty_weight')
if USE_REGION_GROWTH:
    folder_parts.append('region_growth')
else:
    folder_parts.append('normal_train')
if USE_WEAK_CONTINUITY:
    folder_parts.append('weak_continuity')
else:
    folder_parts.append('point_pde')
folder_name = '_'.join(folder_parts)

os.makedirs(folder_name, exist_ok=True)
torch.save(model.state_dict(), f'{folder_name}/region_grow_pinn_boundary_layer.pth')
with open(f'{folder_name}/val_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'train_loss', 'val_loss',
        'RelL2_u', 'RelL2_v',
        'RMSE_u', 'RMSE_v',
        'MAPE_u', 'MAPE_v',
        'acc_u', 'acc_v'])
    writer.writerows(val_metrics)
print('Training finished. Model and validation metrics saved.')
print(f'Model saved to: {folder_name}/region_grow_pinn_boundary_layer.pth')
print(f'Metrics saved to: {folder_name}/val_metrics.csv')

# Learning rate scheduling summary
final_lr = optimizer.param_groups[0]['lr']
print(f'Learning rate scheduling summary:')
print(f'  Initial LR: {LR:.2e}')
print(f'  Final LR: {final_lr:.2e}')
print(f'  Best validation loss: {best_val_loss:.4e}')
print(f'  LR reduction factor: {final_lr/LR:.4f}') 
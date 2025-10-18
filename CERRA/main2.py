import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from cerra_data_loader import create_cerra_dataloader
from pinn_model import PINNNet
import h5py
from tqdm import tqdm, trange
import csv
import time

# GPU selection (command line argument > environment variable > cuda:0)
gpu_id = None
if len(sys.argv) > 1:
    gpu_id = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f"[INFO] Setting CUDA_VISIBLE_DEVICES to: {gpu_id}")

# Ablation study configuration
USE_REGION_GROWTH = True  # Whether to use region growing strategy, False for normal training
USE_WEAK_CONTINUITY = False  # Whether to use weak continuity loss, False for normal point-wise PDE loss
USE_DIFFICULTY_WEIGHT = True  # Whether to use difficulty weighting

# Basic configuration
EPOCHS = 20000
BATCH_SIZE = 128
LR = 1e-3
LAYERS = [3] + [128]*6 + [3]  # Input x,y,t, output u,v,p

# Number of stages will be calculated automatically based on training set points
# Region growing strategy: stage 1 with 2 points, stage 2 with 3 points, stage 3 with 4 points... stage N-1 with N points, stage N with entire dataset
VAL_EVAL_EPOCH = 1000  # Evaluate every N epochs
BETA = 0.3  # Difficulty weighting hyperparameter

L = 570  # km    (characteristic length)
U = 6.57   # m/s    (characteristic velocity)  
DT = 24.1  # h      (characteristic time)
Re = 2.5e+11  # Reynolds number

N_POINTS = 10
# Data path
DATASET_DIR = f'../dataset_N={N_POINTS}_100x100'  # NPY format dataset directory

# Data loading (non-dimensionalization + standardization)
print("Loading training data...")
train_loader, train_dataset = create_cerra_dataloader(
    dataset_dir=DATASET_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    normalize=True,
    nondimensionalize=True,
    L=L, U=U, DT=DT,
    mode='train'
)

print("Loading validation data...")
val_loader, val_dataset = create_cerra_dataloader(
    dataset_dir=DATASET_DIR,
    batch_size=1024,
    shuffle=False,
    normalize=True,
    nondimensionalize=True,
    L=L, U=U, DT=DT,
    mode='val'
)

# Get latitude range from training dataset
latitude_range = train_dataset.get_latitude_range()
print(f"[INFO] Using latitude range from dataset: {latitude_range[0]:.2f}° - {latitude_range[1]:.2f}°")

# Compute boundaries (based on non-dimensionalized data)
print("Computing boundaries...")
all_inputs = train_dataset.get_all_inputs()
lb = torch.min(all_inputs, dim=0)[0].numpy()
ub = torch.max(all_inputs, dim=0)[0].numpy()
print(f"Lower bounds: {lb}")
print(f"Upper bounds: {ub}")

# Extract all data from training dataset
print("Extracting training data for region growing...")
x_all = train_dataset.data['x']
y_all = train_dataset.data['y']
t_all = train_dataset.data['t']
u_all = train_dataset.data['u']
v_all = train_dataset.data['v']

# Get unique spatial points (remove duplicates)
unique_points = np.unique(np.stack([x_all, y_all], axis=1), axis=0)
N_pts = len(unique_points)
T = len(np.unique(t_all))  # Number of time steps
print(f"Unique spatial points: {N_pts}")
print(f"Time steps: {T}")

# Calculate number of stages
if USE_REGION_GROWTH:
    # Region growing strategy: stage 1 with 2 points, stage 2 with 3 points, stage 3 with 4 points... stage N-1 with N points, stage N with entire dataset
    STAGES = N_pts  # From 2 points to N_pts points, plus entire dataset stage, total N_pts stages
    EPOCHS_PER_STAGE = EPOCHS // STAGES
    EXTRA_EPOCHS_LAST_STAGE = 10000  # Additional epochs for the last stage
    
    print(f"Region Growth Mode:")
    print(f"  Total points: {N_pts}")
    print(f"  Region growing stages: {STAGES} (2->3->4->...->{N_pts}->all data)")
    print(f"  Calculated stages: {STAGES}")
    print(f"  Epochs per stage: {EPOCHS_PER_STAGE}")
    print(f"  Extra epochs for last stage: {EXTRA_EPOCHS_LAST_STAGE}")
    print(f"  Total epochs: {STAGES * EPOCHS_PER_STAGE + EXTRA_EPOCHS_LAST_STAGE}")
else:
    # Normal training mode: single stage training with all data
    STAGES = 1
    EPOCHS_PER_STAGE = EPOCHS
    EXTRA_EPOCHS_LAST_STAGE = 0
    
    print(f"Normal Training Mode:")
    print(f"  Total points: {N_pts}")
    print(f"  Single stage training with all data")
    print(f"  Total epochs: {EPOCHS}")

coords = unique_points
center = np.array([np.mean(coords[:, 0]), np.mean(coords[:, 1])])  # Calculate region center
used_idx = []
unused_idx = list(range(N_pts))

# Region growing strategy implementation

def region_grow_strategy_1(coords, center, stages):
    N_pts = coords.shape[0]
    used_idx = []
    unused_idx = list(range(N_pts))
    stage_indices = []
    stage_rects = []

    # 1. Find the point closest to the region center
    center_dist = np.linalg.norm(coords - center, axis=1)
    first_idx = np.argmin(center_dist)
    used_idx.append(first_idx)
    unused_idx.remove(first_idx)

    # 2. Find another point closest to the center point
    dist_to_center = [np.linalg.norm(coords[i] - coords[first_idx]) for i in unused_idx]
    second_idx_in_unused = np.argmin(dist_to_center)
    second_idx = unused_idx[second_idx_in_unused]
    used_idx.append(second_idx)
    unused_idx.remove(second_idx)
    
    # Stage 1: bounding rectangle of 2 points
    stage_indices.append([first_idx, second_idx])
    stage_rects.append([
        min(coords[first_idx, 0], coords[second_idx, 0]),
        min(coords[first_idx, 1], coords[second_idx, 1]),
        max(coords[first_idx, 0], coords[second_idx, 0]),
        max(coords[first_idx, 1], coords[second_idx, 1])
    ])

    # 3. Region growing: increment one point per stage, region is bounding rectangle of all currently selected points
    for stage in range(1, stages - 1):  # Except the last stage
        if len(unused_idx) == 0:
            break
            
        # Number of points current stage should contain: 2 + stage
        target_points = min(2 + stage, N_pts)
        points_to_add = target_points - len(used_idx)
        
        if points_to_add <= 0:
            break

        # Add new points
        for _ in range(points_to_add):
            if len(unused_idx) == 0:
                break
            # Calculate minimum distance from all unused points to used points
            min_distances = []
            for i in unused_idx:
                dist_to_used = [np.linalg.norm(coords[i] - coords[j]) for j in used_idx]
                min_distances.append(min(dist_to_used))
            closest_idx_in_unused = np.argmin(min_distances)
            closest_idx = unused_idx[closest_idx_in_unused]
            used_idx.append(closest_idx)
            unused_idx.remove(closest_idx)
        
        # Current stage contains all selected points (bounding rectangle region)
        current_stage_points = used_idx.copy()
        stage_indices.append(current_stage_points)
        
        # Calculate bounding rectangle of all currently selected points
        current_coords = coords[current_stage_points]
        stage_rects.append([
            np.min(current_coords[:, 0]),
            np.min(current_coords[:, 1]),
            np.max(current_coords[:, 0]),
            np.max(current_coords[:, 1])
        ])

    # 4. Last stage: entire plane
    if len(stage_indices) < stages:
        # Add all points
        all_points = list(range(N_pts))
        stage_indices.append(all_points)
        # Global boundaries of entire plane (using global boundaries of data)
        stage_rects.append([
            np.min(coords[:, 0]), np.min(coords[:, 1]),
            np.max(coords[:, 0]), np.max(coords[:, 1])
        ])

    return stage_indices, stage_rects

def region_grow_strategy_2(coords, center):
    """Strategy 2: reserved for future use"""
    pass

# Select training strategy based on configuration
if USE_REGION_GROWTH:
    # Use region growing strategy
    stage_indices, stage_rects = region_grow_strategy_1(coords, center, STAGES)
    
    # Print stage information
    print(f"\nRegion growing strategy:")
    print(f"Total stages: {len(stage_indices)}")
    for i, (indices, rect) in enumerate(zip(stage_indices, stage_rects)):
        print(f"Stage {i+1}: {len(indices)} points, rect: [{rect[0]:.3f}, {rect[1]:.3f}, {rect[2]:.3f}, {rect[3]:.3f}]")
else:
    # Normal training mode: single stage containing all points
    stage_indices = [list(range(N_pts))]  # All points
    stage_rects = [[
        np.min(coords[:, 0]), np.min(coords[:, 1]),
        np.max(coords[:, 0]), np.max(coords[:, 1])
    ]]  # Boundary of entire region
    
    print(f"\nNormal training strategy:")
    print(f"Single stage with all {N_pts} points")
    print(f"Region: [{stage_rects[0][0]:.3f}, {stage_rects[0][1]:.3f}, {stage_rects[0][2]:.3f}, {stage_rects[0][3]:.3f}]")

# Network - using standard PINN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
if torch.cuda.is_available():
    print(f"[INFO] Current CUDA device: {torch.cuda.current_device()}")
    print(f"[INFO] Device name: {torch.cuda.get_device_name()}")

# Create standard PINN model
class StandardPINNNet(PINNNet):
    def __init__(self, layers, lb, ub, L=570, U=6.57, DT=24.1, latitude_range=(58, 62)):
        super().__init__(layers, lb, ub, L, U, DT, latitude_range)
    
    def forward(self, x, y, t):
        # Ensure tensors are 2D for compatibility with parent class forward method
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Call parent class forward method
        return super().forward(x, y, t)
    
    def loss_data(self, x, y, t, u, v, p=None):
        u_pred, v_pred, _ = self.predict_uvp(x, y, t)
        mse = torch.nn.MSELoss()
        
        # Ensure predicted and target tensors have matching dimensions
        if u_pred.dim() != u.dim():
            if u_pred.dim() == 2 and u.dim() == 1:
                u_pred = u_pred.squeeze(1)
            elif u_pred.dim() == 1 and u.dim() == 2:
                u = u.squeeze(1)
        
        if v_pred.dim() != v.dim():
            if v_pred.dim() == 2 and v.dim() == 1:
                v_pred = v_pred.squeeze(1)
            elif v_pred.dim() == 1 and v.dim() == 2:
                v = v.squeeze(1)
        
        return mse(u_pred, u) + mse(v_pred, v)
        
    def loss_equation(self, x, y, t, epoch=None, total_epoch=None, beta=0.01, use_difficulty_weight=False, current_region=None, use_weak_continuity=False):
        """
        Standard PINN physics equation loss calculation
        Includes region constraints and optional difficulty weighting
        
        Args:
            x, y, t: spatial and temporal coordinates (non-dimensionalized)
            epoch: current epoch
            total_epoch: total number of epochs
            beta: difficulty weighting hyperparameter
            use_difficulty_weight: whether to use difficulty weighting
            current_region: current region (x_min, x_max, y_min, y_max)
            use_weak_continuity: whether to use weak continuity loss
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        # If current region is specified, apply region mask first
        if current_region is not None:
            x_min, x_max, y_min, y_max = current_region
            region_mask = ((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))
            if region_mask.sum() == 0:
                # If no points in region, return zero loss
                return torch.tensor(0.0, device=x.device, requires_grad=True)
            # Only use points within the region for calculation
            x = x[region_mask]
            y = y[region_mask]
            t = t[region_mask]
        
        u, v, p = self.predict_uvp(x, y, t)
        
        # First order derivatives
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        
        # Second order derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        
        # Calculate spatially varying Coriolis parameter
        # Convert non-dimensionalized y coordinate back to actual latitude
        y_physical = y * self.L  # Convert back to km
        lat_deg = torch.tensor(self.latitude_range[0], dtype=torch.float32, device=x.device) + \
                  (self.latitude_range[1] - self.latitude_range[0]) * (y_physical + self.L) / (2 * self.L)
        lat_rad = lat_deg * (torch.pi / 180.0)  # Convert degrees to radians
        f_local = 2 * self.omega * torch.sin(lat_rad)
        
        # Non-dimensionalized Coriolis parameter
        f_star = f_local / self.f_avg
        
        # Geometric factors in spherical coordinate system
        cos_lat = torch.cos(lat_rad)
        sin_lat = torch.sin(lat_rad)
        
        # Non-dimensionalized geometric factors
        cos_lat_star = cos_lat / cos_lat.mean()  # Normalized
        sin_lat_star = sin_lat / sin_lat.mean()  # Normalized
        
        # Continuity equation in spherical coordinate system
        f_mass = (1.0 / cos_lat_star) * u_x + v_y + (v / cos_lat_star) * sin_lat_star
        
        # Momentum equations in spherical coordinate system
        # λ-direction (longitude direction) momentum equation - u component
        f_lambda = (u_t + 
                   (u / cos_lat_star) * u_x + 
                   v * u_y + 
                   (u * v / cos_lat_star) * sin_lat_star - 
                   (1.0 / self.Re) * (u_xx + u_yy) + 
                   (1.0 / self.Ro) * f_star * v)
        
        # φ-direction (latitude direction) momentum equation - v component
        f_phi = (v_t + 
                (u / cos_lat_star) * v_x + 
                v * v_y - 
                (u * u / cos_lat_star) * sin_lat_star - 
                (1.0 / self.Re) * (v_xx + v_yy) - 
                (1.0 / self.Ro) * f_star * u)
        
        
        # Calculate basic PDE loss
        mse = torch.nn.MSELoss()
        zeros = torch.zeros_like(f_lambda)
        
        # Mass conservation loss
        if use_weak_continuity:
            # Weak continuity loss: use mean
            loss_mass = torch.mean(f_mass)**2
        else:
            # Normal point-wise PDE loss: use MSE
            loss_mass = mse(f_mass, zeros)
        
        # Momentum equation loss
        if use_difficulty_weight and epoch is not None and total_epoch is not None:
            # Calculate current stage training progress (between 0 and 1)
            stage_progress = epoch / total_epoch
            
            # Linear transition: gradually increase from negative to positive values
            # Early stage: beta is negative, simple samples have higher weights
            # Late stage: beta is positive, difficult samples have higher weights
            beta_positive = BETA * (2 * stage_progress - 1)  # From -BETA to +BETA
            
            # Calculate difficulty weights for both momentum equations separately
            difficulty_weight_lambda = torch.exp(beta_positive * torch.abs(f_lambda))
            difficulty_weight_phi = torch.exp(beta_positive * torch.abs(f_phi))
            
            # Normalize weights
            difficulty_weight_lambda = difficulty_weight_lambda / difficulty_weight_lambda.mean()
            difficulty_weight_phi = difficulty_weight_phi / difficulty_weight_phi.mean()
            
            # Apply weights to momentum equation loss
            if use_weak_continuity:
                # Weak continuity loss: use mean
                loss_momentum = torch.mean(difficulty_weight_lambda * f_lambda**2) + \
                               torch.mean(difficulty_weight_phi * f_phi**2)
            else:
                # Normal point-wise PDE loss: use MSE
                loss_momentum = torch.mean(difficulty_weight_lambda * (f_lambda - zeros)**2) + \
                               torch.mean(difficulty_weight_phi * (f_phi - zeros)**2)
        else:
            # Standard loss without difficulty weighting
            if use_weak_continuity:
                # Weak continuity loss: use mean
                loss_momentum = torch.mean(f_lambda)**2 + torch.mean(f_phi)**2
            else:
                # Normal point-wise PDE loss: use MSE
                loss_momentum = mse(f_lambda, zeros) + mse(f_phi, zeros)
        
        loss_pde = loss_mass + loss_momentum
        
        return loss_pde

model = StandardPINNNet(LAYERS, lb, ub, L=L, U=U, DT=DT, latitude_range=latitude_range).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler configuration
def create_lr_scheduler(optimizer, total_epochs, initial_lr=LR):
    """Create learning rate scheduler"""
    # Use cosine annealing scheduler, gradually reduce learning rate during training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs,
        eta_min=initial_lr * 0.01  # Minimum learning rate is 1% of initial value
    )
    return scheduler

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

# Record validation metrics
val_metrics = []

# Training parameters (read from configuration)


print(f"\n=== Ablation Study Configuration ===")
print(f"Region Growth: {'Enabled' if USE_REGION_GROWTH else 'Disabled'}")
print(f"Weak Continuity: {'Enabled' if USE_WEAK_CONTINUITY else 'Disabled'}")
print(f"Difficulty Weighting: {'Enabled' if USE_DIFFICULTY_WEIGHT else 'Disabled'}")

print(f"\n=== Training Configuration ===")
print(f"Training for {STAGES} stages, {EPOCHS_PER_STAGE} epochs per stage (last stage +{EXTRA_EPOCHS_LAST_STAGE})")
print(f"Total epochs: {STAGES * EPOCHS_PER_STAGE + EXTRA_EPOCHS_LAST_STAGE}")
print(f"Epochs per stage: {EPOCHS_PER_STAGE} (last stage: {EPOCHS_PER_STAGE + EXTRA_EPOCHS_LAST_STAGE})")
print(f"Total training epochs: {EPOCHS + EXTRA_EPOCHS_LAST_STAGE}")
print(f"Epoch distribution: {STAGES-1} stages × {EPOCHS_PER_STAGE} + 1 stage × {EPOCHS_PER_STAGE + EXTRA_EPOCHS_LAST_STAGE}")
print(f"Learning rate: {LR} (reset to {LR} at each stage start)")
print(f"Learning rate scheduler: CosineAnnealingLR (min_lr = {LR * 0.01:.6f})")
print(f"Standard PINN model")
if USE_DIFFICULTY_WEIGHT:
    print(f"KAN-style exponential difficulty weighting enabled (beta: {BETA})")
else:
    print(f"Standard PDE loss (no difficulty weighting)")

# Multi-stage training
stage_start_time = time.time()
for stage in range(STAGES):
    stage_stage_time = time.time()
    print(f"\n=== Stage {stage+1}/{STAGES} ===")
    
    # Get spatial point indices for current stage
    stage_space_idx = stage_indices[stage]
    
    # Create data mask for current stage
    stage_mask = np.zeros(len(x_all), dtype=bool)
    for i, (x_val, y_val) in enumerate(zip(x_all, y_all)):
        for space_idx in stage_space_idx:
            if np.allclose([x_val, y_val], coords[space_idx]):
                stage_mask[i] = True
                break
    
    # Extract data for current stage
    x_stage = x_all[stage_mask]
    y_stage = y_all[stage_mask]
    t_stage = t_all[stage_mask]
    u_stage = u_all[stage_mask]
    v_stage = v_all[stage_mask]
    
    print(f"Stage {stage+1} data points: {len(x_stage)}")
    
    # Build Dataset - adapt to new data format
    class StageDataset(Dataset):
        def __len__(self):
            return len(x_stage)
        def __getitem__(self, idx):
            return {
                'x': torch.tensor(x_stage[idx], dtype=torch.float32),
                'y': torch.tensor(y_stage[idx], dtype=torch.float32),
                't': torch.tensor(t_stage[idx], dtype=torch.float32),
                'u': torch.tensor(u_stage[idx], dtype=torch.float32),
                'v': torch.tensor(v_stage[idx], dtype=torch.float32)
            }
    
    train_loader = DataLoader(StageDataset(), batch_size=BATCH_SIZE, shuffle=True)
    
    # Train current stage
    # Last stage has additional epochs
    current_stage_epochs = EPOCHS_PER_STAGE + (EXTRA_EPOCHS_LAST_STAGE if stage == STAGES - 1 else 0)
    print(f"Stage {stage+1}: Training for {current_stage_epochs} epochs (base: {EPOCHS_PER_STAGE}" + (f" + extra: {EXTRA_EPOCHS_LAST_STAGE}" if stage == STAGES - 1 else "") + ")")
    
    # Reset learning rate to initial value at the start of each stage
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    print(f"Stage {stage+1}: Learning rate reset to {LR}")
    
    # Create learning rate scheduler for current stage
    scheduler = create_lr_scheduler(optimizer, current_stage_epochs, LR)
    
    # Create progress bar
    pbar = tqdm(range(current_stage_epochs), desc=f"Stage {stage+1}", leave=True)
    
    for epoch in pbar:
        epoch_start_time = time.time()
        # Calculate current global epoch (considering additional epochs in last stage)
        if stage == STAGES - 1:
            # Last stage: epochs from all previous stages + current stage epochs
            global_epoch = (STAGES - 1) * EPOCHS_PER_STAGE + epoch + 1
        else:
            # Other stages: normal calculation
            global_epoch = stage * EPOCHS_PER_STAGE + epoch + 1
        
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            # New data format: dictionary format
            x_b = batch['x'].to(device)
            y_b = batch['y'].to(device)
            t_b = batch['t'].to(device)
            u_b = batch['u'].to(device)
            v_b = batch['v'].to(device)
            
            optimizer.zero_grad()
            loss_data = model.loss_data(x_b, y_b, t_b, u_b, v_b)
            
            # Physics equation loss calculated on all training points in current region
            # Get current stage region boundary
            current_region = stage_rects[stage] if stage < STAGES - 1 else None  # Last stage uses entire plane
            loss_eq = model.loss_equation(x_b, y_b, t_b, epoch=epoch, total_epoch=current_stage_epochs, 
                                        use_difficulty_weight=USE_DIFFICULTY_WEIGHT, 
                                        current_region=current_region, use_weak_continuity=USE_WEAK_CONTINUITY)
            loss = loss_data + loss_eq
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Update progress bar description (including current learning rate)
        pbar.set_description(f"Stage {stage+1} | Epoch {epoch+1}/{current_stage_epochs} | Loss: {train_loss:.4e} | LR: {current_lr:.6f}")
        
        # Validation
        if (global_epoch) % VAL_EVAL_EPOCH == 0 or (stage == STAGES-1 and epoch == current_stage_epochs-1):
            model.eval()
            val_loss = 0
            rel_u, rel_v = 0, 0
            rmse_u, rmse_v = 0, 0
            mape_u, mape_v = 0, 0
            acc_u, acc_v = 0, 0
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    # New data format: dictionary format
                    x_v = batch['x'].to(device)
                    y_v = batch['y'].to(device)
                    t_v = batch['t'].to(device)
                    u_v = batch['u'].to(device)
                    v_v = batch['v'].to(device)
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
            val_metrics.append([
                global_epoch, train_loss, val_loss,
                rel_u/n, rel_v/n,
                rmse_u/n, rmse_v/n,
                mape_u/n, mape_v/n,
                acc_u/n, acc_v/n
            ])
            
            # Print training information
            with torch.no_grad():
                # Calculate current dynamic beta value
                stage_progress = epoch / current_stage_epochs
                current_beta = BETA * (2 * stage_progress - 1)  # From -BETA to +BETA
                
                tqdm.write(f"Stage {stage+1}, Epoch {epoch+1}/{current_stage_epochs} ({epoch_time:.1f}s) - train_loss={train_loss:.4e}, val_loss={val_loss:.4e}")
                if USE_DIFFICULTY_WEIGHT:
                    tqdm.write(f"  Dynamic difficulty beta: {current_beta:.4f} (progress: {stage_progress:.2f})")
                if USE_WEAK_CONTINUITY:
                    tqdm.write(f"  Weak continuity loss enabled")
                tqdm.write(f"  Current learning rate: {current_lr:.6f}")
    
    pbar.close()
    
    stage_time = time.time() - stage_stage_time
    print(f"Stage {stage+1} completed in {stage_time:.1f}s")

# Save model and validation metrics
# Create temporary folder first, rename after training completion
# Generate folder name based on ablation study configuration
config_suffix = []
if USE_REGION_GROWTH:
    config_suffix.append("RG")
else:
    config_suffix.append("NRG")  # No Region Growth

if USE_WEAK_CONTINUITY:
    config_suffix.append("WC")
else:
    config_suffix.append("NWC")  # No Weak Continuity

if USE_DIFFICULTY_WEIGHT:
    config_suffix.append(f"DW_beta={BETA:.3f}")
else:
    config_suffix.append("NDW")  # No Difficulty Weighting

config_str = "_".join(config_suffix)
temp_checkpoint_dir = f'checkpoints_cerra_{N_POINTS}_{config_str}_PINN_temp'

os.makedirs(temp_checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), f'{temp_checkpoint_dir}/pinn_region_grow.pth')

# Save training configuration information
training_config = {
    'use_region_growth': USE_REGION_GROWTH,
    'use_weak_continuity': USE_WEAK_CONTINUITY,
    'use_difficulty_weight': USE_DIFFICULTY_WEIGHT,
}
    
# Save beta information only when using difficulty weighting
if USE_DIFFICULTY_WEIGHT:
    training_config['fixed_beta'] = BETA
    
np.save(f'{temp_checkpoint_dir}/training_config.npy', training_config)

with open(f'{temp_checkpoint_dir}/val_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'train_loss', 'val_loss',
        'RelL2_u', 'RelL2_v',
        'RMSE_u', 'RMSE_v',
        'MAPE_u', 'MAPE_v',
        'acc_u', 'acc_v'])
    writer.writerows(val_metrics)

# Rename folder based on ablation study configuration
final_checkpoint_dir = f'checkpoints_cerra_{N_POINTS}_{config_str}_PINN'

if os.path.exists(temp_checkpoint_dir):
    if os.path.exists(final_checkpoint_dir):
        import shutil
        shutil.rmtree(final_checkpoint_dir)
    os.rename(temp_checkpoint_dir, final_checkpoint_dir)
    print(f'Checkpoint folder renamed to: {final_checkpoint_dir}')

total_time = time.time() - stage_start_time
print(f'Training finished in {total_time:.1f}s. Model and validation metrics saved.')
print(f'Ablation study configuration: {config_str}')
if USE_DIFFICULTY_WEIGHT:
    print(f'Final KAN-style difficulty beta: {BETA:.4f}')
print(f'Results saved to: {final_checkpoint_dir}') 
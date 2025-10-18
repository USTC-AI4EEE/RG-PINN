#!/usr/bin/env python3
"""
CERRA Dataset PyTorch Data Loader
For reading NPY format datasets generated from GRIB
Supports non-dimensionalization and normalization
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class CERRAWindFieldDataset(Dataset):
    """CERRA Wind Field Dataset"""
    
    def __init__(self, dataset_dir, normalize=True, nondimensionalize=True, 
                 L=60, U=8, DT=1, mode='train'):
        """
        Initialize dataset
        
        Args:
            dataset_dir (str): Dataset directory path
            normalize (bool): Whether to normalize data
            nondimensionalize (bool): Whether to non-dimensionalize data
            L (float): Characteristic length scale (km)
            U (float): Characteristic velocity scale (m/s)
            DT (float): Characteristic time scale (hours)
            mode (str): 'train' or 'val', specify loading training or validation set
        """
        self.dataset_dir = Path(dataset_dir)
        self.normalize = normalize
        self.nondimensionalize = nondimensionalize
        self.L = L
        self.U = U
        self.DT = DT
        self.mode = mode
        self.data = None
        self.norm_params = None
        self.nondim_params = None
        
        # Load data
        self.load_data()
        
        # Non-dimensionalization (before normalization)
        if self.nondimensionalize:
            self.compute_nondimensionalization_params()
            self.nondimensionalize_data()
        
        # Compute normalization parameters
        if self.normalize:
            self.compute_normalization_params()
            self.normalize_data()
    
    def load_data(self):
        """Load data from NPY files"""
        print(f"Loading {self.mode} data from: {self.dataset_dir}")
        
        if self.mode == 'train':
            # Load training set
            train_x = np.load(self.dataset_dir / "train_x.npy")  # (n_points,)
            train_y = np.load(self.dataset_dir / "train_y.npy")  # (n_points,)
            train_t = np.load(self.dataset_dir / "train_t.npy")  # (n_time,)
            train_lats = np.load(self.dataset_dir / "train_lats.npy")  # (n_points,)
            train_lons = np.load(self.dataset_dir / "train_lons.npy")  # (n_points,)
            
            # Load wind field data from validation set to get wind field values at training points
            val_u = np.load(self.dataset_dir / "val_u.npy")  # (n_time, n_lats, n_lons)
            val_v = np.load(self.dataset_dir / "val_v.npy")  # (n_time, n_lats, n_lons)
            
            # Process time data: if datetime64, convert to hours
            if train_t.dtype == np.dtype('datetime64[ns]'):
                # Convert to hours relative to first time point
                start_time = train_t[0]
                train_t_hours = (train_t - start_time).astype('timedelta64[h]').astype(float)
            else:
                train_t_hours = train_t
            
            # Create samples for all time steps for each training point
            n_points = len(train_x)
            n_time = len(train_t_hours)
            
            # Expand spatial coordinates to all time steps
            x_expanded = np.tile(train_x[:, None], (1, n_time)).flatten()  # (n_points * n_time,)
            y_expanded = np.tile(train_y[:, None], (1, n_time)).flatten()  # (n_points * n_time,)
            lats_expanded = np.tile(train_lats[:, None], (1, n_time)).flatten()  # (n_points * n_time,)
            lons_expanded = np.tile(train_lons[:, None], (1, n_time)).flatten()  # (n_points * n_time,)
            
            # Expand time to all points
            t_expanded = np.tile(train_t_hours[None, :], (n_points, 1)).flatten()  # (n_points * n_time,)
            
            # Get wind field data corresponding to training points
            u_expanded = np.zeros(n_points * n_time)
            v_expanded = np.zeros(n_points * n_time)
            
            for i in range(n_points):
                for j in range(n_time):
                    idx = i * n_time + j
                    u_expanded[idx] = val_u[j, train_y[i], train_x[i]]
                    v_expanded[idx] = val_v[j, train_y[i], train_x[i]]
            
            self.data = {
                'x': x_expanded,
                'y': y_expanded,
                't': t_expanded,
                'u': u_expanded,
                'v': v_expanded,
                'lats': lats_expanded,
                'lons': lons_expanded
            }
            print(f"Training data loaded: {len(self.data['x'])} samples ({n_points} points × {n_time} time steps)")
            print(f"Training wind data loaded: u and v for all training points")
            
        elif self.mode == 'val':
            # Load validation set
            val_coords = np.load(self.dataset_dir / "val_coords.npy")  # (n_lats, n_lons, 2)
            val_t = np.load(self.dataset_dir / "val_t.npy")
            val_u = np.load(self.dataset_dir / "val_u.npy")  # (n_time, n_lats, n_lons)
            val_v = np.load(self.dataset_dir / "val_v.npy")  # (n_time, n_lats, n_lons)
            
            # Process time data: if datetime64, convert to hours
            if val_t.dtype == np.dtype('datetime64[ns]'):
                # Convert to hours relative to first time point
                start_time = val_t[0]
                val_t_hours = (val_t - start_time).astype('timedelta64[h]').astype(float)
            else:
                val_t_hours = val_t
            
            # Create grid indices
            n_lats, n_lons = val_coords.shape[0], val_coords.shape[1]
            n_time = len(val_t_hours)
            
            # Create complete spatiotemporal grid
            x_indices, y_indices = np.meshgrid(np.arange(n_lons), np.arange(n_lats), indexing='ij')
            x_indices = x_indices.T  # (n_lats, n_lons)
            y_indices = y_indices.T  # (n_lats, n_lons)
            
            # Expand time dimension
            x_grid = np.tile(x_indices[None, :, :], (n_time, 1, 1))  # (n_time, n_lats, n_lons)
            y_grid = np.tile(y_indices[None, :, :], (n_time, 1, 1))  # (n_time, n_lats, n_lons)
            t_grid = np.tile(val_t_hours[:, None, None], (1, n_lats, n_lons))  # (n_time, n_lats, n_lons)
            
            # Expand coordinate dimensions
            lons_grid = np.tile(val_coords[None, :, :, 0], (n_time, 1, 1))  # (n_time, n_lats, n_lons)
            lats_grid = np.tile(val_coords[None, :, :, 1], (n_time, 1, 1))  # (n_time, n_lats, n_lons)
            
            # Flatten all dimensions
            self.data = {
                'x': x_grid.flatten(),
                'y': y_grid.flatten(),
                't': t_grid.flatten(),
                'u': val_u.flatten(),
                'v': val_v.flatten(),
                'lats': lats_grid.flatten(),
                'lons': lons_grid.flatten()
            }
            
            print(f"Validation data loaded: {len(self.data['x'])} points")
            print(f"Original grid shape: {n_time} x {n_lats} x {n_lons}")
        
        print(f"Data shapes: x={self.data['x'].shape}, y={self.data['y'].shape}, t={self.data['t'].shape}")
        if 'u' in self.data:
            print(f"Wind data shapes: u={self.data['u'].shape}, v={self.data['v'].shape}")
    
    def compute_nondimensionalization_params(self):
        """Compute non-dimensionalization parameters"""
        print("Computing nondimensionalization parameters...")
        
        # Characteristic scales
        self.nondim_params = {
            'L': self.L,      # Characteristic length (km)
            'U': self.U,      # Characteristic velocity (m/s)
            'DT': self.DT,    # Characteristic time (hours)
        }
        
        print(f"  Length scale: {self.L} km")
        print(f"  Velocity scale: {self.U} m/s")
        print(f"  Time scale: {self.DT} hours")
    
    def nondimensionalize_data(self):
        """Non-dimensionalize data"""
        print("Nondimensionalizing data...")
        
        # Non-dimensionalization formulas:
        # x* = x / L, y* = y / L, t* = t / (L/U), u* = u / U, v* = v / U
        
        # Spatial coordinate non-dimensionalization (grid indices -> non-dimensional)
        # Note: Here we assume grid spacing is 1, in actual applications may need adjustment based on physical distance
        self.data['x'] = self.data['x'] / self.L
        self.data['y'] = self.data['y'] / self.L
        
        # Time non-dimensionalization (hours -> non-dimensional)
        self.data['t'] = self.data['t'] / (self.L / self.U)
        
        # Velocity non-dimensionalization (m/s -> non-dimensional)
        if 'u' in self.data:
            self.data['u'] = self.data['u'] / self.U
            self.data['v'] = self.data['v'] / self.U
        
        print("Data nondimensionalized successfully!")
    
    def compute_normalization_params(self):
        """Compute normalization parameters (min-max normalization)"""
        print("Computing normalization parameters...")
        
        self.norm_params = {}
        for key in ['x', 'y', 't']:
            if key in self.data:
                data = self.data[key].flatten()
                min_val = np.min(data)
                max_val = np.max(data)
                
                # Avoid division by zero
                if max_val == min_val:
                    max_val = min_val + 1.0
                
                self.norm_params[key] = {
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val
                }
                
                print(f"  {key}: [{min_val:.4f}, {max_val:.4f}]")
        
        # For wind field data (if exists)
        if 'u' in self.data:
            for key in ['u', 'v']:
                data = self.data[key].flatten()
                min_val = np.min(data)
                max_val = np.max(data)
                
                if max_val == min_val:
                    max_val = min_val + 1.0
                
                self.norm_params[key] = {
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val
                }
                
                print(f"  {key}: [{min_val:.4f}, {max_val:.4f}]")
    
    def normalize_data(self):
        """Normalize data"""
        print("Normalizing data...")
        
        for key in ['x', 'y', 't']:
            if key in self.data and key in self.norm_params:
                params = self.norm_params[key]
                self.data[key] = (self.data[key] - params['min']) / params['range']
        
        # Normalize wind field data
        if 'u' in self.data:
            for key in ['u', 'v']:
                if key in self.norm_params:
                    params = self.norm_params[key]
                    self.data[key] = (self.data[key] - params['min']) / params['range']
    
    def denormalize_data(self, normalized_data):
        """Denormalize data"""
        if not self.normalize or self.norm_params is None:
            return normalized_data
        
        denormalized = {}
        for key in ['x', 'y', 't', 'u', 'v']:
            if key in normalized_data and key in self.norm_params:
                params = self.norm_params[key]
                denormalized[key] = normalized_data[key] * params['range'] + params['min']
        
        return denormalized
    
    def redimensionalize_data(self, nondimensionalized_data):
        """Re-dimensionalize data"""
        if not self.nondimensionalize or self.nondim_params is None:
            return nondimensionalized_data
        
        redimensionalized = {}
        
        # Re-dimensionalize spatial coordinates
        if 'x' in nondimensionalized_data:
            redimensionalized['x'] = nondimensionalized_data['x'] * self.L
        if 'y' in nondimensionalized_data:
            redimensionalized['y'] = nondimensionalized_data['y'] * self.L
        
        # Re-dimensionalize time
        if 't' in nondimensionalized_data:
            redimensionalized['t'] = nondimensionalized_data['t'] * (self.L / self.U)
        
        # Re-dimensionalize velocity
        if 'u' in nondimensionalized_data:
            redimensionalized['u'] = nondimensionalized_data['u'] * self.U
        if 'v' in nondimensionalized_data:
            redimensionalized['v'] = nondimensionalized_data['v'] * self.U
        
        return redimensionalized
    
    def __len__(self):
        """Return dataset size"""
        return len(self.data['x'])
    
    def __getitem__(self, idx):
        """Get single sample"""
        sample = {
            'x': torch.tensor(self.data['x'][idx], dtype=torch.float32).unsqueeze(0),
            'y': torch.tensor(self.data['y'][idx], dtype=torch.float32).unsqueeze(0),
            't': torch.tensor(self.data['t'][idx], dtype=torch.float32).unsqueeze(0),
            'lats': torch.tensor(self.data['lats'][idx], dtype=torch.float32).unsqueeze(0),
            'lons': torch.tensor(self.data['lons'][idx], dtype=torch.float32).unsqueeze(0)
        }
        
        # Add wind field data (if exists)
        if 'u' in self.data:
            sample['u'] = torch.tensor(self.data['u'][idx], dtype=torch.float32).unsqueeze(0)
            sample['v'] = torch.tensor(self.data['v'][idx], dtype=torch.float32).unsqueeze(0)
        
        return sample
    
    def get_input_tensor(self, idx):
        """Get input tensor (x, y, t)"""
        return torch.cat([
            torch.tensor(self.data['x'][idx], dtype=torch.float32),
            torch.tensor(self.data['y'][idx], dtype=torch.float32), 
            torch.tensor(self.data['t'][idx], dtype=torch.float32)
        ], dim=0).float()
    
    def get_output_tensor(self, idx):
        """Get output tensor (u, v)"""
        if 'u' in self.data and 'v' in self.data:
            return torch.cat([
                torch.tensor(self.data['u'][idx], dtype=torch.float32),
                torch.tensor(self.data['v'][idx], dtype=torch.float32)
            ], dim=0).float()
        else:
            return torch.tensor([], dtype=torch.float32)
    
    def get_all_inputs(self):
        """Get all input data"""
        x_tensor = torch.tensor(self.data['x'], dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(self.data['y'], dtype=torch.float32).unsqueeze(1)
        t_tensor = torch.tensor(self.data['t'], dtype=torch.float32).unsqueeze(1)
        
        return torch.cat([x_tensor, y_tensor, t_tensor], dim=1)
    
    def get_all_outputs(self):
        """Get all output data"""
        if 'u' in self.data and 'v' in self.data:
            u_tensor = torch.tensor(self.data['u'], dtype=torch.float32).unsqueeze(1)
            v_tensor = torch.tensor(self.data['v'], dtype=torch.float32).unsqueeze(1)
            return torch.cat([u_tensor, v_tensor], dim=1)
        else:
            return torch.tensor([], dtype=torch.float32)
    
    def get_latitude_range(self):
        """Get latitude range"""
        if 'lats' in self.data:
            lats = self.data['lats']
            return float(lats.min()), float(lats.max())
        else:
            # If no latitude data, return default range
            return (35.0, 65.0)

def create_cerra_dataloader(dataset_dir, batch_size=128, shuffle=True, 
                           normalize=True, nondimensionalize=True,
                           L=60, U=8, DT=1, num_workers=0, mode='train'):
    """
    Create CERRA data loader
    
    Args:
        dataset_dir (str): Dataset directory path
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        normalize (bool): Whether to normalize
        nondimensionalize (bool): Whether to non-dimensionalize
        L (float): Characteristic length scale (km)
        U (float): Characteristic velocity scale (m/s)
        DT (float): Characteristic time scale (hours)
        num_workers (int): Number of worker processes
        mode (str): 'train' or 'val'
    
    Returns:
        DataLoader: PyTorch data loader
        CERRAWindFieldDataset: Dataset object
    """
    dataset = CERRAWindFieldDataset(
        dataset_dir, 
        normalize=normalize, 
        nondimensionalize=nondimensionalize,
        L=L, U=U, DT=DT, mode=mode
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader, dataset

def test_dataloader(dataset_dir="dataset_new_format", L=60, U=8, DT=1):
    """Test data loader functionality"""
    print("Testing CERRA Data Loader")
    print("=" * 50)
    
    try:
        # Test training set
        print(f"\nTesting training dataset: {dataset_dir}")
        train_loader, train_dataset = create_cerra_dataloader(
            dataset_dir=dataset_dir,
            batch_size=16,
            shuffle=True,
            normalize=True,
            nondimensionalize=True,
            L=L, U=U, DT=DT,
            mode='train'
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Number of training batches: {len(train_loader)}")
        
        # Get a training batch
        train_batch = next(iter(train_loader))
        print(f"Training batch keys: {train_batch.keys()}")
        print(f"Training batch shapes:")
        for key, value in train_batch.items():
            print(f"  {key}: {value.shape}")
        
        # Test validation set
        print(f"\nTesting validation dataset: {dataset_dir}")
        val_loader, val_dataset = create_cerra_dataloader(
            dataset_dir=dataset_dir,
            batch_size=32,
            shuffle=False,
            normalize=True,
            nondimensionalize=True,
            L=L, U=U, DT=DT,
            mode='val'
        )
        
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        # Get a validation batch
        val_batch = next(iter(val_loader))
        print(f"Validation batch keys: {val_batch.keys()}")
        print(f"Validation batch shapes:")
        for key, value in val_batch.items():
            print(f"  {key}: {value.shape}")
        
        # Test input/output tensors
        train_inputs = train_dataset.get_all_inputs()
        train_outputs = train_dataset.get_all_outputs()
        print(f"Training input tensor shape: {train_inputs.shape}")
        print(f"Training output tensor shape: {train_outputs.shape}")
        
        val_inputs = val_dataset.get_all_inputs()
        val_outputs = val_dataset.get_all_outputs()
        print(f"Validation input tensor shape: {val_inputs.shape}")
        print(f"Validation output tensor shape: {val_outputs.shape}")
        
        # Test non-dimensionalization parameters
        if train_dataset.nondimensionalize:
            print("Nondimensionalization parameters:")
            for key, value in train_dataset.nondim_params.items():
                print(f"  {key}: {value}")
        
        # Test normalization parameters
        if train_dataset.normalize:
            print("Normalization parameters:")
            for key, params in train_dataset.norm_params.items():
                print(f"  {key}: min={params['min']:.4f}, max={params['max']:.4f}")
        
        print(f"✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='CERRA Data Loader Test')
    parser.add_argument('--dataset_dir', type=str, default='dataset_new_format', 
                       help='Dataset directory path')
    parser.add_argument('--L', type=float, default=60, 
                       help='Characteristic length scale (km)')
    parser.add_argument('--U', type=float, default=8, 
                       help='Characteristic velocity scale (m/s)')
    parser.add_argument('--DT', type=float, default=1, 
                       help='Characteristic time scale (hours)')
    parser.add_argument('--test', action='store_true', 
                       help='Run test')
    
    args = parser.parse_args()
    
    if args.test:
        test_dataloader(args.dataset_dir, args.L, args.U, args.DT)
    else:
        print("Use --test flag to run the test")

if __name__ == "__main__":
    main() 
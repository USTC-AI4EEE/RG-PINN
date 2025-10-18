import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

class BoundaryLayerDataset(Dataset):
    def __init__(self, h5_path, normalize=True, L=3.0, U=3.76, T=0.79787, 
                 lb=None, ub=None, u_mean=2.5, v_mean=0.0, u_std=1.5, v_std=0.5):
        """
        Load boundary layer dataset
        Args:
            h5_path: HDF5 file path for training or validation set
            normalize: Whether to perform non-dimensionalization and standardization
            L: Characteristic length (m)
            U: Characteristic velocity (m/s)
            T: Characteristic time (s)
            lb: Standardized lower bound [x*, y*, t*]
            ub: Standardized upper bound [x*, y*, t*]
            u_mean, v_mean: Velocity standardization mean
            u_std, v_std: Velocity standardization standard deviation
        """
        # Non-dimensionalization parameters
        self.L = L
        self.U = U
        self.T = T
        
        # Standardization parameters
        if lb is None:
            self.lb = np.array([0.0, 0.0, 0.0])
        else:
            self.lb = np.array(lb)
            
        if ub is None:
            self.ub = np.array([2.0, 1.0, 2.066])
        else:
            self.ub = np.array(ub)
        
        # Velocity standardization parameters
        self.u_mean = u_mean
        self.v_mean = v_mean
        self.u_std = u_std
        self.v_std = v_std
        
        self.normalize = normalize

        with h5py.File(h5_path, 'r') as f:
            # Check if it's training set or validation set
            if 'x' in f and 'y' in f and 't' in f:
                # Training set: contains data from fixed observation stations
                self.x = np.array(f['x']).flatten()[:, None]
                self.y = np.array(f['y']).flatten()[:, None]
                self.t = np.array(f['t']).flatten()
                self.u = np.array(f['ux']).flatten()  # Note: original data is ux, changed to u here
                self.v = np.array(f['uy']).flatten()  # Note: original data is uy, changed to v here
                
                # Boundary layer data has no pressure field, set to None
                self.p = None
                
                # Training set data is already in flattened format, no additional processing needed
                self.x = self.x
                self.y = self.y
                self.t = self.t[:, None]
                self.u = self.u[:, None]
                self.v = self.v[:, None]
                
            else:
                # Validation set: contains complete grid data
                self.x_grid = np.array(f['x_grid'])
                self.y_grid = np.array(f['y_grid'])
                self.t_grid = np.array(f['t_grid'])
                self.u_full = np.array(f['ux_full'])  # Note: original data is ux, changed to u here
                self.v_full = np.array(f['uy_full'])  # Note: original data is uy, changed to v here
                
                # Create complete grid coordinates
                X, Y = np.meshgrid(self.x_grid, self.y_grid)
                self.X_mesh = X
                self.Y_mesh = Y
                
                # Validation set data maintains grid format for evaluation
                self.p = None
                
                # For Dataset interface compatibility, create flattened validation data
                N_x, N_y = len(self.x_grid), len(self.y_grid)
                N_t = len(self.t_grid)
                
                # Create time series for all grid points
                # Ensure time dimension is correctly expanded
                x_all = np.tile(X.flatten(), (N_t, 1)).T.flatten()[:, None]
                y_all = np.tile(Y.flatten(), (N_t, 1)).T.flatten()[:, None]
                t_all = np.repeat(self.t_grid, N_x * N_y)[:, None]
                u_all = self.u_full.reshape(N_t, -1).T.flatten()[:, None]
                v_all = self.v_full.reshape(N_t, -1).T.flatten()[:, None]
                
                # Validate data shape
                assert len(x_all) == N_x * N_y * N_t, f"Expected {N_x * N_y * N_t}, got {len(x_all)}"
                assert len(t_all) == N_x * N_y * N_t, f"Expected {N_x * N_y * N_t}, got {len(t_all)}"
                
                self.x = x_all
                self.y = y_all
                self.t = t_all
                self.u = u_all
                self.v = v_all
        
        # Perform non-dimensionalization and standardization
        if self.normalize:
            self._normalize_data()
    
    def _normalize_data(self):
        """Non-dimensionalize and standardize data"""
        print(f"[INFO] Normalizing data with parameters:")
        print(f"  L = {self.L} m, U = {self.U} m/s, T = {self.T} s")
        print(f"  lb = {self.lb}, ub = {self.ub}")
        
        # Non-dimensionalize coordinates and time
        self.x_original = self.x.copy()
        self.y_original = self.y.copy()
        self.t_original = self.t.copy()
        self.u_original = self.u.copy()
        self.v_original = self.v.copy()
        
        # Non-dimensionalization: x* = x/L, y* = y/L, t* = t/T
        self.x = self.x / self.L
        self.y = self.y / self.L
        self.t = self.t / self.T
        
        # Non-dimensionalize velocity: u* = u/U, v* = v/U
        self.u = self.u / self.U
        self.v = self.v / self.U
        
        # Standardize to [0,1] range (based on lb and ub)
        # For coordinates and time, directly use lb and ub for linear transformation
        # For velocity, use Z-score standardization
        self.x = (self.x - self.lb[0]) / (self.ub[0] - self.lb[0])
        self.y = (self.y - self.lb[1]) / (self.ub[1] - self.lb[1])
        self.t = (self.t - self.lb[2]) / (self.ub[2] - self.lb[2])
        
        # Velocity standardization (Z-score)
        self.u = (self.u - self.u_mean) / self.u_std
        self.v = (self.v - self.v_mean) / self.v_std
        
        print(f"[INFO] Normalization completed")
        print(f"  Original ranges: x[{self.x_original.min():.3f}, {self.x_original.max():.3f}], "
              f"y[{self.y_original.min():.3f}, {self.y_original.max():.3f}], "
              f"t[{self.t_original.min():.3f}, {self.t_original.max():.3f}]")
        print(f"  Normalized ranges: x[{self.x.min():.3f}, {self.x.max():.3f}], "
              f"y[{self.y.min():.3f}, {self.y.max():.3f}], "
              f"t[{self.t.min():.3f}, {self.t.max():.3f}]")
    
    def denormalize_coordinates(self, x_norm, y_norm, t_norm):
        """Denormalize coordinates and time"""
        # Denormalize to non-dimensional space
        x_dimless = x_norm * (self.ub[0] - self.lb[0]) + self.lb[0]
        y_dimless = y_norm * (self.ub[1] - self.lb[1]) + self.lb[1]
        t_dimless = t_norm * (self.ub[2] - self.lb[2]) + self.lb[2]
        
        # Reverse non-dimensionalization to physical space
        x_phys = x_dimless * self.L
        y_phys = y_dimless * self.L
        t_phys = t_dimless * self.T
        
        return x_phys, y_phys, t_phys
    
    def denormalize_velocity(self, u_norm, v_norm):
        """Denormalize velocity"""
        # Reverse Z-score standardization
        u_dimless = u_norm * self.u_std + self.u_mean
        v_dimless = v_norm * self.v_std + self.v_mean
        
        # Reverse non-dimensionalization to physical space
        u_phys = u_dimless * self.U
        v_phys = v_dimless * self.U
        
        return u_phys, v_phys

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        if self.p is not None:
            return (torch.tensor(self.x[idx], dtype=torch.float32),
                    torch.tensor(self.y[idx], dtype=torch.float32),
                    torch.tensor(self.t[idx], dtype=torch.float32),
                    torch.tensor(self.u[idx], dtype=torch.float32),
                    torch.tensor(self.v[idx], dtype=torch.float32),
                    torch.tensor(self.p[idx], dtype=torch.float32))
        else:
            return (torch.tensor(self.x[idx], dtype=torch.float32),
                    torch.tensor(self.y[idx], dtype=torch.float32),
                    torch.tensor(self.t[idx], dtype=torch.float32),
                    torch.tensor(self.u[idx], dtype=torch.float32),
                    torch.tensor(self.v[idx], dtype=torch.float32))

class WindFieldDataset(Dataset):
    """Keep original WindFieldDataset class for compatibility with existing code"""
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.x = np.array(f['x_all']).flatten()[:, None]
            self.y = np.array(f['y_all']).flatten()[:, None]
            self.t = np.array(f['t_all']).flatten()
            self.u = np.array(f['u_all'])
            self.v = np.array(f['v_all'])
            # p_all may not exist in sparse data, need compatibility
            self.p = np.array(f['p_all']) if 'p_all' in f else None

        # Process into (N, 1) observation points × time
        N = self.x.shape[0]
        T = self.t.shape[0]
        self.x = np.tile(self.x, (T, 1))
        self.y = np.tile(self.y, (T, 1))
        self.t = np.repeat(self.t, N)[:, None]
        self.u = self.u.flatten()[:, None]
        self.v = self.v.flatten()[:, None]
        if self.p is not None:
            self.p = self.p.flatten()[:, None]

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        if self.p is not None:
            return (torch.tensor(self.x[idx], dtype=torch.float32),
                    torch.tensor(self.y[idx], dtype=torch.float32),
                    torch.tensor(self.t[idx], dtype=torch.float32),
                    torch.tensor(self.u[idx], dtype=torch.float32),
                    torch.tensor(self.v[idx], dtype=torch.float32),
                    torch.tensor(self.p[idx], dtype=torch.float32))
        else:
            return (torch.tensor(self.x[idx], dtype=torch.float32),
                    torch.tensor(self.y[idx], dtype=torch.float32),
                    torch.tensor(self.t[idx], dtype=torch.float32),
                    torch.tensor(self.u[idx], dtype=torch.float32),
                    torch.tensor(self.v[idx], dtype=torch.float32)) 
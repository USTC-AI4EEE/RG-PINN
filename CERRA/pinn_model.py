import torch
import torch.nn as nn
import numpy as np

class PINNNet(nn.Module):
    def __init__(self, layers, lb, ub, L=60, U=8, DT=1, latitude_range=(35, 65)):
        """
        Initialize PINN network
        
        Args:
            layers: Network layer structure
            lb, ub: Input boundaries
            L: Characteristic length scale (km)
            U: Characteristic velocity scale (m/s)
            DT: Characteristic time scale (hours)
            latitude_range: Latitude range (min_lat, max_lat), used for calculating Coriolis parameter
        """
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        
        # Physical parameters
        self.L = L  # km
        self.U = U  # m/s
        self.DT = DT  # hours
        
        # Coriolis parameter calculation
        self.omega = 7.2921e-5  # Earth rotation angular velocity (rad/s)
        self.latitude_range = latitude_range
        
        # Calculate average Coriolis parameter (for non-dimensionalization)
        avg_lat = (latitude_range[0] + latitude_range[1]) / 2
        self.avg_latitude_rad = np.radians(avg_lat)
        self.f_avg = 2 * self.omega * np.sin(self.avg_latitude_rad)
        
        # Non-dimensionalization parameters (using average Coriolis parameter)
        self.Re = self.U * self.L * 1000 / 1.5e-5  # Reynolds number (L converted from km to m)
        self.Ro = self.U / (self.f_avg * self.L * 1000)  # Rossby number
        self.Ek = 1.5e-5 / (self.f_avg * (self.L * 1000)**2)  # Ekman number
        
        print(f"Physical parameters:")
        print(f"  Length scale: {self.L} km")
        print(f"  Velocity scale: {self.U} m/s")
        print(f"  Time scale: {self.DT} hours")
        print(f"  Latitude range: {latitude_range[0]}° - {latitude_range[1]}°")
        print(f"  Average latitude: {avg_lat:.1f}°")
        print(f"  Average Coriolis parameter f: {self.f_avg:.2e} 1/s")
        print(f"  Reynolds number: {self.Re:.2e}")
        print(f"  Rossby number: {self.Ro:.2e}")
        print(f"  Ekman number: {self.Ek:.2e}")
        
        layer_list = []
        for i in range(len(layers)-2):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*layer_list)
        self.init_weights()
        
        # Current stage latitude range (initialized to global range)
        self.current_latitude_range = self.latitude_range
        self.current_avg_latitude_rad = self.avg_latitude_rad
        self.current_f_avg = self.f_avg
        self.current_Ro = self.Ro
        self.current_Ek = self.Ek

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1)
        lb = self.lb.to(X.device)
        ub = self.ub.to(X.device)
        X_norm = 2.0 * (X - lb) / (ub - lb) - 1.0
        return self.net(X_norm)

    def predict_uvp(self, x, y, t):
        out = self.forward(x, y, t)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

    def loss_data(self, x, y, t, u, v, p=None):
        u_pred, v_pred, _ = self.predict_uvp(x, y, t)
        mse = nn.MSELoss()
        return mse(u_pred, u) + mse(v_pred, v)

    def loss_equation(self, x, y, t, Rey=None, epoch=None, total_epoch=None, beta=0.5, use_difficulty_weight=True):
        """
        Calculate NS equation residuals in spherical coordinates including Coriolis force, with optional difficulty weighting
        
        Args:
            x, y, t: spatial and temporal coordinates (non-dimensionalized)
            Rey: Reynolds number (if None, use pre-computed Re)
            epoch: current epoch
            total_epoch: total number of epochs
            beta: difficulty weighting hyperparameter
            use_difficulty_weight: whether to use difficulty weighting
        """
        if Rey is None:
            Rey = self.Re
            
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        u, v, _ = self.predict_uvp(x, y, t)
        
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
        # ∇·v = (1/(R*cos(φ))) * ∂u/∂λ + (1/R) * ∂v/∂φ + (v/R) * tan(φ)
        # where λ is longitude, φ is latitude, u is longitude direction velocity, v is latitude direction velocity
        f_mass = (1.0 / cos_lat_star) * u_x + v_y + (v / cos_lat_star) * sin_lat_star
        
        # Momentum equations in spherical coordinate system
        # λ-direction (longitude direction) momentum equation - u component
        # ∂u/∂t + (u/(R*cos(φ))) * ∂u/∂λ + (v/R) * ∂u/∂φ + (u*v/R) * tan(φ) 
        # = (1/Re) * ∇²u + (1/Ro) * f_star * v
        f_lambda = (u_t + 
                   (u / cos_lat_star) * u_x + 
                   v * u_y + 
                   (u * v / cos_lat_star) * sin_lat_star - 
                   (1.0 / Rey) * (u_xx + u_yy) + 
                   (1.0 / self.Ro) * f_star * v)
        
        # φ-direction (latitude direction) momentum equation - v component
        # ∂v/∂t + (u/(R*cos(φ))) * ∂v/∂λ + (v/R) * ∂v/∂φ - (u²/(R*cos(φ))) * tan(φ)
        # = (1/Re) * ∇²v - (1/Ro) * f_star * u
        f_phi = (v_t + 
                (u / cos_lat_star) * v_x + 
                v * v_y - 
                (u * u / cos_lat_star) * sin_lat_star - 
                (1.0 / Rey) * (v_xx + v_yy) - 
                (1.0 / self.Ro) * f_star * u)
        
        # Calculate residuals
        mse = nn.MSELoss()
        zeros = torch.zeros_like(f_lambda)
        
        # Basic loss
        # Mass conservation: average over entire spatiotemporal domain tends to 0
        loss_mass = torch.mean(f_mass)**2  # Use squared mean as loss
        loss_momentum = mse(f_lambda, zeros) + mse(f_phi, zeros)
        
        # Difficulty weighting (if enabled)
        if use_difficulty_weight and epoch is not None and total_epoch is not None:
            # Calculate training progress
            progress = epoch / total_epoch
            
            # Calculate difficulty weight (based on residual magnitude)
            difficulty_weight = torch.exp(beta * torch.abs(f_lambda + f_phi))
            difficulty_weight = difficulty_weight / difficulty_weight.mean()  # Normalize
            
            # Apply difficulty weight
            loss_momentum = torch.mean(difficulty_weight * (f_lambda**2 + f_phi**2))
        
        # Total loss
        total_loss = loss_mass + loss_momentum
        
        return total_loss

    def get_physical_parameters(self):
        """Get physical parameters"""
        return {
            'L': self.L,
            'U': self.U,
            'DT': self.DT,
            'Re': self.Re,
            'Ro': self.Ro,
            'Ek': self.Ek,
            'f_avg': self.f_avg,
            'latitude_range': self.latitude_range
        }

    def update_stage_latitude_range(self, stage_rect):
        """Update current stage latitude range (for region growing)"""
        # Calculate latitude range from rectangular region
        y_min, y_max = stage_rect[1], stage_rect[3]
        
        # Convert to physical latitude
        lat_min = self.latitude_range[0] + (self.latitude_range[1] - self.latitude_range[0]) * (y_min + self.L) / (2 * self.L)
        lat_max = self.latitude_range[0] + (self.latitude_range[1] - self.latitude_range[0]) * (y_max + self.L) / (2 * self.L)
        
        self.current_latitude_range = (lat_min, lat_max)
        self.current_avg_latitude_rad = np.radians((lat_min + lat_max) / 2)
        self.current_f_avg = 2 * self.omega * np.sin(self.current_avg_latitude_rad)
        self.current_Ro = self.U / (self.current_f_avg * self.L * 1000)
        self.current_Ek = 1.5e-5 / (self.current_f_avg * (self.L * 1000)**2) 
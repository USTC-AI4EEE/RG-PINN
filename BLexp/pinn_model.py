import torch
import torch.nn as nn
import numpy as np

class PINNNet(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)
        layer_list = []
        for i in range(len(layers)-2):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(nn.Tanh())
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*layer_list)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1)
        # Data loader has already normalized data to [0,1], use directly here
        # If conversion to [-1,1] is needed, uncomment the line below
        # X = 2.0 * X - 1.0
        return self.net(X)

    def predict_uvp(self, x, y, t):
        out = self.forward(x, y, t)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

    def loss_data(self, x, y, t, u, v, p=None):
        u_pred, v_pred, _ = self.predict_uvp(x, y, t)
        mse = nn.MSELoss()
        # Boundary layer data has no pressure field, only calculate velocity loss
        return mse(u_pred, u) + mse(v_pred, v)

    def loss_equation(self, x, y, t, Rey, epoch=None, total_epoch=None, beta=0.5, use_difficulty_weight=True, current_region=None, use_weak_continuity=True):
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        u, v, _ = self.predict_uvp(x, y, t)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        f_mass = u_x + v_y
        f_lambda = u_t + (u * u_x + v * u_y) - 1.0 / Rey * (u_xx + u_yy)
        f_phi = v_t + (u * v_x + v * v_y) - 1.0 / Rey * (v_xx + v_yy)
        mse = nn.MSELoss(reduction='none')
        zeros = torch.zeros_like(f_lambda)
        
        # If current region is specified, only calculate loss within that region
        if current_region is not None:
            x_min, x_max, y_min, y_max = current_region
            region_mask = ((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))
            if region_mask.sum() > 0:
                # Only use points within the region to calculate loss
                f_mass = f_mass[region_mask]
                f_lambda = f_lambda[region_mask]
                f_phi = f_phi[region_mask]
                zeros = zeros[region_mask]
            else:
                # If no points in the region, return zero loss
                return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # Continuity loss: choose between weak continuity or normal point-wise loss
        if use_weak_continuity:
            # Weak continuity loss
            weak_mass = torch.mean(f_mass)
            loss_mass = nn.MSELoss()(weak_mass, torch.tensor(0., device=weak_mass.device))
        else:
            # Normal point-wise PDE loss
            loss_mass = mse(f_mass, zeros).mean()
        
        # Traditional PINN: equal weights
        alpha_softmax = torch.tensor([1.0, 1.0, 1.0], device=x.device)
        
        # ---Difficulty weight---
        if use_difficulty_weight and epoch is not None and total_epoch is not None:
            # Calculate difficulty weights for two momentum equations separately
            difficulty_weight_lambda = torch.exp(beta * torch.abs(f_lambda))
            difficulty_weight_phi = torch.exp(beta * torch.abs(f_phi))
            
            # Normalize weights
            difficulty_weight_lambda = difficulty_weight_lambda / difficulty_weight_lambda.mean()
            difficulty_weight_phi = difficulty_weight_phi / difficulty_weight_phi.mean()
            
            # Apply difficulty weights
            loss_momentum = alpha_softmax[1] * torch.mean(difficulty_weight_lambda * f_lambda**2) + \
                           alpha_softmax[2] * torch.mean(difficulty_weight_phi * f_phi**2)
        else:
            # Standard loss when not using difficulty weights
            loss_momentum = alpha_softmax[1] * mse(f_lambda, zeros).mean() + \
                           alpha_softmax[2] * mse(f_phi, zeros).mean()
        
        # Calculate total PDE loss
        loss_pde = alpha_softmax[0] * loss_mass + loss_momentum
            
        return loss_pde 
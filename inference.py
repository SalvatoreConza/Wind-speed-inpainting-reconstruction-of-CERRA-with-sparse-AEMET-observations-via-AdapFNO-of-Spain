import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import yaml
from types import SimpleNamespace

# --- IMPORTS ---
from models.adaptfno_inpainting import AdaptFNOInpainting 

# --- CONFIGURATION ---
CONFIG_PATH = "config.yaml" # Ensure correct path
CHECKPOINT_PATH = "checkpoints/best_model.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Config
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
        
    print(f"Loading config from {CONFIG_PATH}...")
    cfg_dict = load_config(CONFIG_PATH)
    
    arch = cfg_dict['architecture']
    data_cfg = cfg_dict['dataset']

    # 2. Initialize Model
    print("Initializing Model...")
    model = AdaptFNOInpainting(
        in_channels=2,   
        out_channels=1, 
        img_size=tuple(arch['img_size']), 
        patch_size=tuple(arch['patch_size']),
        embedding_dim=arch['embedding_dim'],
        n_layers=arch['n_layers'],
        block_size=arch['block_size'],
        dropout=arch.get('dropout_rate', 0.0),
        global_downsample_factor=arch.get('global_downsample_factor', 2)
    ).to(DEVICE)

    # 3. Load Weights
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
         print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with random weights!")
    else:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()

    # ---------------------------------------------------------
    # 4. LOAD NORMALIZATION STATISTICS
    # ---------------------------------------------------------
    # We must normalize input exactly like training, and denormalize output to see m/s.
    stats_path = cfg_dict['normalization']['stats_file']
    print(f"Loading stats from {stats_path}...")
    stats = np.load(stats_path)
    train_mean = float(stats['mean'])
    train_std = float(stats['std'])

    # 5. Load Test Sample
    # Fallback to validation paths if 'inference' keys aren't in config
    test_gt_path = data_cfg.get('inf_gt', data_cfg['val_gt']) 
    test_mask_path = data_cfg.get('inf_mask', data_cfg['val_mask'])
    
    print(f"Loading test sample from: {test_gt_path}")
    ds_gt = xr.open_dataset(test_gt_path)
    ds_mask = xr.open_dataset(test_mask_path)
    
    gt_var = data_cfg.get('gt_var') or list(ds_gt.data_vars)[0]
    mask_var = data_cfg.get('mask_var') or list(ds_mask.data_vars)[0]

    # Select specific sample (e.g., Index 0)
    # raw_gt is in REAL UNITS (m/s)
    raw_gt = ds_gt[gt_var].values[0]     
    raw_mask = ds_mask[mask_var].values[0] 
    
    # Ensure mask is 0 or 1
    raw_mask = (raw_mask > 0.5).astype(np.float32)
    
    # ---------------------------------------------------------
    # 6. NORMALIZE INPUT
    # ---------------------------------------------------------
    # (Value - Mean) / Std
    norm_gt = (raw_gt - train_mean) / train_std
    
    # Create Normalized Sparse Input (Zeros stay zeros because 0 * x = 0)
    norm_sparse_input = norm_gt * raw_mask

    # 7. Prepare Tensor
    # Add Batch and Time dims: (1, 1, 1, H, W)
    t_sparse = torch.tensor(norm_sparse_input).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    t_mask = torch.tensor(raw_mask).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # Concatenate -> (Batch=1, Time=1, Channels=2, H, W)
    input_tensor = torch.cat([t_sparse, t_mask], dim=2).to(DEVICE)

    # 8. Inference
    print("Running inference...")
    with torch.no_grad():
        # Output is in NORMALIZED units
        norm_prediction = model(input_tensor) 

    # 9. Post-process & DENORMALIZE
    # Convert back to numpy
    norm_pred_np = norm_prediction.squeeze().cpu().numpy()
    
    # Denormalize: (Value * Std) + Mean
    # This gives us the Wind Speed in m/s
    real_pred_np = (norm_pred_np * train_std) + train_mean

    # 10. Visualization (Plotting in Real Units m/s)
    # We compare Real GT vs Real Prediction
    
    # For visualization consistency, let's also define the real sparse input
    real_sparse_input = raw_gt * raw_mask

    # Determine common color scale based on GT range
    vmin = raw_gt.min()
    vmax = raw_gt.max()

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    # A. Ground Truth (Real m/s)
    im0 = axs[0].imshow(raw_gt, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axs[0].set_title("Ground Truth (m/s)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # B. Input (Real m/s, masked)
    # Set 0s to NaN for cleaner visualization (white background)
    vis_input = np.where(raw_mask == 0, np.nan, real_sparse_input)
    im1 = axs[1].imshow(vis_input, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axs[1].set_title("Sparse Input (m/s)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # C. Prediction (Real m/s)
    im2 = axs[2].imshow(real_pred_np, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axs[2].set_title("Reconstruction (m/s)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # D. Error (Real m/s)
    diff = real_pred_np - raw_gt
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    limit = max(abs(diff.min()), abs(diff.max()))
    im3 = axs[3].imshow(diff, cmap='seismic', vmin=-limit, vmax=limit, origin='lower')
    axs[3].set_title(f"Difference (m/s)\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    out_file = "inference_result.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_file}")
    
    ds_gt.close()
    ds_mask.close()

if __name__ == "__main__":
    main()
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class CERRADataset(Dataset):
    def __init__(self, gt_path, mask_path, config, gt_var_name=None, mask_var_name=None):
        """
        Args:
            gt_path: Path to the Ground Truth .nc file
            mask_path: Path to the Mask .nc file
            config: The full configuration dictionary (loaded from yaml)
            gt_var_name: (Optional) Name of the variable in the GT file.
            mask_var_name: (Optional) Name of the variable in the Mask file.
        """
        super().__init__()
        
        # 1. Load NetCDF files
        print(f"Loading GT: {gt_path}")
        ds_gt = xr.open_dataset(gt_path)
        
        print(f"Loading Mask: {mask_path}")
        ds_mask = xr.open_dataset(mask_path)

        # 2. Auto-detect variable names
        if gt_var_name is None:
            gt_var_name = list(ds_gt.data_vars)[0]
            print(f"Auto-detected GT variable: '{gt_var_name}'")
            
        if mask_var_name is None:
            mask_var_name = list(ds_mask.data_vars)[0]
            print(f"Auto-detected Mask variable: '{mask_var_name}'")

        # 3. Convert to Torch Tensors
        self.ground_truth = torch.from_numpy(ds_gt[gt_var_name].values).float()
        self.masks = torch.from_numpy(ds_mask[mask_var_name].values).float()

        ds_gt.close()
        ds_mask.close()

        # 4. Validation
        if len(self.ground_truth) != len(self.masks):
            raise ValueError(f"Size Mismatch! GT: {len(self.ground_truth)}, Mask: {len(self.masks)}")

        # 5. Process Masks (Strictly 0 or 1)
        self.masks = (self.masks > 0.5).float()

        # ---------------------------------------------------------
        # 6. LOAD NORMALIZATION STATS
        # ---------------------------------------------------------
        self.norm_method = config['normalization']['method']
        stats_path = config['normalization']['stats_file']
        
        print(f"Loading normalization stats from: {stats_path}")
        stats = np.load(stats_path)
        
        # We load these as simple floats (scalars) because your stats file 
        # computed them over (time, y, x).
        # If you computed per-pixel stats, you would keep the spatial dims.
        self.mean = float(stats['mean'])
        self.std = float(stats['std'])
        
        # If using min/max, load those instead
        if self.norm_method == 'minmax':
            self.min = float(stats['min'])
            self.max = float(stats['max'])

    def normalize(self, x):
        """Applies normalization based on the method in config."""
        if self.norm_method == 'z_score':
            # (x - mean) / std
            return (x - self.mean) / (self.std + 1e-6)
        elif self.norm_method == 'minmax':
            # (x - min) / (max - min)
            return (x - self.min) / (self.max - self.min + 1e-6)
        return x

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, idx):
        # 1. Get sample
        gt = self.ground_truth[idx]      # (H, W)
        mask = self.masks[idx]           # (H, W)

        # 2. APPLY NORMALIZATION TO GROUND TRUTH
        # Important: We normalize the GT *before* applying the mask.
        gt_norm = self.normalize(gt)

        # 3. Create Sparse Input
        # We multiply by mask AFTER normalization.
        # This ensures that valid pixels are normalized values (e.g., 0.5, -1.2)
        # and missing pixels are exactly 0.0.
        sparse = gt_norm * mask

        # 4. Handle Dimensions
        # Reshape to (1, 1, H, W) for (Time, Channel, H, W)
        if gt_norm.ndim == 2:
            gt_norm = gt_norm.unsqueeze(0).unsqueeze(0)        
            mask = mask.unsqueeze(0).unsqueeze(0)     
            sparse = sparse.unsqueeze(0).unsqueeze(0)
        elif gt_norm.ndim == 3: # (C, H, W)
             gt_norm = gt_norm.unsqueeze(0)        
             mask = mask.unsqueeze(0)   
             sparse = sparse.unsqueeze(0)

        # 5. Concatenate for Model Input (Channel 0=Data, Channel 1=Mask)
        model_input = torch.cat([sparse, mask], dim=1) # (1, 2, H, W)

        return {
            "model_input": model_input,
            "ground_truth": gt_norm,  # Return Normalized GT for loss calculation
            "mask": mask
        }
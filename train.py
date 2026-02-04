import os
import argparse
import yaml
import torch
import mlflow
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- IMPORTS ---
# Ensure these match your actual file names/folder structure!
from data.cerra_dataset import CERRADataset  
from models.adaptfno_inpainting import AdaptFNOInpainting
from utils.loss import InpaintingLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading config from: {args.config}")

    # 2. Setup MLflow (Optional)
    use_mlflow = config.get('logging', {}).get('use_mlflow', False)
    if use_mlflow:
        mlflow.set_tracking_uri(config['logging']['mlflow_uri'])
        mlflow.set_experiment(config['logging']['experiment_name'])
        mlflow.start_run(run_name=config['logging'].get('run_name'))
        # Log params
        mlflow.log_params(config['training'])
        mlflow.log_params(config['architecture'])
        mlflow.log_artifact(args.config)

    # 3. Setup Directories
    save_dir = config['training']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # -----------------------------------------------------------
    # 4. INITIALIZE DATASETS (THE FIX IS HERE)
    # -----------------------------------------------------------
    print("Initializing Training Dataset...")
    # We must pass 'config' so the dataset can read normalization stats!
    train_dataset = CERRADataset(
        gt_path=config['dataset']['train_gt'], 
        mask_path=config['dataset']['train_mask'],
        config=config,   # <--- ADDED THIS ARGUMENT
        gt_var_name=config['dataset'].get('gt_var'),
        mask_var_name=config['dataset'].get('mask_var')
    )
    
    print("Initializing Validation Dataset...")
    val_dataset = CERRADataset(
        gt_path=config['dataset']['val_gt'], 
        mask_path=config['dataset']['val_mask'],
        config=config,   # <--- ADDED THIS ARGUMENT
        gt_var_name=config['dataset'].get('gt_var'),
        mask_var_name=config['dataset'].get('mask_var')
    )

    # 5. Auto-Detect Resolution
    # Get one sample to check dimensions
    sample_item = train_dataset[0] 
    # shape is (1, 2, H, W) -> we want H, W
    # dimensions: Batch(1), Channels(2), H, W
    sample_shape = sample_item["model_input"].shape 
    actual_h, actual_w = sample_shape[-2], sample_shape[-1]
    print(f"--> DETECTED DATA RESOLUTION: {actual_h}x{actual_w}")
    
    # Check compatibility
    patch_h, patch_w = config['architecture']['patch_size']
    if actual_h % patch_h != 0 or actual_w % patch_w != 0:
        print(f"WARNING: Image size ({actual_h},{actual_w}) not divisible by patch size!")

    # 6. Create Loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 7. Initialize Model
    print("Initializing AdaptFNOInpainting Model...")
    arch = config['architecture']
    model = AdaptFNOInpainting(
        in_channels=2,   # Sparse + Mask
        out_channels=1,  # Output Wind
        img_size=(actual_h, actual_w), 
        patch_size=tuple(arch['patch_size']),
        embedding_dim=arch['embedding_dim'],
        n_layers=arch['n_layers'],
        block_size=arch['block_size'],
        dropout=arch.get('dropout_rate', 0.0),
        global_downsample_factor=arch.get('global_downsample_factor', 2)
    ).to(device)

    # 8. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    criterion = InpaintingLoss()

    # 9. Training Loop
    best_val_loss = float('inf')
    epochs = config['training']['n_epochs']

    try:
        for epoch in range(epochs):
            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in loop:
                # Move to device
                # Add Batch Dim: (B, 1, 2, H, W) -> (B, 1, 2, H, W) 
                # Note: DataLoader usually adds Batch dim automatically. 
                # If your Dataset returns (1, 2, H, W), Loader makes it (B, 1, 2, H, W).
                # Model likely expects (B, T, C, H, W) where T=1.
                
                inputs = batch["model_input"].to(device) # (B, 1, 2, H, W)
                targets = batch["ground_truth"].to(device) # (B, 1, 1, H, W) (Normalized)
                masks = batch["mask"].to(device)           # (B, 1, 1, H, W)

                optimizer.zero_grad()
                outputs = model(inputs) # Returns (B, 1, 1, H, W)
                
                # Calculate Loss (on Normalized Data)
                loss = criterion(outputs, targets, masks)
                loss.backward()
                optimizer.step()

                # .item() is crucial to prevent memory leak
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / len(train_loader)

            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    inputs = batch["model_input"].to(device)
                    targets = batch["ground_truth"].to(device)
                    masks = batch["mask"].to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets, masks)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Val Loss = {avg_val_loss:.5f}")

            # Logging
            if use_mlflow:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                print(f"--> Saved Best Model to {save_path}")
                
            # Periodic Save
            if (epoch + 1) % config['training'].get('save_frequency', 10) == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if use_mlflow:
            mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/modified_config_file.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
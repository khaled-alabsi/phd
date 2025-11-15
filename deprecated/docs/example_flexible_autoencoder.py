"""
Example: Using FlexibleAutoencoder to train small and large bottleneck autoencoders
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.flexible_autoencoder import FlexibleAutoencoder

# Generate sample data (replace with your actual data)
np.random.seed(42)
X_data = np.random.randn(1500, 52)

# Split data
X_train, X_val = train_test_split(X_data, test_size=0.2, random_state=42)

print("="*80)
print("TRAINING TWO AUTOENCODERS WITH DIFFERENT CONFIGURATIONS")
print("="*80)

# ============================================================================
# Configuration 1: Small Bottleneck (Aggressive Compression)
# ============================================================================
print("\n[1] Training SMALL BOTTLENECK autoencoder...")
print("-"*80)

ae_small = FlexibleAutoencoder(
    verbose=1,
    cache_prefix="autoencoder_small_bottleneck"
)

# Generate small bottleneck range: input_dim/12 to input_dim/8
input_dim = X_train.shape[1]
bottleneck_grid_small = ae_small.generate_bottleneck_grid(
    input_dim=input_dim,
    min_ratio=12.0,  # 52/12 = 4.3 → 4
    max_ratio=8.0,   # 52/8 = 6.5 → 6
)

SMALL_GRID = {
    "latent_dim": bottleneck_grid_small["latent_dim"],  # [4, 5, 6]
    "encoder_layers": [(256, 128)],
    "activation": ["relu"],
    "learning_rate": [1e-3],
    "batch_size": [256],
    "epochs": [50],  # Reduced for demo
    "patience": [10],
}

print(f"\nSmall bottleneck sizes: {SMALL_GRID['latent_dim']}")
print(f"Total configurations to test: {len(SMALL_GRID['latent_dim'])}")

# Train
ae_small.fit(
    X_train,
    X_val,
    param_grid=SMALL_GRID,
    reset=False,  # Use cache if available
)

print(f"\n✓ Small autoencoder trained!")
print(f"  Config: {ae_small.config}")
print(f"  Metrics: {ae_small.metrics}")

# Plot elbow curve
print("\n  Plotting elbow curve...")
optimal_small, loss_small = ae_small.plot_elbow_curve(
    metric="val_loss",
    show=False,
    save_path="code_v2/models/small_bottleneck_elbow.png"
)
print(f"  Elbow-selected bottleneck: {optimal_small} (val_loss={loss_small:.6f})")

# ============================================================================
# Configuration 2: Large Bottleneck (Minimal Compression)
# ============================================================================
print("\n\n[2] Training LARGE BOTTLENECK autoencoder...")
print("-"*80)

ae_large = FlexibleAutoencoder(
    verbose=1,
    cache_prefix="autoencoder_large_bottleneck"
)

# Generate large bottleneck range: input_dim/4 to input_dim/2
bottleneck_grid_large = ae_large.generate_bottleneck_grid(
    input_dim=input_dim,
    min_ratio=4.0,   # 52/4 = 13
    max_ratio=2.0,   # 52/2 = 26
)

LARGE_GRID = {
    "latent_dim": bottleneck_grid_large["latent_dim"],  # [13, 14, ..., 26]
    "encoder_layers": [(512, 256, 128)],  # Deeper network
    "activation": ["selu"],  # Different activation
    "learning_rate": [5e-4],
    "batch_size": [128],
    "epochs": [50],
    "patience": [10],
}

print(f"\nLarge bottleneck sizes: {LARGE_GRID['latent_dim']}")
print(f"Total configurations to test: {len(LARGE_GRID['latent_dim'])}")

# Train
ae_large.fit(
    X_train,
    X_val,
    param_grid=LARGE_GRID,
    reset=False,
)

print(f"\n✓ Large autoencoder trained!")
print(f"  Config: {ae_large.config}")
print(f"  Metrics: {ae_large.metrics}")

# Plot elbow curve
print("\n  Plotting elbow curve...")
optimal_large, loss_large = ae_large.plot_elbow_curve(
    metric="val_loss",
    show=False,
    save_path="code_v2/models/large_bottleneck_elbow.png"
)
print(f"  Elbow-selected bottleneck: {optimal_large} (val_loss={loss_large:.6f})")

# ============================================================================
# Compare Results
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

comparison = f"""
{'Metric':<30} {'Small Bottleneck':<20} {'Large Bottleneck':<20}
{'-'*70}
{'Bottleneck Size':<30} {ae_small.config['latent_dim']:<20} {ae_large.config['latent_dim']:<20}
{'Encoder Layers':<30} {str(ae_small.config['encoder_layers']):<20} {str(ae_large.config['encoder_layers']):<20}
{'Activation':<30} {ae_small.config['activation']:<20} {ae_large.config['activation']:<20}
{'Train Loss':<30} {ae_small.metrics['train_loss']:<20.6f} {ae_large.metrics['train_loss']:<20.6f}
{'Val Loss':<30} {ae_small.metrics['val_loss']:<20.6f} {ae_large.metrics['val_loss']:<20.6f}
{'Elbow-selected Bottleneck':<30} {optimal_small:<20} {optimal_large:<20}
"""

print(comparison)

# ============================================================================
# Demonstrate Encoder Extraction
# ============================================================================
print("\n" + "="*80)
print("ENCODER EXTRACTION DEMO")
print("="*80)

# Generate test data
X_test = np.random.randn(10, 52)

print("\n[Small Autoencoder]")
print(f"  Input shape: {X_test.shape}")

# Get reconstructions
reconstructions_small = ae_small.predict(X_test)
print(f"  Reconstructions shape: {reconstructions_small.shape}")

# Get latent representations
latent_small = ae_small.encode(X_test)
print(f"  Latent representations shape: {latent_small.shape}")
print(f"  Latent dim: {latent_small.shape[1]} (should be {ae_small.config['latent_dim']})")

# Get reconstruction error
errors_small = ae_small.get_reconstruction_error(X_test)
print(f"  Reconstruction errors shape: {errors_small.shape}")
print(f"  Mean error: {np.mean(errors_small):.6f}")

# Get encoder model
encoder_small = ae_small.get_encoder()
print(f"  Encoder model: {encoder_small.name}")
print(f"  Encoder input shape: {encoder_small.input_shape}")
print(f"  Encoder output shape: {encoder_small.output_shape}")

print("\n[Large Autoencoder]")
latent_large = ae_large.encode(X_test)
print(f"  Latent representations shape: {latent_large.shape}")
print(f"  Latent dim: {latent_large.shape[1]} (should be {ae_large.config['latent_dim']})")

encoder_large = ae_large.get_encoder()
print(f"  Encoder model: {encoder_large.name}")
print(f"  Encoder output shape: {encoder_large.output_shape}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nElbow curves saved to:")
print(f"  - code_v2/models/small_bottleneck_elbow.png")
print(f"  - code_v2/models/large_bottleneck_elbow.png")

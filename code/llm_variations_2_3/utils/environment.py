"""
utils/environment.py - One-time environment bootstrap.

Usage:
    from utils.environment import setup_environment
    setup_environment()          # prints GPU info, sets seeds
"""
import os, sys, random, warnings
import numpy as np, torch

def setup_environment(config_or_seed=None):
    """Set reproducibility seeds, detect GPU, silence warnings."""
    if isinstance(config_or_seed, dict):
        seed = config_or_seed.get("seed", 42)
    else:
        seed = config_or_seed if config_or_seed is not None else 42

    # Hide broken torchvision components (Python 3.14 compat)
    sys.modules["torchvision"] = None

    warnings.filterwarnings("ignore")

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    # Device report
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("NOTE: Running on CPU - training will be slow.")

    return device

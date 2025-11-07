"""Generate environment hash for reproducibility tracking."""

import platform
import sys
from pathlib import Path

import numpy as np


def get_blas_info():
    """Get BLAS library information."""
    try:
        # Try to get BLAS config from numpy
        blas_info = np.show_config()
        return str(blas_info)
    except Exception:
        try:
            # Fallback: try to get from numpy config
            config = np.__config__
            return str(config)
        except Exception:
            return "BLAS info unavailable"


def get_numpy_config():
    """Get NumPy configuration."""
    try:
        return {
            "version": np.__version__,
            "config": str(np.show_config()),
        }
    except Exception:
        return {"version": np.__version__, "config": "unavailable"}


def generate_env_hash(output_path: Path = Path("audit/env_hash.txt")):
    """
    Generate environment hash file with system and library information.
    
    Args:
        output_path: Path to output file (default: audit/env_hash.txt)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 80)
    lines.append("Environment Hash")
    lines.append("=" * 80)
    lines.append("")
    
    # Python information
    lines.append("Python:")
    lines.append(f"  Version: {sys.version}")
    lines.append(f"  Executable: {sys.executable}")
    lines.append(f"  Platform: {platform.platform()}")
    lines.append("")
    
    # OS information
    lines.append("Operating System:")
    lines.append(f"  System: {platform.system()}")
    lines.append(f"  Release: {platform.release()}")
    lines.append(f"  Version: {platform.version()}")
    lines.append(f"  Architecture: {platform.machine()}")
    lines.append(f"  Processor: {platform.processor()}")
    lines.append("")
    
    # CPU information
    try:
        import psutil
        lines.append("CPU:")
        lines.append(f"  Physical cores: {psutil.cpu_count(logical=False)}")
        lines.append(f"  Logical cores: {psutil.cpu_count(logical=True)}")
        lines.append(f"  Frequency: {psutil.cpu_freq()}")
        lines.append("")
    except ImportError:
        lines.append("CPU:")
        lines.append(f"  Count: {platform.processor()}")
        lines.append("")
    
    # NumPy configuration
    lines.append("NumPy Configuration:")
    np_config = get_numpy_config()
    lines.append(f"  Version: {np_config['version']}")
    lines.append("  Config:")
    for line in np_config.get("config", "").split("\n"):
        if line.strip():
            lines.append(f"    {line}")
    lines.append("")
    
    # BLAS information
    lines.append("BLAS Information:")
    blas_info = get_blas_info()
    for line in blas_info.split("\n"):
        if line.strip():
            lines.append(f"  {line}")
    lines.append("")
    
    # Python packages (if available)
    try:
        import pkg_resources
        lines.append("Key Packages:")
        key_packages = ["numpy", "scipy", "hypothesis", "pytest"]
        for pkg_name in key_packages:
            try:
                pkg = pkg_resources.get_distribution(pkg_name)
                lines.append(f"  {pkg_name}: {pkg.version}")
            except Exception:
                pass
        lines.append("")
    except ImportError:
        pass
    
    lines.append("=" * 80)
    
    # Write to file
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Environment hash written to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate environment hash")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit/env_hash.txt"),
        help="Output file path (default: audit/env_hash.txt)",
    )
    args = parser.parse_args()
    
    generate_env_hash(args.output)


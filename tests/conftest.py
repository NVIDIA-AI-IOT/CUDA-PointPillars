"""Shared pytest fixtures and configuration for all tests."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import numpy as np
import torch


# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_point_cloud() -> np.ndarray:
    """Generate a sample point cloud for testing."""
    # Generate random 3D points with intensity
    # Format: [x, y, z, intensity]
    num_points = 1000
    points = np.random.randn(num_points, 4).astype(np.float32)
    points[:, :3] *= 50  # Scale x, y, z coordinates
    points[:, 3] = np.random.rand(num_points)  # Intensity between 0 and 1
    return points


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary."""
    return {
        "model": {
            "name": "pointpillar",
            "voxel_size": [0.16, 0.16, 4],
            "point_cloud_range": [0, -39.68, -3, 69.12, 39.68, 1],
            "max_num_points": 32,
            "max_voxels": 16000,
        },
        "dataset": {
            "name": "kitti",
            "data_path": "/path/to/data",
            "class_names": ["Car", "Pedestrian", "Cyclist"],
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 80,
            "learning_rate": 0.003,
        }
    }


@pytest.fixture
def mock_model_path(temp_dir: Path) -> Path:
    """Create a mock model file path."""
    model_path = temp_dir / "test_model.onnx"
    # Create an empty file to simulate model
    model_path.touch()
    return model_path


@pytest.fixture
def sample_bin_file(temp_dir: Path) -> Path:
    """Create a sample .bin file with point cloud data."""
    bin_path = temp_dir / "sample.bin"
    points = np.random.randn(1000, 4).astype(np.float32)
    points.tofile(bin_path)
    return bin_path


@pytest.fixture
def mock_yaml_config(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a mock YAML configuration file."""
    import yaml
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs():
    """Fixture to capture log messages during tests."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Get root logger
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    # Clean up
    logger.removeHandler(handler)


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
"""Validation tests to ensure the testing infrastructure is working correctly."""

import pytest
import numpy as np
from pathlib import Path


class TestSetupValidation:
    """Test class to validate the testing setup."""
    
    @pytest.mark.unit
    def test_imports_work(self):
        """Test that all required imports work correctly."""
        import torch
        import onnx
        import fire
        
        assert torch is not None
        assert onnx is not None
        assert fire is not None
    
    @pytest.mark.unit
    def test_fixtures_work(self, temp_dir, sample_point_cloud, sample_config):
        """Test that pytest fixtures are working correctly."""
        # Test temp_dir fixture
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        # Test sample_point_cloud fixture
        assert isinstance(sample_point_cloud, np.ndarray)
        assert sample_point_cloud.shape[1] == 4
        assert sample_point_cloud.dtype == np.float32
        
        # Test sample_config fixture
        assert isinstance(sample_config, dict)
        assert "model" in sample_config
        assert "dataset" in sample_config
    
    @pytest.mark.unit
    def test_file_creation_in_temp_dir(self, temp_dir):
        """Test that we can create files in the temporary directory."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, testing!")
        
        assert test_file.exists()
        assert test_file.read_text() == "Hello, testing!"
    
    @pytest.mark.unit
    def test_numpy_operations(self, sample_point_cloud):
        """Test basic numpy operations on point cloud data."""
        mean_points = np.mean(sample_point_cloud[:, :3], axis=0)
        assert mean_points.shape == (3,)
        
        # Test intensity values are in expected range
        intensities = sample_point_cloud[:, 3]
        assert np.all(intensities >= 0)
        assert np.all(intensities <= 1)
    
    @pytest.mark.integration
    def test_bin_file_fixture(self, sample_bin_file):
        """Test that the bin file fixture creates valid binary files."""
        assert sample_bin_file.exists()
        assert sample_bin_file.suffix == ".bin"
        
        # Read the binary file
        data = np.fromfile(sample_bin_file, dtype=np.float32)
        assert len(data) == 4000  # 1000 points * 4 dimensions
    
    @pytest.mark.unit
    def test_yaml_config_fixture(self, mock_yaml_config):
        """Test that YAML config fixture works correctly."""
        assert mock_yaml_config.exists()
        assert mock_yaml_config.suffix == ".yaml"
        
        # Test we can read the YAML file
        import yaml
        with open(mock_yaml_config, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config["model"]["name"] == "pointpillar"
        assert len(config["dataset"]["class_names"]) == 3
    
    @pytest.mark.unit
    @pytest.mark.parametrize("value,expected", [
        (5, 25),
        (10, 100),
        (-3, 9),
    ])
    def test_parametrize_works(self, value, expected):
        """Test that pytest parametrize decorator works."""
        assert value ** 2 == expected
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker can be used (run with pytest -m slow)."""
        import time
        start = time.time()
        time.sleep(0.1)
        duration = time.time() - start
        assert duration >= 0.1
    
    def test_coverage_works(self):
        """Test that coverage tracking is working."""
        def add(a, b):
            if a > 0:
                return a + b
            else:
                return b
        
        assert add(5, 3) == 8
        assert add(-1, 3) == 3
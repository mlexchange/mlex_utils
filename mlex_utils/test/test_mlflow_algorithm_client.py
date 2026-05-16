# mlex_utils/test/test_mlflow_algorithm_client.py
import json
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mlex_utils.mlflow_utils.mlflow_algorithm_client import MlflowAlgorithmClient
from mlex_utils.test.test_utils import (
    mlflow_test_algorithm_client,
    mock_mlflow_algorithm_client,
    mock_os_makedirs,
)


class TestMlflowAlgorithmClient:

    def test_init(self, mlflow_test_algorithm_client, mock_os_makedirs):
        """Test initialization of MlflowAlgorithmClient"""
        client = mlflow_test_algorithm_client
        # Verify environment variables were set
        assert os.environ["MLFLOW_TRACKING_USERNAME"] == "test-user"
        assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "test-password"

        # Verify cache directory creation was attempted
        mock_os_makedirs.assert_any_call(client.cache_dir, exist_ok=True)

        # Verify initial state
        assert client.algorithms == {}
        assert client.algorithm_names == []
        assert client.modelname_list == []

    def test_check_mlflow_ready_success(
        self, mlflow_test_algorithm_client, mock_mlflow_algorithm_client
    ):
        """Test check_mlflow_ready when MLflow is reachable"""
        client = mlflow_test_algorithm_client
        # Configure the mock to return a result
        mock_mlflow_algorithm_client.search_experiments.return_value = []

        # Call the method
        result = client.check_mlflow_ready()

        # Verify the result is True
        assert result is True
        mock_mlflow_algorithm_client.search_experiments.assert_called_once_with(
            max_results=1
        )

    def test_check_mlflow_ready_failure(
        self, mlflow_test_algorithm_client, mock_mlflow_algorithm_client
    ):
        """Test check_mlflow_ready when MLflow is not reachable"""
        client = mlflow_test_algorithm_client
        # Configure the mock to raise an exception
        mock_mlflow_algorithm_client.search_experiments.side_effect = Exception(
            "Connection error"
        )

        # Call the method
        result = client.check_mlflow_ready()

        # Verify the result is False
        assert result is False

    def test_getitem_success(self, mlflow_test_algorithm_client):
        """Test dictionary-style access to algorithms"""
        client = mlflow_test_algorithm_client
        # Add test algorithm
        test_algo = {"name": "test", "type": "classification"}
        client.algorithms = {"test_algo": test_algo}

        # Test successful access
        result = client["test_algo"]
        assert result == test_algo

    def test_getitem_failure(self, mlflow_test_algorithm_client):
        """Test dictionary-style access with missing key"""
        client = mlflow_test_algorithm_client

        # Test missing algorithm
        with pytest.raises(KeyError) as exc_info:
            _ = client["missing_algo"]
        assert "An algorithm with name 'missing_algo' does not exist" in str(
            exc_info.value
        )

    def test_load_from_mlflow_success(
        self, mlflow_test_algorithm_client, mock_mlflow_algorithm_client
    ):
        """Test successful loading of algorithms from MLflow"""
        client = mlflow_test_algorithm_client

        # Setup mock registered model
        mock_model = MagicMock()
        mock_model.name = "test_algorithm"
        mock_mlflow_algorithm_client.search_registered_models.return_value = [
            mock_model
        ]

        # Setup mock version
        mock_version = MagicMock()
        mock_version.run_id = "run-123"
        mock_version.version = "1"
        mock_mlflow_algorithm_client.get_latest_versions.return_value = [mock_version]

        # Setup mock run
        mock_run = MagicMock()
        mock_run.info.run_id = "run-123"
        mock_mlflow_algorithm_client.get_run.return_value = mock_run

        # Setup mock download and file reading
        algorithm_config = {
            "model_name": "test_algorithm",
            "type": "classification",
        }

        with (
            patch("mlflow.artifacts.download_artifacts"),
            patch(
                "builtins.open",
                mock_open(read_data=json.dumps(algorithm_config)),
            ),
        ):
            result = client.load_from_mlflow()

        # Verify result
        assert result is True
        assert "test_algorithm" in client.algorithm_names
        assert client.algorithms["test_algorithm"] == algorithm_config
        assert client.modelname_list == ["test_algorithm"]

    def test_load_from_mlflow_no_algorithms(
        self, mlflow_test_algorithm_client, mock_mlflow_algorithm_client
    ):
        """Test loading when no algorithms found"""
        client = mlflow_test_algorithm_client

        # Configure to return no models
        mock_mlflow_algorithm_client.search_registered_models.return_value = []

        result = client.load_from_mlflow()

        # Verify result
        assert result is False
        assert len(client.algorithms) == 0

    def test_register_algorithm_success(
        self, mlflow_test_algorithm_client, mock_mlflow_algorithm_client
    ):
        """Test successful algorithm registration"""
        client = mlflow_test_algorithm_client

        # Setup test algorithm config
        algorithm_config = {
            "model_name": "new_algorithm",
            "type": "classification",
            "image_name": "test_image",
            "image_tag": "latest",
            "description": "Test algorithm",
        }

        # Configure mocks
        mock_mlflow_algorithm_client.get_latest_versions.return_value = []
        mock_mlflow_algorithm_client.get_experiment_by_name.return_value = None
        mock_mlflow_algorithm_client.create_experiment.return_value = "exp-123"

        mock_model_details = MagicMock()
        mock_model_details.version = "1"

        with (
            patch("mlflow.start_run") as mock_start_run,
            patch("mlflow.set_tag"),
            patch("mlflow.log_param"),
            patch("mlflow.log_artifact"),
            patch("mlflow.log_dict"),
            patch("mlflow.register_model", return_value=mock_model_details),
        ):
            # Configure start_run context manager
            mock_run = MagicMock()
            mock_run.info.run_id = "run-456"
            mock_start_run.return_value.__enter__.return_value = mock_run
            mock_start_run.return_value.__exit__.return_value = None

            result = client.register_algorithm(algorithm_config)

        # Verify result
        assert result["status"] == "success"
        assert result["model_name"] == "new_algorithm"
        assert result["version"] == "1"

    def test_register_algorithm_already_exists(
        self, mlflow_test_algorithm_client, mock_mlflow_algorithm_client
    ):
        """Test registering algorithm that already exists"""
        client = mlflow_test_algorithm_client

        # Setup test algorithm config
        algorithm_config = {
            "model_name": "existing_algorithm",
            "type": "classification",
        }

        # Configure mock to return existing version
        mock_version = MagicMock()
        mock_version.version = "2"
        mock_mlflow_algorithm_client.get_latest_versions.return_value = [mock_version]

        result = client.register_algorithm(algorithm_config, overwrite=False)

        # Verify result
        assert result["status"] == "exists"
        assert result["model_name"] == "existing_algorithm"
        assert result["version"] == "2"

    def test_register_algorithm_no_model_name(self, mlflow_test_algorithm_client):
        """Test registering algorithm without model_name"""
        client = mlflow_test_algorithm_client

        # Algorithm config without model_name
        algorithm_config = {"type": "classification"}

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            client.register_algorithm(algorithm_config)
        assert "Algorithm configuration must include 'model_name'" in str(
            exc_info.value
        )

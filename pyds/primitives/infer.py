"""
Infer primitive module for DeepChem Server.

Contains the InferPrimitive class for inference tasks.
"""

from typing import Any, Dict, Optional, Union

from .base import Primitive

class Infer(Primitive):
    """
    Primitive for inference tasks.

    This class handles submitting inference jobs to the DeepChem Server API.
    """

    def run(
        self,
        model_address: str,
        data_address: str,
        output: str,
        dataset_column: Optional[str] = None,
        shard_size: Optional[int] = 8192,
        threshold: Optional[Union[int, float]] = None,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the inference primitive.

        Args:
            model_address: Datastore address of the trained model
            data_address: Datastore address of the dataset to run inference on
            output: Name of the output inference results
            dataset_column: Column in the dataset to perform inference on (required for raw CSV data)
            shard_size: Shard size for the inference operation (default: 8192)
            threshold: Threshold for binarizing predictions (for classification tasks)
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing the inference results address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """
        profile, project = self.validate_common_params(profile_name, project_name)

        data = {
            "profile_name": profile,
            "project_name": project,
            "model_address": model_address,
            "data_address": data_address,
            "output": output,
            "dataset_column": dataset_column,
            "shard_size": shard_size,
            "threshold": threshold,
        }

        data = {k: v for k, v in data.items() if v is not None}

        response = self._post("/primitive/infer", json=data)
        return self._validate_response(response)

"""
Train primitive module for DeepChem Server.

Contains the TrainPrimitive class for training tasks.
"""

from typing import Any, Dict, Optional

from .base import Primitive

class Train(Primitive):
    """
    Primitive for training tasks.

    This class handles submitting training jobs to the DeepChem Server API.
    """

    def run(
        self,
        dataset_address: str,
        model_type: str,
        model_name: str,
        init_kwargs: Optional[Dict[str, Any]] = None,
        train_kwargs: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the training primitive.

        Args:
            dataset_address: Datastore address of dataset to train on
            model_type: Type of model to train
            model_name: Name of the trained model
            init_kwargs: Keyword arguments to pass to model on initialization
            train_kwargs: Keyword arguments to pass to model on training
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing the trained model address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """
        profile, project = self.validate_common_params(profile_name, project_name)

        if init_kwargs is None:
            init_kwargs = {}
        if train_kwargs is None:
            train_kwargs = {}

        data = {
            "profile_name": profile,
            "project_name": project,
            "dataset_address": dataset_address,
            "model_type": model_type,
            "model_name": model_name,
            "init_kwargs": init_kwargs,
            "train_kwargs": train_kwargs,
        }

        response = self._post("/primitive/train", json=data)
        return self._validate_response(response)

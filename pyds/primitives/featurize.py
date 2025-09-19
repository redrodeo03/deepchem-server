"""
Featurize primitive module for DeepChem Server.

Contains the FeaturizePrimitive class for featurization tasks.
"""

from typing import Any, Dict, Optional

from .base import Primitive

class Featurize(Primitive):
    """
    Primitive for featurization tasks.

    This class handles submitting featurization jobs to the DeepChem Server API.
    """

    def run(
        self,
        dataset_address: str,
        featurizer: str,
        output: str,
        dataset_column: str,
        feat_kwargs: Optional[Dict[str, Any]] = None,
        label_column: Optional[str] = None,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the featurization primitive.

        Args:
            dataset_address: Datastore address of dataset to featurize
            featurizer: Featurizer to use
            output: Name of the featurized dataset
            dataset_column: Column containing the input for featurizer
            feat_kwargs: Keyword arguments to pass to featurizer on initialization
            label_column: The target column in case this dataset is going to be used for training
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing the featurized file address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """

        profile, project = self.validate_common_params(profile_name, project_name)

        if feat_kwargs is None:
            feat_kwargs = {}

        # Server expects feat_kwargs in nested format
        feat_kwargs_formatted = {"feat_kwargs": feat_kwargs}

        data = {
            "profile_name": profile,
            "project_name": project,
            "dataset_address": dataset_address,
            "featurizer": featurizer,
            "output": output,
            "dataset_column": dataset_column,
            "feat_kwargs": feat_kwargs_formatted,
        }

        if label_column is not None:
            data["label_column"] = label_column

        response = self._post("/primitive/featurize", json=data, headers={"Content-Type": "application/json"})
        return self._validate_response(response)

"""
Evaluate primitive module for DeepChem Server.

Contains the EvaluatePrimitive class for evaluation tasks.
"""

from typing import Any, Dict, List, Optional

from .base import Primitive

class Evaluate(Primitive):
    """
    Primitive for evaluation tasks.

    This class handles submitting evaluation jobs to the DeepChem Server API.
    """

    def run(
        self,
        dataset_addresses: List[str],
        model_address: str,
        metrics: List[str],
        output_key: str,
        is_metric_plots: bool = False,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the evaluation primitive.

        Args:
            dataset_addresses: List of dataset addresses to evaluate the model on
            model_address: Datastore address of the trained model
            metrics: List of metrics to evaluate the model with
            output_key: Name of the evaluation output
            is_metric_plots: Whether plot based metric is used or not
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Response containing the evaluation result address

        Raises:
            ValueError: If required settings are missing
            requests.exceptions.RequestException: If API request fails
        """
        profile, project = self.validate_common_params(profile_name, project_name)

        data = {
            "profile_name": profile,
            "project_name": project,
            "dataset_addresses": dataset_addresses,
            "model_address": model_address,
            "metrics": metrics,
            "output_key": output_key,
            "is_metric_plots": is_metric_plots,
        }

        response = self._post("/primitive/evaluate", json=data)
        return self._validate_response(response)

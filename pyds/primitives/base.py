"""
Base primitive class for all DeepChem Server primitives.

Contains the abstract Primitive class that all specific primitive implementations must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..base import BaseClient
from ..settings import Settings

class Primitive(BaseClient, ABC):
    """
    Abstract base class for all DeepChem Server primitives.

    This class provides common functionality for all primitives and defines
    the interface that each primitive must implement, including the abstract 'run' method.
    """

    def __init__(self, settings: Optional[Settings] = None, base_url: Optional[str] = None):
        """
        Initialize Primitive.

        Args:
            settings: Settings instance for configuration
            base_url: Base URL for the API (overrides settings if provided)
        """
        super().__init__(settings, base_url)

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to run the primitive.

        This method must be implemented by all concrete primitive classes.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            Response dictionary containing the result of the primitive execution

        Raises:
            NotImplementedError: If not implemented by concrete class
        """
        pass

    def validate_common_params(self,
                               profile_name: Optional[str] = None,
                               project_name: Optional[str] = None) -> tuple[str, str]:
        """
        Validate and get common parameters used by all primitives.

        Args:
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Tuple of (profile, project) names

        Raises:
            ValueError: If required settings are missing
        """
        return self._get_profile_and_project(profile_name, project_name)

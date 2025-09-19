"""
Base client class with common functionality for all DeepChem API clients.
"""

from typing import Any, Dict, Optional

import requests

from ..settings import Settings

class BaseClient:
    """
    Base client class containing common functionality for all DeepChem API clients.

    This class provides shared methods for HTTP requests, configuration validation,
    and other common operations that are used by specific client implementations.
    """

    def __init__(self, settings: Optional[Settings] = None, base_url: Optional[str] = None):
        """
        Initialize BaseClient.

        Args:
            settings: Settings instance for configuration
            base_url: Base URL for the API (overrides settings if provided)
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.base_url = base_url or settings.get_base_url()
        self.session = requests.Session()

    def _check_configuration(self) -> None:
        """
        Check if profile and project are configured.

        Raises:
            ValueError: If profile or project is not configured
        """
        if not self.settings.is_configured():
            missing = []
            if not self.settings.get_profile():
                missing.append("profile")
            if not self.settings.get_project():
                missing.append("project")
            raise ValueError(f"Missing required settings: {', '.join(missing)}. "
                             f"Please configure using settings.set_profile() and settings.set_project()")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to the API.

        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"API request failed: {e}")

    def _get(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make GET request to the API.

        Args:
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """
        return self._make_request("GET", endpoint, **kwargs)

    def _post(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make POST request to the API.

        Args:
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """
        return self._make_request("POST", endpoint, **kwargs)

    def _put(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make PUT request to the API.

        Args:
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """
        return self._make_request("PUT", endpoint, **kwargs)

    def _delete(self, endpoint: str, **kwargs) -> requests.Response:
        """
        Make DELETE request to the API.

        Args:
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response object
        """
        return self._make_request("DELETE", endpoint, **kwargs)

    def _validate_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Validate response and return JSON data.

        Args:
            response: Response object to validate

        Returns:
            JSON response as dictionary

        Raises:
            Exception: If response indicates an error
        """
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("detail", f"HTTP {response.status_code}")
            except:
                error_message = f"HTTP {response.status_code}"
            raise Exception(error_message)

        return response.json()

    def _get_profile_project(self,
                             profile_name: Optional[str] = None,
                             project_name: Optional[str] = None) -> tuple[str, str]:
        """
        Get profile and project names, either from parameters or settings.

        Args:
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Tuple of (profile, project) names

        Raises:
            ValueError: If required settings are missing
        """
        profile = profile_name or self.settings.get_profile()
        project = project_name or self.settings.get_project()

        if not profile or not project:
            self._check_configuration()

        # After _check_configuration(), we know both values are not None
        assert profile is not None and project is not None
        return profile, project

    def _get_profile_and_project(self,
                                 profile_name: Optional[str] = None,
                                 project_name: Optional[str] = None) -> tuple[str, str]:
        """
        Get profile and project names, either from parameters or settings.

        Args:
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)

        Returns:
            Tuple of (profile, project) names

        Raises:
            ValueError: If required settings are missing
        """
        profile = profile_name or self.settings.get_profile()
        project = project_name or self.settings.get_project()

        if not profile or not project:
            self._check_configuration()

        # After _check_configuration(), we know both values are not None
        assert profile is not None and project is not None
        return profile, project

    def healthcheck(self) -> Dict[str, Any]:
        """
        Check server health status.

        Returns:
            Health status response

        Raises:
            Exception: If API request fails
        """
        response = self._get("/healthcheck")
        return self._validate_response(response)

    def get_settings(self) -> Settings:
        """
        Get the current settings instance.

        Returns:
            Settings instance
        """
        return self.settings

    def get_base_url(self) -> str:
        """
        Get the current base URL.

        Returns:
            Base URL for the API
        """
        return self.base_url

    def set_base_url(self, url: str) -> None:
        """
        Set a new base URL for this client instance.

        Args:
            url: New base URL for the API
        """
        self.base_url = url.rstrip("/")

    def close(self) -> None:
        """
        Close the HTTP session.
        """
        self.session.close()

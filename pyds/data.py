"""
Data client module for interacting with DeepChem Server data endpoints.

Contains the Data class for all data management operations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from requests_toolbelt import MultipartEncoder

from .base import BaseClient
from .settings import Settings

class Data(BaseClient):
    """
    Client for interacting with DeepChem Server data endpoints.

    This class provides methods for data management operations.
    """

    def __init__(self, settings: Optional[Settings] = None, base_url: Optional[str] = None):
        """
        Initialize Data client.

        Args:
            settings: Settings instance for configuration
            base_url: Base URL for the API (overrides settings if provided)
        """
        super().__init__(settings, base_url)

    def upload_data(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        description: Optional[str] = None,
        profile_name: Optional[str] = None,
        project_name: Optional[str] = None,
        backend: str = "local",
    ) -> Dict[str, Any]:
        """
        Upload data to datastore.

        Args:
            file_path: Path to the file to upload
            filename: File name to save the uploaded file (uses original name if not provided)
            description: Description of the file
            profile_name: Profile name (uses settings if not provided)
            project_name: Project name (uses settings if not provided)
            backend: Backend to be used to run the job (Default: local)

        Returns:
            Response containing the dataset address

        Raises:
            ValueError: If required settings are missing or file doesn't exist
            requests.exceptions.RequestException: If API request fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get profile and project names (validates configuration)
        profile, project = self._get_profile_and_project(profile_name, project_name)

        if filename is None:
            filename = file_path.name

        # Prepare the multipart form data using MultipartEncoder
        fields: dict[str, Any] = {
            "profile_name": profile,
            "project_name": project,
            "filename": filename,
            "backend": backend,
            "file": (filename, open(file_path, "rb")),
        }

        if description is not None:
            fields["description"] = description

        multipart_data = MultipartEncoder(fields=fields)

        try:
            response = self._post(
                "/data/uploaddata",
                data=multipart_data,
                headers={"Content-Type": multipart_data.content_type},
            )
            fields["file"][1].close()
            return self._validate_response(response)
        except Exception as e:
            fields["file"][1].close()
            raise ValueError(f"Failed to upload data: {str(e)}") from e

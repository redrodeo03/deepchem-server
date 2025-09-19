Installation
============

DeepChem Server can be installed and run in several ways. The recommended approach is using Docker for the easiest setup and deployment.

Docker Installation (Recommended)
----------------------------------

Docker provides the simplest way to get DeepChem Server running with all dependencies pre-configured.

Prerequisites
~~~~~~~~~~~~~

* Docker installed on your system
* Docker Compose (usually included with Docker Desktop)

Steps
~~~~~

1. **Clone the Repository**

   .. code-block:: bash

      git clone <repository-url>
      cd deepchem-server

2. **Start the Server**

   .. code-block:: bash

      bash docker.sh

   This script will:
   
   * Build the Docker image with all required dependencies
   * Start the server container
   * Expose the API on port 8000

3. **Verify Installation**

   Open your browser and navigate to:
   
   * API Documentation: http://localhost:8000/docs
   * Health Check: http://localhost:8000/healthcheck

Manual Installation
-------------------

For development or custom deployments, you can install DeepChem Server manually.

Prerequisites
~~~~~~~~~~~~~

* Python 3.8 or higher
* pip package manager
* Virtual environment (recommended)

Steps
~~~~~

1. **Clone the Repository**

   .. code-block:: bash

      git clone <repository-url>
      cd deepchem-server

2. **Create Virtual Environment**

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**

   .. code-block:: bash

      pip install -r deepchem_server/requirements.txt

4. **Start the Server**

   .. code-block:: bash

      bash start-dev-server.sh

   Or manually:

   .. code-block:: bash

      cd deepchem_server
      uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Development Setup
-----------------

For developers who want to contribute or customize the server:

1. **Install in Development Mode**

   .. code-block:: bash

      pip install -e .

2. **Run Tests**

   .. code-block:: bash

      cd pyds/tests
      python test_upload_featurize.py

Server Configuration
--------------------

DeepChem Server is built with FastAPI. For detailed information about server configuration, deployment, and advanced settings, please refer to the `FastAPI documentation <https://fastapi.tiangolo.com/deployment/>`_.

For interactive API documentation and testing, visit http://localhost:8000/docs once your server is running.

Verification
------------

After installation, verify that everything is working correctly:

1. **Health Check**

   .. code-block:: bash

      curl http://localhost:8000/healthcheck

   Expected response: ``{"status": "ok"}``

2. **API Documentation**

   Visit http://localhost:8000/docs to see the interactive API documentation.

3. **Run Test Upload**

   .. code-block:: bash

      cd pyds/tests
      python test_upload_featurize.py

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Port Already in Use**
   If port 8000 is already in use, either stop the service using it or refer to the `FastAPI documentation <https://fastapi.tiangolo.com/deployment/>`_ for configuration options.

**Docker Issues**
   Make sure Docker is running and you have sufficient permissions:
   
   .. code-block:: bash

      docker --version
      docker ps

**Memory Issues**
   DeepChem operations can be memory-intensive. Ensure you have at least 4GB of available RAM.

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the server logs for error messages
2. Verify all dependencies are correctly installed
3. Ensure your system meets the minimum requirements
4. Consult the FastAPI documentation for deployment and configuration questions
5. Create an issue on the repository for DeepChem Server specific problems 
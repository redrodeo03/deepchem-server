# DeepChem Server

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://deep-forest-sciences-deepchem-server.readthedocs-hosted.com/en/latest/)

A minimal cloud infrastructure for [DeepChem](https://github.com/deepchem/deepchem)

---

## Table of Contents

1. [Key Features](#key-features)
2. [Requirements](#requirements)
3. [Installation](#installation)
   * [Docker (recommended)](#docker-recommended)
   * [From Source](#from-source)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Contributing](#contributing)
7. [License](#license)

---

## Key Features

* **FastAPI Backend** – blazing-fast web framework with automatic, interactive API docs (Swagger UI & ReDoc).
* **DeepChem Integration** – built-in support for DeepChem featurizers and models.
* **Flexible Datastore** – pluggable storage layer (local disk by default) for datasets & models.
* **Python SDK (`pyds`)** – high-level client library for programmatic access & batch workflows.
* **Containerised Deployment** – first-class Docker support for easy, reproducible setups.

## Requirements

DeepChem Server currently supports **Python 3.7 – 3.10** and the requirements are listed in [`deepchem_server/requirements.txt`](deepchem_server/requirements.txt).

---

## Installation

### Docker (recommended)

```bash
# 1. Clone the repo
$ git clone https://github.com/deepforestsci/deepchem-server.git
$ cd deepchem-server

# 2. Build & start (basic usage)
$ ./docker.sh

# Or specify a dockerfile (e.g., for GPU support)
$ ./docker.sh -f Dockerfile.gpu

# Show all options
$ ./docker.sh -h
```

**Available options:**
- `-f, --dockerfile DOCKERFILE`: Specify dockerfile to use (default: `Dockerfile`)
- `-d, --data-dir DIRECTORY`: Specify data directory (default: `./local_datastore`)  
- `-h, --help`: Show help message

**Available Dockerfiles:**
- `Dockerfile`: Standard CPU-based setup
- `Dockerfile-GPU`: GPU-accelerated setup with NVIDIA PyTorch base

The API will be live at **<http://localhost:8000>** with interactive docs at **<http://localhost:8000/docs>**.

### From Source

```bash
# 1. Clone the repo
$ git clone https://github.com/your-org/deepchem-server.git
$ cd deepchem-server

# 2. Create & activate virtualenv (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r deepchem_server/requirements.txt

# 4. Run the server (development mode)
$ bash start-dev-server.sh  # or uvicorn deepchem_server.main:app --reload
```

---

## Quick Start

1. **Upload data**
   ```bash
   curl -X POST \
     -F "file=@zinc5k.csv" \
     http://localhost:8000/data/uploaddata
   ```
2. **Featurize molecules**
   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"dataset_id": "<ID>", "featurizer": "ecfp"}' \
     http://localhost:8000/primitive/featurize
   ```
3. **Retrieve results**
   ```bash
   curl -O http://localhost:8000/data/<ID>/download
   ```

For a fully-worked example, run the integration test:

```bash
python pyds/tests/test_upload_featurize.py
```

---

## Project Structure

```text
├── deepchem_server          # FastAPI application & core modules
│   ├── core/                # Business logic (featurization, datastore, …)
│   ├── routers/             # API routes grouped by domain
│   └── main.py              # ASGI entry-point
├── pyds/                   # Python client (SDK)
├── docs/                    # Sphinx documentation (see docs/source)
├── Dockerfile & docker.sh   # Container build & helper script
└── tests/                   # Py-test suites & sample assets
```

---

## Contributing

Contributions are welcome — check out [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) and open a pull request. Please ensure new code is covered by tests and conforms to the existing style guidelines.

---

## License

DeepChem Server is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.
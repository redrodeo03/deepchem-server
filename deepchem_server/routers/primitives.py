import json
import math
from typing import Annotated, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.params import Body

from deepchem_server.core import model_mappings
from deepchem_server.core.feat import featurizer_map
from deepchem_server.utils import parse_boolean_none_values_from_kwargs, run_job


router = APIRouter(
    prefix="/primitive",
    tags=["primitive"],
)


@router.post("/featurize")
async def featurize(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    featurizer: Annotated[str, Body()],
    output: Annotated[str, Body()],
    dataset_column: Annotated[str, Body()],
    feat_kwargs: Annotated[Optional[Dict], Body()] = None,
    label_column: Annotated[Optional[str], Body()] = None,
) -> dict:
    """
    Submits a featurization job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to featurize
    featurizer: str
        featurizer to use
    output: str
        name of the featurized dataset
    dataset_column: Optional[str]
        Column containing the input for featurizer
    feat_kwargs: Optional[Dict]
        Keyword arguments to pass to featurizer on initialization
    label_column: Optional[str]
        The target column in case this dataset is going to be used for training purposes

    Raises
    ------
    HTTPException
        If the featurizer is invalid or if the featurization fails.

    Returns
    -------
    dict
        A dictionary containing the address of the featurized dataset.
    """
    if not feat_kwargs:
        feat_kwargs = {'feat_kwargs': {}}

    if featurizer not in featurizer_map.keys():
        message = f"Invalid featurizer: {featurizer}. Use one of {list(featurizer_map.keys())}."
        raise HTTPException(status_code=404, detail=message)

    if isinstance(feat_kwargs['feat_kwargs'], str):
        feat_kwargs['feat_kwargs'] = json.loads(feat_kwargs['feat_kwargs'])

    for key, value in feat_kwargs['feat_kwargs'].items():
        if isinstance(value, str):
            if value.lower() == "true":
                feat_kwargs['feat_kwargs'][key] = True
            elif value.lower() == "false":
                feat_kwargs['feat_kwargs'][key] = False
            elif value.lower() == "none":
                feat_kwargs['feat_kwargs'][key] = None

    program: Dict = {
        'program_name': 'featurize',
        'dataset_address': dataset_address,
        'featurizer': featurizer,
        'output': output,
        'dataset_column': dataset_column,
        'feat_kwargs': feat_kwargs['feat_kwargs'],
        'label_column': label_column,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Featurization failed: {str(e)}")

    return {"featurized_file_address": str(result)}


@router.post("/train")
async def train(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    model_type: Annotated[str, Body()],
    model_name: Annotated[str, Body()],
    init_kwargs: Annotated[Optional[Dict], Body()] = None,
    train_kwargs: Annotated[Optional[Dict], Body()] = None,
) -> dict:
    """
    Submits a training job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_address: str
        datastore address of dataset to train on
    model_type: str
        type of model to train
    model_name: str
        name of the trained model
    init_kwargs: Optional[Dict]
        Keyword arguments to pass to model on initialization
    train_kwargs: Optional[Dict]
        Keyword arguments to pass to model on training

    Raises
    ------
    HTTPException
        If the model type is invalid or if the training fails.

    Returns
    -------
    dict
        A dictionary containing the address of the trained model.
    """
    if not init_kwargs:
        init_kwargs = {}
    if not train_kwargs:
        train_kwargs = {}

    if model_type not in model_mappings.model_address_map.keys():
        message = f"Invalid model type: {model_type}. Use one of {list(model_mappings.model_address_map.keys())}."
        raise HTTPException(status_code=404, detail=message)

    if isinstance(init_kwargs, str):
        init_kwargs = json.loads(init_kwargs)

    if isinstance(train_kwargs, str):
        train_kwargs = json.loads(train_kwargs)

    if init_kwargs is not None:
        init_kwargs = parse_boolean_none_values_from_kwargs(init_kwargs)
    if train_kwargs is not None:
        train_kwargs = parse_boolean_none_values_from_kwargs(train_kwargs)

    program: Dict = {
        "program_name": "train",
        "dataset_address": dataset_address,
        "model_type": model_type,
        "model_name": model_name,
        "init_kwargs": init_kwargs,
        "train_kwargs": train_kwargs,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    return {"trained_model_address": str(result)}


@router.post("/evaluate")
async def evaluate(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    dataset_addresses: Annotated[List[str], Body()],
    model_address: Annotated[str, Body()],
    metrics: Annotated[List[str], Body()],
    output_key: Annotated[str, Body()],
    is_metric_plots: Annotated[bool, Body()] = False,
) -> dict:
    """
    Submits an evaluation job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    dataset_addresses: List[str]
        List of dataset addresses to evaluate the model on
    model_address: str
        datastore address of the trained model
    metrics: List[str]
        List of metrics to evaluate the model with
    output_key: str
        Name of the evaluation output
    is_metric_plots: bool
        Whether plot based metric is used or not
    """

    program: Dict = {
        "program_name": "evaluate",
        "dataset_addresses": dataset_addresses,
        "model_address": model_address,
        "metrics": metrics,
        "output_key": output_key,
        "is_metric_plots": is_metric_plots,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

    return {"evaluation_result_address": str(result)}


@router.post("/infer")
async def infer(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    model_address: Annotated[str, Body()],
    data_address: Annotated[str, Body()],
    output: Annotated[str, Body()],
    dataset_column: Annotated[Optional[str], Body()] = None,
    shard_size: Annotated[Optional[int], Body()] = 8192,
    threshold: Annotated[Optional[Union[int, float]], Body()] = None,
) -> dict:
    """
    Submits an inference job

    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    model_address: str
        datastore address of the trained model
    data_address: str
        datastore address of the dataset to run inference on
    output: str
        Name of the output inference results
    dataset_column: Optional[str]
        Column in the dataset to perform inference on (required for raw CSV data)
    shard_size: Optional[int]
        Shard size for the inference operation (default: 8192)
    threshold: Optional[Union[int, float]]
        Threshold for binarizing predictions (for classification tasks)

    Raises
    ------
    HTTPException
        If the model address or data address is invalid, or if the inference fails.

    Returns
    -------
    Dict or JSONResponse
        Success: {"inference_results_address": str}
        Error: JSONResponse with error message
    """
    # Convert string parameters for JSON parsing
    if isinstance(threshold, str):
        try:
            threshold = float(threshold)
        except ValueError:
            if threshold.lower() == "none":
                threshold = None
            else:
                raise HTTPException(status_code=400, detail={f"Invalid threshold value: {threshold}"})

    if isinstance(shard_size, str):
        try:
            shard_size = int(shard_size)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid shard_size value: {shard_size}")

    # Handle None values
    if dataset_column == "None":
        dataset_column = None

    # Build the program for inference
    program = {
        "program_name": "infer",
        "model_address": model_address,
        "data_address": data_address,
        "output": output,
        "dataset_column": dataset_column,
        "shard_size": shard_size,
        "threshold": threshold,
    }

    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return {"inference_results_address": str(result)}


@router.post("/train-valid-test-split")
async def train_valid_test_split(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    splitter_type: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> dict:
    """
    API for making train, test and validation split of data

    Parameters
    ----------
    splitter_type: str
        Type of splitter to use - `random` or `index` or `scaffold`
    dataset_address: str
        dataset to split
    frac_train: float
        fraction of train dataset
    frac_test: float
        fraction of train dataset
    frac_valid: float
        fraction of train dataset
    """

    # Build the program for Train Valid Test split
    program = {
        "program_name": "train_valid_test_split",
        "splitter_type": splitter_type,
        "dataset_address": dataset_address,
        "frac_train": frac_train,
        "frac_test": frac_test,
        "frac_valid": frac_valid,
    }

    if not math.isclose(frac_valid + frac_test + frac_train, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise HTTPException(status_code=400, detail=f"Invalid fractions: {frac_train}, {frac_test}, {frac_valid}")
    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Train valid test split failed: {str(e)}")

    return {"train_valid_test_split_results_address": result}

@router.post("/generate_pose")
async def docking_generate_pose(
    profile_name: str,
    project_name: str,
    protein_address: str,
    ligand_address: str,
    output: str,
    exhaustiveness: int = 10,
    num_modes: int = 9,
) -> dict:
    """
    Generate VINA molecular docking poses.
    
    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    protein_address: str
        DeepChem address of the protein PDB file
    ligand_address: str
        DeepChem address of the ligand file (PDB/SDF)
    output: str
        Output name for the docking results
    exhaustiveness: int
        Vina exhaustiveness parameter (default: 10)
    num_modes: int
        Number of binding modes to generate (default: 9)
        
    Returns
    -------
    dict
        Dictionary containing the address of the docking results
    """
    
    program = {
        'program_name': 'generate_pose',
        'protein_address': protein_address,
        'ligand_address': ligand_address,
        'output': output,
        'exhaustiveness': exhaustiveness,
        'num_modes': num_modes,
    }
    
    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Inference failed: {str(e)}")

    return {"inference_results_address": str(result)}

@router.post("/train-valid-test-split")
async def train_valid_test_split(
    profile_name: Annotated[str, Body()],
    project_name: Annotated[str, Body()],
    splitter_type: Annotated[str, Body()],
    dataset_address: Annotated[str, Body()],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> dict:
    """
    API for making train, test and validation split of data

    Parameters
    ----------
    splitter_type: str
        Type of splitter to use - `random` or `index` or `scaffold`
    dataset_address: str
        dataset to split
    frac_train: float
        fraction of train dataset
    frac_test: float
        fraction of train dataset
    frac_valid: float
        fraction of train dataset
    """

    # Build the program for Train Valid Test split
    program = {
        "program_name": "train_valid_test_split",
        "splitter_type": splitter_type,
        "dataset_address": dataset_address,
        "frac_train": frac_train,
        "frac_test": frac_test,
        "frac_valid": frac_valid,
    }

    if not math.isclose(frac_valid + frac_test + frac_train, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise HTTPException(status_code=400, detail=f"Invalid fractions: {frac_train}, {frac_test}, {frac_valid}")
    try:
        result = run_job(profile_name=profile_name, project_name=project_name, program=program)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Train valid test split failed: {str(e)}")

    return {"train_valid_test_split_results_address": result}


@router.post("/generate_pose")
async def docking_generate_pose(
    profile_name: str,
    project_name: str,
    protein_address: str,
    ligand_address: str,
    output: str,
    exhaustiveness: int = 10,
    num_modes: int = 9,
) -> dict:
    """
    Generate VINA molecular docking poses.
    
    Parameters
    ----------
    profile_name: str
        Name of the Profile where the job is run
    project_name: str
        Name of the Project where the job is run
    protein_address: str
        DeepChem address of the protein PDB file
    ligand_address: str
        DeepChem address of the ligand file (PDB/SDF)
    output: str
        Output name for the docking results
    exhaustiveness: int
        Vina exhaustiveness parameter (default: 10)
    num_modes: int
        Number of binding modes to generate (default: 9)
        
    Returns
    -------
    dict
        Dictionary containing the address of the docking results
    """
    
    program = {
        'program_name': 'generate_pose',
        'protein_address': protein_address,
        'ligand_address': ligand_address,
        'output': output,
        'exhaustiveness': exhaustiveness,
        'num_modes': num_modes,
    }
    
    try:
        result = run_job(profile_name=profile_name,
                         project_name=project_name,
                         program=program)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"VINA docking failed: {str(e)}")
    
    return {"docking_results_address": str(result)}

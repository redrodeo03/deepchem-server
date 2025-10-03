"""This file contains utilities to run inference against datasets with deepchem_server."""
import csv
import os
import pathlib
import tempfile
from typing import Callable, Iterator, Optional, Sequence, Union

import deepchem as dc
import numpy as np
import pandas as pd

from deepchem_server.core import config
from deepchem_server.core.address import DeepchemAddress
from deepchem_server.core.cards import DataCard
from deepchem_server.core.feat import featurizer_map
from deepchem_server.core.progress_logger import log_progress


def _infer_with_featurize(model_address: str,
                          data_address: str,
                          dataset_column: str,
                          shard_size: Optional[int] = 8192) -> Callable[[], Iterator[Sequence[np.ndarray]]]:
    """
    This function takes in csv file, and returns a callable iterator that
    featurizes it based on the featurizer used for train dataset and yields predictions

    Parameters
    ----------
    model_address: str
        deepchem_server address of model to run inference for
    data_address: str
        deepchem_server address of raw data to run inference on
    dataset_column: str
        The column in the raw dataset to featurize.
    shard_size: Optional[int]
        The shard size for the featurize and inference operation.

    Returns
    -------
    iterator: Callable[[], Iterator[Sequence[np.ndarray]]]
        iterator function that yields raw inputs and predictions
    """
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError('No datastore found')
    model = datastore.get(model_address, kind='model')
    model_card = datastore.get(model_address + '.cmc')

    # Get featurizer from model card
    train_dataset_address = model_card.train_dataset_address
    train_dataset_card = datastore.get(train_dataset_address + '.cdc', kind='data')
    feat_kwargs = train_dataset_card.feat_kwargs
    featurizer_code = train_dataset_card.featurizer
    if featurizer_code not in featurizer_map:
        raise ValueError("Featurizer not recognized.")
    featurizer = featurizer_map[featurizer_code](**feat_kwargs)

    # Read CSV in chunks, featurize it, infer from it, write in chunks
    def iterator() -> Iterator[Sequence[np.ndarray]]:
        tempdir = tempfile.TemporaryDirectory()
        dataset_path = pathlib.Path(tempdir.name, 'in.csv')
        datastore.download_object(data_address, str(dataset_path))
        for df_block in pd.read_csv(dataset_path, chunksize=shard_size):
            featurized_rows = featurizer.featurize(df_block[dataset_column])
            prediction = model.predict(dc.data.NumpyDataset(featurized_rows))
            raw_inputs = df_block[dataset_column].values
            yield raw_inputs, prediction

    return iterator


def _infer_without_featurize(model_address: str,
                             data_address: str,
                             shard_size: Optional[int] = 8192) -> Callable[[], Iterator[Sequence[np.ndarray]]]:
    """
    This function takes in csv file, and returns a callable iterator that yields predictions
    on featurized data

    Parameters
    ----------
    model_address: str
        deepchem_server address of model to run inference for
    data_address: str
        deepchem_server address of raw data to run inference on
    shard_size: Optional[int]
        The shard size for the featurize and inference operation.

    Returns
    -------
    iterator: Callable[[], Iterator[Sequence[np.ndarray]]]
        iterator function that yields raw inputs and predictions
    """
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError('No datastore found')
    dataset = datastore.get(data_address)
    model = datastore.get(model_address, kind='model')

    def iterator() -> Iterator[Sequence[np.ndarray]]:
        for X, _, _, ids in dataset.iterbatches(batch_size=shard_size, deterministic=True):
            prediction = model.predict(dc.data.NumpyDataset(X))
            yield ids, prediction

    return iterator


def infer(model_address: str,
          data_address: str,
          output: str,
          dataset_column: Optional[str] = None,
          shard_size: Optional[int] = 8192,
          threshold: Optional[Union[int, float]] = None):
    """Runs inference for the specified model against specified dataset and featurization.

    Parameters
    ----------
    model_address: str
        deepchem_server address of model to run inference for
    data_address: str
        deepchem_server address of raw data to run inference on
    output: str
        The output file to write results to.
    dataset_column: str
        The column in the raw dataset to featurize.
    shard_size: Optional[int]
        The shard size for the featurize and inference operation.
    threshold: Optional[Union[int, float]]
        Threshold for binarizing the predictions.

    Example
    -------
    >>> import os
    >>> from deepchem_server.core import config
    >>> from deepchem_server.core.feat import featurize
    >>> from deepchem_server.core.cards import DataCard
    >>> from deepchem_server.core.train import train
    >>> from deepchem_server.core.inference import infer
    >>> from deepchem_server.core.datastore import DiskDataStore
    >>> import tempfile
    >>> disk_datastore = DiskDataStore('profile', 'project', tempfile.mkdtemp())
    >>> config.set_datastore(disk_datastore)
    >>> df = pd.DataFrame([["CCC", 0], ["CCCCC", 1]], columns=["smiles", "label"])
    >>> card = DataCard(address='', file_type='csv', data_type='pandas.DataFrame')
    >>> data_address = disk_datastore.upload_data_from_memory(df, "test.csv", card)
    >>> feat_address = featurize(data_address,
    ... featurizer='ecfp',
    ... output='featurized_data',
    ... dataset_column='smiles',
    ... label_column='label')
    >>> model_address = train(model_type='linear_regression',
    ... dataset_address=feat_address,
    ... model_name='ecfp_reg')
    >>> infer_address = infer(model_address, feat_address, output='infer.csv')
    """
    if dataset_column == 'None':
        dataset_column = None

    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError('No datastore found')

    if datastore.exists(output):
        raise FileExistsError(f"Output address {output} already exists.")

    data_card = datastore.get(data_address + '.cdc')
    if data_card.featurizer is None and data_address.endswith('.csv'):
        if dataset_column is None:
            raise Exception("Requires dataset column name which contains raw inputs (example: smiles)")
        log_progress('inference', 10, 'downloading dataset')
        iterator = _infer_with_featurize(model_address=model_address,
                                         data_address=data_address,
                                         dataset_column=dataset_column,
                                         shard_size=shard_size)
    else:
        log_progress('inference', 10, 'downloading dataset')
        iterator = _infer_without_featurize(model_address=model_address,
                                            data_address=data_address,
                                            shard_size=shard_size)

    tempdir = tempfile.TemporaryDirectory()
    temp_output_path = os.path.join(tempdir.name, 'temp.csv')

    # write inference data in a csv file
    with open(temp_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        is_header = False
        for raw_inputs, prediction in iterator():
            # The squeeze operation is ignored for 0th dim to avoid edge cases where,
            # for example, the prediction of shape (1,2,1) is reshaped to (2,) instead of (1, 2)
            prediction = np.squeeze(prediction,
                                    axis=tuple(ax for ax in range(1, prediction.ndim) if prediction.shape[ax] == 1))

            if len(prediction.shape) == 1:
                pred_rows = [prediction]
            else:
                pred_rows = [prediction[:, i] for i in range(prediction.shape[-1])]

            # supports only upto binary classification
            if threshold is not None:
                binary_predictions = (pred_rows[-1] > threshold).astype(int)

            # sets header based on first prediction results
            if not is_header:
                header_columns = ['X']
                if len(pred_rows) == 1:
                    header_columns.append('y_preds')
                else:
                    header_columns.extend([f'y{i + 1}_preds' for i in range(len(pred_rows))])
                if threshold is not None:
                    header_columns.append('binarized_preds')
                    pred_rows.append(binary_predictions)
                writer.writerow(header_columns)
                is_header = True

            rows = [raw_inputs] + pred_rows
            for row in zip(*rows):
                writer.writerow(row)

    if not output.endswith('.csv'):
        output = output + '.csv'

    card = DataCard(address='', data_type='pandas.DataFrame', file_type='csv')
    return datastore.upload_data(DeepchemAddress.get_key(output), temp_output_path, card)

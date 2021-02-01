# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Volatility dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class StockFormatter(GenericDataFormatter):
  """Defines and formats data for the stock dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('Symbol', DataTypes.CATEGORICAL, InputTypes.ID),
      ('t', DataTypes.DATE, InputTypes.TIME),
      ('c', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('o', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('l', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('h', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('natr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('atr', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('rsi', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('macd', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('macdsignal', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('dema', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('tsf', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('minutes_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hour', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      #('week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('weekday', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('dayofweek', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      #('weekofyear', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('Region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df, valid_boundary=20, test_boundary=10, enable_scaling=False):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')
    #Reduce Memory footprint
    df = utils.reduce_mem_usage(df)
    		
    ind = len(df)
    train_index = int((1 - (valid_boundary/100))*ind)
    test_index = int((1 - (test_boundary/100))*ind)
    
    train = df.loc[:train_index]
    valid = df.loc[train_index+1: test_index]
    test = df.loc[test_index+1:]
		
    self.set_scalers(train, enable_scaling)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df, enable_scaling=False):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    print("Scaling Enabled:"+str(enable_scaling))
    self._real_scalers = sklearn.preprocessing.StandardScaler(with_mean=enable_scaling, with_std=enable_scaling).fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler(with_mean=enable_scaling, with_std=enable_scaling).fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def transform_test_data(self, df, enable_scaling):
    """Method used in prediction to transform Test data into Model Input data.

    This also calibrates scaling object, and transforms data for provided Dataframe.

    Args:
      df: Source data frame to split.

    Returns:
      Dataframe of transformed test data.
    """

    print('Formatting test splits.')
		#Reduce memory footprint
    df = utils.reduce_mem_usage(df)
    print(df)
    self.set_scalers(df, enable_scaling)
    ret = self.transform_inputs(df)
    print(ret)
    return ret
  

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[col])

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 120 + 5,
        'num_encoder_steps': 120,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.2,
        'hidden_layer_size': 80,
        'learning_rate': 0.001,
        'minibatch_size': 256,
        'max_gradient_norm': 0.01,
        'num_heads': 1,
        'stack_size': 1
    }
    
    #model_params = {
    #    'dropout_rate': 0.5,
    #    'hidden_layer_size': 80,
    #    'learning_rate': 0.0001,
    #    'minibatch_size': 128,
    #    'max_gradient_norm': 1.0,
    #    'num_heads': 4,
    #    'stack_size': 1
    #}
    return model_params

    

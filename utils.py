from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate, StratifiedKFold

class Trainer(object):
    """
    Class for training models to predict injuries.
    """
    def __init__(self, data, target, ignore_cols):
        self.data = data
        self.target = target
        self.ignore_cols = ignore_cols
        self.feat_cols = [c for c in data.columns if (c != target) and (c not in ignore_cols)]

        # TODO: split X and y into train/test (create X_train, X_test, y_train, y_test)
        self.X = data[self.feat_cols].values
        self.y = data[target].values

        self.models = {}
        self.scores = {}
        self._valid_scaler_types = {'standard': StandardScaler, None: None}
        self._valid_model_types = {'log_reg': LogisticRegression}

    def _check_scaler_model_types(self, model_type, scaler_type):
        assert scaler_type in self._valid_scaler_types, f"{scaler_type} must be one of {', '.join(self._valid_scaler_types)}."
        assert model_type in self._valid_model_types, f"{model_type} must be one of {', '.join(self._valid_model_types)}."

        print('Model type: ', model_type)
        print('Scaling: ', scaler_type)

    def add_model(self, scaler_type, model_type, **kwargs):
        """
        Add model of specified type and scaler w/ cross-validation.
        """
        # Check for valid scaling and model types
        self._check_scaler_model_types(model_type, scaler_type)

        # Build pipeline object
        model = self._valid_model_types[model_type](**kwargs)
        if scaler_type is None:
            pipeline = Pipeline(('model', model))
        else:
            scaler = self._valid_scaler_types[scaler_type]()
            pipeline = Pipeline([('scaler', scaler), ('model', model)])
        self.models[(model_type, scaler_type)]  = pipeline

    def cross_validate(self, scaler_type, model_type, cv, scoring='recall', **kwargs):
        """
        Cross-validate the specified model and scaler type. cv specifies the number of folds.
        """
        # Check for valid scaling and model types
        self._check_scaler_model_types(model_type, scaler_type)
        assert (model_type, scaler_type) in self.models, f"{(model_type, scaler_type)} not built; run add_model() method."

        # Cross-validate the model
        pipeline = self.models[(model_type, scaler_type)]
        # TODO: only cross-validate on the training set
        self.scores[(model_type, scaler_type)] = cross_validate(pipeline, self.X, self.y, scoring=scoring, cv=StratifiedKFold(n_splits=cv, shuffle=True))

if __name__ == "__main__":
    import os
    import glob
    import itertools
    import pickle

    import numpy as np
    import pandas as pd

    data_path = os.path.join('.', 'data')
    daily_data_path = os.path.join(data_path, 'day_approach_maskedID_timeseries.csv')
    weekly_data_path = os.path.join(data_path, 'week_approach_maskedID_timeseries.csv')

    daily_df = pd.read_csv(daily_data_path)

    model_trainer = Trainer(daily_df, 'injury', ['Athlete ID'])

    model_trainer.add_model('standard', 'log_reg')
    print(model_trainer.models[('log_reg', 'standard')])

    model_trainer.cross_validate('standard', 'log_reg', cv=5, scoring='f1')
    print(model_trainer.scores[('log_reg', 'standard')]['test_score'])
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import joblib
from Scaler_driver import DataScaler
#print(sys.path)

class surrogate_driver:
    def __init__(self, x_cols=None, y_cols=None, model=None, scaler=None, load_path=None):
        self.debug = False
        if load_path is not None:
            # Initialize from saved model file
            self._load_from_file(load_path)
        else:
            # Initialize from provided components
            if x_cols is None or y_cols is None or model is None or scaler is None:
                raise ValueError("When not loading from file, x_cols, y_cols, model, and scaler must be provided")
            self.x_cols = x_cols
            self.y_cols = y_cols
            self.model = model
            self.scaler = scaler
            self.model_versions = {'base': model}  # Store all model versions
            self.current_version = 'base'

    def _load_from_file(self, path):
        """Internal method to load model data from file"""
        try:
            # Load model and metadata
            model_data = joblib.load(path)
            
            # Extract components
            self.model = model_data['model']
            self.x_cols = model_data['x_cols']
            self.y_cols = model_data['y_cols']
            self.scaler = model_data['scaler']
            
            # Restore version information if available
            if 'model_versions' in model_data:
                self.model_versions = model_data['model_versions']
            else:
                self.model_versions = {'base': self.model}
                
            if 'current_version' in model_data:
                self.current_version = model_data['current_version']
            else:
                self.current_version = 'base'
            
            if self.debug:
                print(f"Model loaded from {path}")
                if 'saved_version' in model_data:
                    print(f"Loaded version: {model_data['saved_version']}")
                if 'timestamp' in model_data:
                    print(f"Saved on: {model_data['timestamp']}")
                
        except Exception as e:
            raise ValueError(f"Error loading model from {path}: {str(e)}")

    def transform_input(self, input_params):
        xdata = []
        for k in self.x_cols:
            xdata.append(input_params[k])
        xdata = np.array(xdata).reshape(1, -1)
        return xdata
    
    def predict(self, input_params, model_version=None):
        # Select model version
        if model_version and model_version in self.model_versions:
            model_to_use = self.model_versions[model_version]
        else:
            #print(f"Model version '{model_version}' not found. Using self.model.")
            model_to_use = self.model

        # Handle different input types
        if isinstance(input_params, dict):
            # Single row prediction from dictionary
            xdata = self.transform_input(input_params)
            X = pd.DataFrame(xdata, columns=self.x_cols)
        elif isinstance(input_params, pd.DataFrame):
            # Multiple row prediction from DataFrame
            X = input_params[self.x_cols]
        else:
            raise ValueError("Input must be either a dictionary or a DataFrame")

        # Scale X using DataScaler
        X_scaled = self.scaler.transform(X, x_only=True)
        
        # Predict
        y_pred_ev = model_to_use.predict(X_scaled) #  verbose=0
        
        # Inverse scale y
        y_pred_is = self.scaler.inverse_transform_y(y_pred_ev)

        # Return format based on input type
        if isinstance(input_params, dict):
            y_pred_is = y_pred_is.iloc[0]
            return {k: v for k, v in zip(self.y_cols, y_pred_is)}
        
        return y_pred_is

    def add_model_version(self, model, version_name):
        """Add a new model version (for online learning)"""
        self.model_versions[version_name] = model
        self.current_version = version_name
        if self.debug:
            print(f"Added model version: {version_name}")

    def set_current_version(self, version_name):
        """Set the current active model version"""
        if version_name in self.model_versions:
            self.current_version = version_name
            self.model = self.model_versions[version_name]
            if self.debug:
                print(f"Set current model version to: {version_name}")
        else:
            Warning(f"Model version '{version_name}' not found. Available versions: {list(self.model_versions.keys())}")
            #print(f"Model version '{version_name}' not found. Available versions: {list(self.model_versions.keys())}")

    def list_versions(self):
        """List all available model versions"""
        return list(self.model_versions.keys())

    def save_model(self, path='Models/', filename=None, version='current'):
        """Save model with metadata and version handling"""
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"surrogate_model_{timestamp}.joblib"
        
        # Select model version to save
        if version == 'current':
            model_to_save = self.model
            version_name = self.current_version
        elif version in self.model_versions:
            model_to_save = self.model_versions[version]
            version_name = version
        else:
            print(f"Version '{version}' not found. Saving current model.")
            model_to_save = self.model
            version_name = self.current_version
        
        # Save model and metadata together
        model_data = {
            'model': model_to_save,
            'x_cols': self.x_cols,
            'y_cols': self.y_cols,
            'scaler': self.scaler,
            'model_versions': self.model_versions,
            'current_version': version_name,
            'saved_version': version_name,
            'timestamp': datetime.now(),
            'class_type': 'surrogate_driver'
        }
        
        full_path = os.path.join(path, filename)
        joblib.dump(model_data, full_path)
        print(f"Model (version: {version_name}) saved to {full_path}")
        return full_path

    @classmethod
    def load_model(cls, path, debug=False):
        """Load model with metadata and version handling"""
        try:
            # Load model and metadata
            model_data = joblib.load(path)
            
            # Extract components
            model = model_data['model']
            x_cols = model_data['x_cols']
            y_cols = model_data['y_cols']
            scaler = model_data['scaler']
            
            # Create new instance
            surrogate = cls(x_cols, y_cols, model, scaler)
            
            # Restore version information if available
            if 'model_versions' in model_data:
                surrogate.model_versions = model_data['model_versions']
            if 'current_version' in model_data:
                surrogate.current_version = model_data['current_version']
            
            if debug:
                print(f"Model loaded from {path}")
                if 'saved_version' in model_data:
                    print(f"Loaded version: {model_data['saved_version']}")
                if 'timestamp' in model_data:
                    print(f"Saved on: {model_data['timestamp']}")
            
            return surrogate
            
        except Exception as e:
            raise ValueError(f"Error loading model from {path}: {str(e)}")
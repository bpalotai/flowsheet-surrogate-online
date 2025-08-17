import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataScaler:
    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.x_cols = None
        self.y_cols = None


    def save_scalers(self, path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scaler_data = {
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'x_cols': self.x_cols,
            'y_cols': self.y_cols
        }
        joblib.dump(scaler_data, path)

    def load_scalers(self, path):
        scaler_data = joblib.load(path)
        self.x_scaler = scaler_data['x_scaler']
        self.y_scaler = scaler_data['y_scaler']
        self.x_cols = scaler_data['x_cols']
        self.y_cols = scaler_data['y_cols']

    def fit(self, X_df, y_df=None):
        """
        Fit the scalers on new data
        
        Parameters:
            X_df: DataFrame containing input features
            y_df: DataFrame containing output features (optional)
        
        Returns:
            self: returns an instance of self
        """
        # Make sure we have column names
        if self.x_cols is None and isinstance(X_df, pd.DataFrame):
            self.x_cols = X_df.columns.tolist()
        
        # Fit the X scaler
        if isinstance(X_df, pd.DataFrame):
            self.x_scaler.fit(X_df[self.x_cols])
        else:
            self.x_scaler.fit(X_df)
        
        # Fit the Y scaler if y_df is provided
        if y_df is not None:
            if self.y_cols is None and isinstance(y_df, pd.DataFrame):
                self.y_cols = y_df.columns.tolist()
            
            if isinstance(y_df, pd.DataFrame):
                self.y_scaler.fit(y_df[self.y_cols])
            else:
                self.y_scaler.fit(y_df)
        
        return self

    def fit_transform(self, df, x_cols, y_cols, scaling=True):
        self.x_cols = x_cols
        self.y_cols = y_cols
        
        xdf = df[x_cols]
        ydf = df[y_cols]
        
        if scaling:
            x_scaled = self.x_scaler.fit_transform(xdf)
            y_scaled = self.y_scaler.fit_transform(ydf)
            
            xdf_scaled = pd.DataFrame(x_scaled, columns=x_cols)
            ydf_scaled = pd.DataFrame(y_scaled, columns=y_cols)
        else:
            xdf_scaled = xdf
            ydf_scaled = ydf
            
        scaled_df = pd.concat([xdf_scaled, ydf_scaled], axis=1)
        return scaled_df
    
    def transform(self, df, x_only=False):
        xdf = df[self.x_cols]
        x_scaled = self.x_scaler.transform(xdf)
        xdf_scaled = pd.DataFrame(x_scaled, columns=self.x_cols)
        
        if x_only:
            return xdf_scaled
            
        ydf = df[self.y_cols]
        y_scaled = self.y_scaler.transform(ydf)
        ydf_scaled = pd.DataFrame(y_scaled, columns=self.y_cols)
        
        return pd.concat([xdf_scaled, ydf_scaled], axis=1)
    
    def inverse_transform_y(self, y_scaled):
        if isinstance(y_scaled, pd.DataFrame):
            y_orig = self.y_scaler.inverse_transform(y_scaled)
            return pd.DataFrame(y_orig, columns=self.y_cols)
        return pd.DataFrame(
            self.y_scaler.inverse_transform(y_scaled), 
            columns=self.y_cols
        )
    
    def inverse_transform_x(self, x_scaled):
        if isinstance(x_scaled, pd.DataFrame):
            x_orig = self.x_scaler.inverse_transform(x_scaled)
            return pd.DataFrame(x_orig, columns=self.x_cols)
        return pd.DataFrame(
            self.x_scaler.inverse_transform(x_scaled), 
            columns=self.x_cols
        )


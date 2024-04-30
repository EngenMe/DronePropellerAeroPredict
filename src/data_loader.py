import os
import pandas as pd
import numpy as np
from scipy.integrate import trapz
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        path = './data/'

        # Geometric data
        geom_volume1_dir = os.path.join(path, 'volume1_geom.csv')
        geom_volume2_dir = os.path.join(path, 'volume2_geom.csv')
        geom_volume3_dir = os.path.join(path, 'volume3_geom.csv')

        # Performance data
        data_volume1_dir = os.path.join(path, 'volume1_exp.csv')
        data_volume2_dir = os.path.join(path, 'volume2_exp.csv')
        data_volume3_dir = os.path.join(path, 'volume3_exp.csv')

        # Read datasets
        geom_volume1_dataset = pd.read_csv(geom_volume1_dir)
        geom_volume2_dataset = pd.read_csv(geom_volume2_dir)
        geom_volume3_dataset = pd.read_csv(geom_volume3_dir)

        data_volume1_dataset = pd.read_csv(data_volume1_dir)
        data_volume2_dataset = pd.read_csv(data_volume2_dir)
        data_volume3_dataset = pd.read_csv(data_volume3_dir)

        # Merge into two DataFrames
        self.geom_dataset = pd.concat([geom_volume1_dataset, geom_volume2_dataset, geom_volume3_dataset], ignore_index=True)
        self.data = pd.concat([data_volume1_dataset, data_volume2_dataset, data_volume3_dataset], ignore_index=True)

        # Calculate solidity
        solidity = self.solidity()
        self.data = self.data.merge(solidity, how='left', on='PropName')

        # Fill null solidity elements
        self.miss_sol_filler()

        # Define input and output keys
        self.output_keys = ['CT', 'CP', 'eta']
        self.input_keys = [item for item in list(self.data.keys()) if item not in self.output_keys + ['PropName', 'BladeName', 'Family']]

    def solidity(self):
        props = self.data['PropName'].value_counts().index
        solidity = pd.DataFrame(columns=['PropName', 'Solidity'])
        line = ['PropName', 'BladeName', 'Family', 'B', 'D', 'P']
        for prop in props:
            df_line = self.data.loc[self.data['PropName'] == prop, line].drop_duplicates()
            bn = df_line.loc[:, 'BladeName'].item()
            b = df_line.loc[:, 'B'].item()
            d = df_line.loc[:, 'D'].item()
            mask = self.geom_dataset['BladeName'] == bn
            if mask.sum() == 0:
                continue
            df_geom = self.geom_dataset.loc[mask, :]
            c = df_geom['c/R'].to_numpy() * d * 0.5
            r = df_geom['r/R'].to_numpy() * d * 0.5
            I = trapz(c, r)
            sol = 4 * b * I /(d ** 2 * np.pi)
            prop_sol = pd.DataFrame({'PropName': [prop], 'Solidity': [sol]})
            solidity = pd.concat([solidity, prop_sol], ignore_index=True)
        return solidity

    def data_description(self):
        print('Data Information \n\n', self.data.describe())

    def miss_sol_filler(self):
        known_data = self.data[self.data['Solidity'].notnull()]
        unknown_data = self.data[self.data['Solidity'].isnull()]
        columns_to_drop = ['PropName', 'BladeName', 'Family', 'Solidity']
        X = known_data.drop(columns=columns_to_drop, axis=1)
        y = known_data['Solidity']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = DecisionTreeRegressor()
        model.fit(X_scaled, y)
        unknown_data_scaled = scaler.transform(unknown_data.drop(columns=columns_to_drop, axis=1))
        predicted_values = model.predict(unknown_data_scaled)
        self.data.loc[self.data['Solidity'].isnull(), 'Solidity'] = predicted_values

    def X_y(self, target_key):
        X = self.data[self.input_keys].values
        y = self.data[target_key].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

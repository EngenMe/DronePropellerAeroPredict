import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_loader import DataLoader
import joblib

class Results:
    def __init__(self):
        self.data = DataLoader().data
        self.input_keys = DataLoader().input_keys
        self.model()

    def model(self):
        self.ct_model = joblib.load('./models/CT_model.joblib')
        self.cp_model = joblib.load('./models/CP_model.joblib')


    def prop_choser(self):
        self.family = rd.choice(list(self.data['Family']))
        self.filtered_data = self.data[self.data['Family'] == self.family]
        self.blade_name = rd.choice(list(self.filtered_data['BladeName']))
        self.filtered_data = self.filtered_data[self.filtered_data['BladeName'] == self.blade_name]
        self.prop_name = rd.choice(list(self.filtered_data['PropName']))
        self.filtered_data = self.filtered_data[self.filtered_data['PropName'] == self.prop_name]
        self.rpm = rd.choice(list(self.filtered_data['N']))
        self.filtered_data = self.filtered_data[self.filtered_data['N'] == self.rpm]

    def X_y(self):
        self.prop_choser()
        while all(i == 0 for i in self.filtered_data['J']):
            self.prop_choser()
        self.adv = self.filtered_data['J']
        self.ct_true = self.filtered_data['CT']
        self.cp_true = self.filtered_data['CP']
        self.pe_true = self.filtered_data['eta']
        self.ct_pred = self.ct_model.predict(self.filtered_data[self.input_keys].values)
        self.cp_pred = self.cp_model.predict(self.filtered_data[self.input_keys].values)
        self.pe_pred = np.divide(np.multiply(np.array(self.ct_pred), np.array(self.adv)), self.cp_pred)

    def error(self):
        ct_mae = mean_absolute_error(self.ct_true, self.ct_pred)
        cp_mae = mean_absolute_error(self.cp_true, self.cp_pred)
        pe_mae = mean_absolute_error(self.pe_true, self.pe_pred)
        ct_mse = mean_squared_error(self.ct_true, self.ct_pred)
        cp_mse = mean_squared_error(self.cp_true, self.cp_pred)
        pe_mse = mean_squared_error(self.pe_true, self.pe_pred)
        print('Mean Absolute Error of ct, cp, and pe:\n', ct_mae, cp_mae, pe_mae)
        print('Mean Squared Error of ct, cp, and pe:\n', ct_mse, cp_mse, pe_mse)

    def plotter(self):
        self.X_y()
        self.error()
        fig, ax1 = plt.subplots()
        ax1.plot(self.adv, self.ct_true, 'og', label='Actual-CT')
        ax1.plot(self.adv, self.ct_pred, '-g', label='Pred-CT')
        ax1.plot(self.adv, self.cp_true, 'or', label='Actual-CP')
        ax1.plot(self.adv, self.cp_pred, '-r', label='Pred-CP')
        ax2 = ax1.twinx()
        ax2.plot(self.adv, self.pe_true, 'ob', label=r'Actual-$\eta$')
        ax2.plot(self.adv, self.pe_pred, '-b', label=r'Pred-$\eta$')
        fig.legend(loc='lower left')
        ax1.set_xlabel('J')
        ax1.set_ylabel('CT/CP')
        ax2.set_ylabel(r'$\eta$')
        plt.title('Predicted Model Results Compared to the Actual Values \nFamily=%s, BladeName=%s, PropName=%s, and RPM=%d' % (self.family, self.blade_name, self.prop_name, self.rpm))
        plt.show()

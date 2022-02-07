# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from scipy.stats import gamma
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

#%%
class Complete_Gamma:
    def __init__(self, data):
        self.data = pd.Series(data)
        self.sorted_data = sorted(self.data)
        self.N = len(self.data)
        self.μ = self.data[self.data.iloc[:] > 0].mean() #promedio registros diferentes de 0
        self.df = pd.DataFrame(
            {
                'm': range(1, self.N + 1),
                'P': self.sorted_data,
                'LN(P)': [np.log(x) for x in self.sorted_data],
                '(X-Xm)^2': [(x-self.μ)**2 for x in self.sorted_data],
            }
        )
        self.sum_LN_P = self.df['LN(P)'].sum()
        self.LN_μ = np.log(self.μ)
        self.A = self.LN_μ - (self.sum_LN_P/self.N)

        self.alpha = (1 / 4*self.A) * (1 + np.sqrt(4*self.A / 3))
        self.beta = self.μ / self.alpha

        self.years_list = [2, 5, 10, 22, 50, 100, 200, 300, 400, 500]
        self.years_list_derivated = [1 - 1 / x for x in self.years_list]
        self.result_df = pd.DataFrame(
            {
                'T(años)': self.years_list,
                'g(x)=P(%)': self.years_list_derivated,
            }
        )
        self.result_df['P(mm)'] = self.result_df['g(x)=P(%)'].apply(
            lambda x: gamma.ppf(x, self.alpha, 0, self.beta))  # Percent point function (inverse of cdf — percentiles).


        self.params, self.cov = self.adjust_curve_from_data()

    def my_fit_function(self, x, a, b):
        """
        Función logarítmica que caracteriza las precipitaciones en función del tiempo
        Args:
            x: valores experimentales
            a: valor a ajustar
            b: valor a ajustar
        """
        return a * np.log(x) + b

    def adjust_curve_from_data(self):
        params, cov = curve_fit(f=self.my_fit_function,
                                xdata=self.result_df['T(años)'],
                                ydata=self.result_df['P(mm)'],
                                bounds=(-np.inf, np.inf))
        return params, cov

    def plot_function(self):
        pass
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
        axes.plot(self.result_df['T(años)'], self.result_df['P(mm)'], 'b.', label="Puntos experimentales")

        xnew = np.linspace(1, 500, 200)
        axes.plot(xnew, self.my_fit_function(xnew, *self.params), 'g', label="Función estimada")
        axes.set_xlabel('T(años)')
        axes.set_ylabel('P(mm)')
        axes.set_title('Estimación usando la distribución Gamma Completa')

        function = ('{0}*ln(x)+({1})'.format(round(self.params[0], 3),
                                             round(self.params[1], 3)))

        y_original = self.result_df['P(mm)']
        y_stimated = self.my_fit_function(self.result_df['T(años)'], *self.params)
        at = AnchoredText('f(x)={0}\n R^2 = {1}'
                          .format(function, round(pow(np.corrcoef(y_original, y_stimated)[0, 1], 2), 4)),
                          prop=dict(size=10, color='m'), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.3")
        axes.add_artist(at)

        axes.legend()
        fig.show()

    def get_adjust_function(self):
        return '{0}*ln(x)+{1}'.format(self.params[0], self.params[1])

original_list = pd.Series(
    [1413.30, 2256.40, 1396.50, 1945.80, 1300.00, 1146.70, 1479.00, 1160.20, 1443.30, 922.20, 1105.80,
     1897.50, 933.40, 1110.00, 1326.90, 1348.00, 1362.10, 1236.70, 1717.90, 1300.50, 1139.60, 1233.30,
     1192.50, 1171.10, 1249.70, 1967.40, 1408.10, 617.70, 1034.20, 1371.90])

complete_gamma = Complete_Gamma(original_list)
complete_gamma.plot_function()

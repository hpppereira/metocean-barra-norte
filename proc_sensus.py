
# Processamento dos dados do sensor de pressao
# Sensus colocados na ancora do NHi Sirius na
# campanha de medicao de Barra Norte/AM
# Henrique Pereira
# Francisco Sudau
# 28/07/2019
# Sensus:
# - Pressao (mbar), Temperatura (kelvin)
# - Data inicial do primeiro mergulho

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

def read_sensus(pathname):
    """
    """

    df = pd.read_csv(pathname, header=None,
                     names=['index','dev_id','file_id','year','month','day',
                            'hour','minu','sec','offset','pres','temp'])

    # intervalo de amostragem
    dt = df.offset[2] - df.offset[1]

    datei = str(pd.datetime(df.year[0], df.month[0], df.day[0],
                            df.hour[0], df.minu[0], df.sec[0]))

    dr = pd.date_range(start=datei, periods=len(df), freq='%iS' %dt)

    df.index = dr
    df.index.name = 'date'

    return df

def concat_sensus(s1, s2):
    """
    """

    s = pd.concat((s1, s2))

    return s

def plot_sensus(s):
    """
    """

    plt.figure()
    plt.plot(s.pres, '-', label='s1')
    plt.show()

    return

if __name__ == '__main__':

    path_sensus1 = 'data/ldsc/sensus/ancora1.csv'
    path_sensus2 = 'data/ldsc/sensus/ancora2.csv'

    s1 = read_sensus(path_sensus1)
    s2 = read_sensus(path_sensus2)

    s = concat_sensus(s1, s2)

    plot_sensus(s)

    s.to_csv('data/ldsc/sensus/sensus_mare.csv', sep=',', na_rep=np.nan,
                float_format='%.2f', index=True, index_label='date')
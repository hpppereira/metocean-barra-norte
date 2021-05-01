# Concatena os dados do ecobatimetro do sirius

import numpy as np
import pandas as pd
from glob import glob
from datetime import timedelta

def read_concat_mare_eco_sirius(pathname):
    """
    Input: Hora UTC
    Output: Hora local
    """

    list_xyz = np.sort(glob(pathname + '*.xyz'))

    dateparse = lambda x: pd.datetime.strptime(x, '%d%m%Y %H%M%S.%f')

    sirius = pd.DataFrame()

    for a in list_xyz:

        print (a)

        sirius1 = pd.read_table(a, header=None, sep=' ',
                               parse_dates=[[3,4]], date_parser=dateparse,
                               index_col=['3_4'])

        # renomeia as colunas
        sirius1.rename(columns={0: "lat", 1: "lon", 2: "mare"}, inplace=True)

        # remove coluna 5
        # sirius1.drop(columns=5, inplace=True)

        # coloca nome do index de date
        sirius1.index.name = 'date'

        # converte index para datetime
        sirius1.index = pd.DatetimeIndex(sirius1.index)

        # reamostra serie em minuto
        sirius1 = sirius1.resample('T').mean()

        sirius = pd.concat((sirius, sirius1))

    # converte UTC para local
    sirius.index = sirius.index - timedelta(hours=3)

    # retira o primeiro dado com erro
    sirius = sirius.iloc[2:,:]

    # remove a media
    sirius = sirius - sirius.mean()

    # frequencia 1 e 2 do eco
    # sirius_f1 = sirius.iloc[0::2,:]
    # sirius_f2 = sirius.iloc[1::2,:]

    return sirius

if __name__ == '__main__':

    # pathname dos arquivos
    path_eco_sirius = 'data/sirius/mare/eco_xyz/'

    mare = read_concat_mare_eco_sirius(path_eco_sirius)

    mare.to_csv('data/sirius/mare/mare_eco_sirius.csv', sep=',', na_rep=np.nan,
                      index=True, index_label='date')
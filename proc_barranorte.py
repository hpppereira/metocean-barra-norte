"""
Processamento dos dados de mare de Barra Norte
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab
from datetime import datetime, timedelta
from glob import glob
plt.close('all')

def read_adcp(pathname):
    """
    Leitura dos dados exportados pelo WinRiver
    Input: pathname
    Output: data frame com data no indice
    """

    df = pd.read_csv(pathname, parse_dates=True, index_col='date')

    # dateparse = lambda x: pd.datetime.strptime(x, '%y %m %d %H %M %S')

    # df = pd.read_table(pathname, sep=',', header=None, date_parser=dateparse, parse_dates=[[0,1,2,3,4,5]],
    #                    index_col='0_1_2_3_4_5')

    # df.rename(columns={7: "dir", 8: "spd"}, inplace=True)

    # df.index.name = 'date'

    # # retira coluna que nao sabemos o que é
    # # df.drop(columns=6, inplace=True)

    # # retira spikes
    # df.loc[(df.spd > 3) | (df.spd < 0)] = np.nan

    # # reamostra a cada minuto
    # df = df.resample('10T').mean()

    return df

def read_mare_eco_sirius(pathname):
    """
    Arquivo de dados concatenados do ecobatimetro
    do Sirius
    Output: Hora local no nivel zero
    Obs: 
    dt = 1 min (media movel de 1min feita pelo Borba)
    """

    dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')

    df = pd.read_csv(pathname, parse_dates=['date'], date_parser=dateparse, index_col=['date'])

    # df.index = df.index - timedelta(hours=3)

    df['mare'] = df.mare - df.mare.mean() 

    return df

def read_mare_eco_antares(pathname):
    """
    Input: Hora local
    Output: Hora local no nivel zero
    Obs: 
    dt = 1 min (media movel de 1min feita pelo Borba)
    """

    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')

    df = pd.read_csv(pathname, parse_dates=['date'], date_parser=dateparse, index_col=['date'])

    df['mare'] = df.mare - df.mare.mean() 

    return df

def read_mare_visual(pathname):
    """
    Observacao visual da mare
    Escala de -3 a 3 (negativo para mare vazante e positivo
                      para mare enchente)
    Input: Hora UTC
    Output: Hora local
    """
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')

    df = pd.read_csv(pathname, parse_dates=['date'], date_parser=dateparse, index_col=['date'])

    # divide escala para -1.5 a 1.5
    df = df / 2

    return df

def read_mare_prev_ldsc(pathname):
    """
    Input: Hora local
    Output: Hora local
    Obs: dt = 10 min
    """

    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')

    df = pd.read_csv(pathname, parse_dates=['date'], date_parser=dateparse, index_col=['date'])

    df['mare'] = df.mare - df.mare.mean() 

    return df

def read_mare_prev_sirius(pathname):
    """
    Mare prevista para o ponto B.
    Mare corrigida a partir da mare da tabua da ponta
    do ceu pelo Cmd. Borba
    """

    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M')

    df = pd.read_csv(pathname, parse_dates=['date'], date_parser=dateparse, index_col=['date'])

    df['mare'] = df.mare - df.mare.mean() 

    return df

def calc_espec(x, nfft, fs):

    s, f = mlab.psd(x, NFFT=int(nfft), Fs=fs, detrend=mlab.detrend_mean,
                  window=mlab.window_hanning, noverlap=nfft/2)

    return s, f

def id2uv(ndr_icomp, ndr_dcomp, str_conv='cart'):
    """
    intensidade e direcao para u e v
    """

    if str_conv == 'cart':
        # Componentes vetoriais na convenção CARTESIANA.
        ucomp = ndr_icomp * np.cos(np.deg2rad(ndr_dcomp))
        vcomp = ndr_icomp * np.sin(np.deg2rad(ndr_dcomp))
    elif str_conv == 'meteo':
        # Componentes vetoriais na convenção METEOROLÓGICA.
        ucomp = ndr_icomp * np.sin(np.deg2rad(ndr_dcomp + 180.))
        vcomp = ndr_icomp * np.cos(np.deg2rad(ndr_dcomp + 180.))
    elif str_conv == 'ocean':
        # Componentes vetoriais na convenção OCEANOGRÁFICA.
        ucomp = ndr_icomp * np.sin(np.deg2rad(ndr_dcomp))
        vcomp = ndr_icomp * np.cos(np.deg2rad(ndr_dcomp))

    return ucomp, vcomp

def plot_mare_serie(antares, ldsc, eco_sirius,
                    idatenow, adcp, u, v):
    """
    """
    plt.figure()
    plt.subplot(211)
    # plt.plot(antares.mare, label='Antares')
    plt.plot(ldsc.mare, 'k', label='Mare Prev.')
    plt.legend(loc=2)
    plt.ylabel('Mare (m)')
    plt.grid()    
    # plt.plot(eco_sirius.mare, label='Eco_Sirius')
    # plt.plot(prev_sirius.mare, label='Prev_Sirius')
    # plt.plot(visual.index, visual.mare, 'r*', label='Visual')
    plt.twinx()
    plt.plot(adcp.index, adcp.spd/0.51, 'b', label='Vel. Corr.')
    plt.legend(loc=1)
    # plt.plot(ldsc.index[idatenow], ldsc.mare[idatenow], 'ko', label='Agora')
    # plt.xlim('2019-07-21 00:00:00','2019-07-23 00:00:00')
    plt.ylabel('Vel. Corrente (nós)')
    # plt.legend(loc=2)
    plt.subplot(212)
    plt.plot(adcp.index, adcp.spd/0.51, 'b-', label='Vel. Corr.')
    plt.legend(loc=2)
    plt.ylabel('Vel. Corrente (nós)')
    plt.grid()
    plt.twinx()
    plt.plot(adcp.index, adcp.dir, 'r.' , label='Dir. Corr')
    plt.ylabel('Dir. Corrente (º)')
    plt.legend(loc=1)
    # plt.xlim('2019-07-21 00:00:00','2019-07-23 00:00:00')


    # plt.twinx()
    # plt.plot(adcp.index, adcp.spd, 'r', label='ADCP')
    # plt.quiver(adcp.index, np.zeros(len(adcp)), u, v, scale=10,
    #            units='xy',
    #            headwidth=0,
    #            pivot='tail',
    #            width=0.15,
    #            linewidths=(0.001,),
    #            edgecolors='k',
    #            color='k',
    #            alpha=1
    #            )
    # plt.xlim(adcp.index[0], adcp.index[-1])
    # plt.legend(loc=1)
    # plt.ylabel('Vel. (m/s)')
    plt.show()

    return

def plot_mare_time(ldsc):
    """
    """
    plt.figure()
    plt.plot(ldsc.mare, 'b')
    # plt.plot(ldsc['2019-07-16 12:40:00':'2019-07-16 13:30:00'], 'r', linewidth=3)
    plt.plot(ldsc['2019-07-18 18:30:00':'2019-07-18 20:00:00'], 'r', linewidth=3)
    # plt.plot(ldsc['2019-07-18 09:50:00':'2019-07-18 10:00:00'], 'r', linewidth=3)
    # plt.plot(ldsc['2019-07-17 22:40:00':'2019-07-18 15:15:00'], 'r', linewidth=3)
    # plt.plot(ldsc['2019-07-18 00:00:00':'2019-07-18 07:00:00'], 'r', linewidth=3)
    # plt.plot(ldsc['2019-07-18 07:00:00':'2019-07-18 12:00:00'], 'r', linewidth=3)
    plt.ylabel('Mare (m)')
    plt.grid()    
    plt.show()

    return


def plot_adcp(adcp):
    """
    """

    plt.figure()
    plt.plot(adcp.index, adcp.spd, 'b-', label='Spd')
    plt.grid()
    plt.twinx()
    plt.plot(adcp.index, adcp.dir, 'r.' , label='Dir')
    plt.show()

    return


def plot_mare_espec(f_antares, s_antares,
                    f_ldsc, s_ldsc):

    plt.figure()
    plt.loglog(f_antares, s_antares, label='Antares')
    plt.loglog(f_ldsc, s_ldsc, label='LDSC')
    plt.xlabel('CPD')
    plt.ylabel('Energia')
    plt.xlim(0,10**2)
    plt.legend()
    plt.grid()    
    plt.show()

    return

if __name__ == '__main__':

    # pathname dos arquivos
    path_antares = 'antares/mare/mare_bn_antares.csv'
    path_ldsc = 'ldsc/mare/mare_bn_ldsc.csv'
    path_eco_sirius = 'sirius/mare/mare_sirius_eco.csv'
    path_prev_sirius = 'sirius/mare/mare_ponto_b.csv'
    path_visual = 'ldsc/mare/mare_bn_visual.csv'
    path_adcp = 'ldsc/adcp/rapazes/adcp.csv'

    # leitura dos dados
    antares = read_mare_eco_antares(path_antares)
    ldsc = read_mare_prev_ldsc(path_ldsc)
    eco_sirius = read_mare_eco_sirius(path_eco_sirius)
    # prev_sirius = read_mare_prev_sirius(path_prev_sirius)
    visual = read_mare_visual(path_visual)
    adcp = read_adcp(path_adcp)

    # calculo do espectro
    s_antares, f_antares = calc_espec(x=antares.mare, nfft=len(antares)/2, fs=1.0/(1.0/60/24))
    s_ldsc, f_ldsc = calc_espec(x=ldsc.mare, nfft=len(ldsc)/2, fs=1.0/(10.0/60/24))

    # data atual no formato da previsao
    datenow = str(pd.datetime.now())[:-11] + '0:00'

    # index da data atual no array da previsao
    idatenow = np.where(ldsc.index == datenow)[0][0]

    # converte intensidade e direcao para u e v
    u, v = id2uv(adcp.spd, adcp.dir, 'ocean')

    # plotagem da mare
    plot_mare_serie(antares, ldsc, eco_sirius,
                    idatenow, adcp, u, v)

    # plot_mare_time(ldsc)

    # plot_adcp(adcp)

    # plot_mare_espec(f_antares, s_antares,
    #                 f_ldsc, s_ldsc)

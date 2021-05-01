# Processamento dos dados meteorologicos do Sirius


import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

def read_meteo_sirius(pathname):
    """
    """

    df = pd.read_csv(pathname, parse_dates=True, index_col='date')

    return df

def plot_params(df):
    """
    """
    fig = plt.figure(figsize=(10,15),facecolor='w')

    ax1 = fig.add_subplot(221)
    ax1.plot(df.ate, 'b', label='Temp. Ar')
    ax1.plot(df.ote, 'r', label='Temp. Orv')
    ax1.set_ylabel('Temp. Ar / Orvalho ºC')
    ax1.legend(fontsize=7)
    ax1.grid()
    plt.xticks(rotation=20, visible=False)

    ax2 = fig.add_subplot(222, sharex=ax1)
    ax2.plot(df.rh)
    ax2.set_ylabel('Umid. Rel. (%)')
    ax2.grid()
    plt.xticks(rotation=20, visible=False)

    ax3 = fig.add_subplot(223, sharex=ax1)
    ax3.plot(df.bp_st, 'b', label='Estação')
    ax3.plot(df.bp_nm, 'r', label='Nivel Med.')    
    ax3.set_ylabel('Pressão Atm. (hPa)')
    ax3.legend(fontsize=7)
    ax3.grid()
    plt.xticks(rotation=20, visible=True)

    ax4 = fig.add_subplot(224, sharex=ax1)
    ax4.plot(df.ws)
    ax4.set_ylabel('Int. Vento (nós)')
    ax4.grid()
    plt.xticks(rotation=20, visible=True)
    ax4.legend(loc=1)
    ax44 = ax4.twinx()
    ax44.plot(df.wd, 'r', label='Dir. Vento')
    ax44.set_ylabel('Dir. Vento (º)')
    ax44.legend(loc=2)

    plt.show()

    return

def plot_wind(df):

    fig = plt.figure(figsize=(6,6),facecolor='w')

    ax1 = fig.add_subplot(211)
    ax1.plot(df.ws)
    ax1.set_ylabel('Int. Vento (nós)')
    ax1.grid()
    ax1.set_ylim(0,35)
    plt.xticks(rotation=15, visible=True)

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(df.wd, 'b', label='Dir. Vento')
    ax2.set_ylabel('Dir. Vento (º)')
    ax2.grid()
    plt.xticks(rotation=15, visible=True)

    plt.show()

    return

if __name__ == '__main__':

    path_meteo = 'data/sirius/meteo/meteo_sirius.csv'

    df = read_meteo_sirius(path_meteo)

    plot_params(df)
    plot_wind(df)

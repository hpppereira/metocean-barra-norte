# Processamento dos dados do ADCP
# da campanha de Barra Norte/AM
# concatenados pela rotina concat_qc_adcp.py
# Henrique Pereira
# 28/09/2019

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from glob import glob
register_matplotlib_converters()
import warnings
warnings.filterwarnings('ignore')
plt.close('all')

def read_mare(pathname, filename):
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%y %H:%M')

    mare = pd.read_csv(pathname + filename, sep=',', header=0, names=None,
                     date_parser=dateparse,
                     parse_dates=['Fuso_P'],
                     index_col='Fuso_P')

    # correcao da unidade da mare na serie ALT
    # mare['ALT'].loc[mare.ALT > 100] = mare['ALT'].loc[mare.ALT > 100] / 1000

    return mare

def winriver_header():

    head = ['ano','mes','dia','hora','min','seg','dec_seg','temperatura','depth',
            'pitch','roll','heading','lat','lon','water_speed','water_dir',
            'mag_1','mag_2','mag_3','mag_4','mag_5','mag_6','mag_7','mag_8','mag_9',
            'mag_10','mag_11','mag_12','mag_13','mag_14','mag_15','mag_16','mag_17',
            'mag_18','mag_19','mag_20','mag_21','mag_22','mag_23','mag_24','mag_25',
            'mag_26','mag_27','mag_28','mag_29','mag_30','mag_31','mag_32','mag_33',
            'mag_34','mag_35','mag_36','mag_37','mag_38','mag_39','mag_40','mag_41',
            'mag_42','mag_43','mag_44','mag_45','mag_46','mag_47','mag_48','mag_49',
            'mag_50','mag_51','mag_52','mag_53','mag_54','mag_55','mag_56','mag_57',
            'mag_58','mag_59','mag_60','mag_61','mag_62','mag_63','mag_64','mag_65',
            'mag_66','mag_67','mag_68','mag_69','mag_70','mag_71','mag_72','mag_73',
            'mag_74','mag_75','dir_1','dir_2','dir_3','dir_4','dir_5','dir_6','dir_7',
            'dir_8','dir_9','dir_10','dir_11','dir_12','dir_13','dir_14','dir_15',
            'dir_16','dir_17','dir_18','dir_19','dir_20','dir_21','dir_22','dir_23',
            'dir_24','dir_25','dir_26','dir_27','dir_28','dir_29','dir_30','dir_31',
            'dir_32','dir_33','dir_34','dir_35','dir_36','dir_37','dir_38','dir_39',
            'dir_40','dir_41','dir_42','dir_43','dir_44','dir_45','dir_46','dir_47',
            'dir_48','dir_49','dir_50','dir_51','dir_52','dir_53','dir_54','dir_55',
            'dir_56','dir_57','dir_58','dir_59','dir_60','dir_61','dir_62','dir_63',
            'dir_64','dir_65','dir_66','dir_67','dir_68','dir_69','dir_70','dir_71',
            'dir_72','dir_73','dir_74','dir_75']

    return head

def concat_ascii(path_ascii, path_out, head):
    """
    Media movel de 1 min
    """

    list_adcp = np.sort(glob(path_ascii + '**/*.TXT', recursive=True))

    dateparse = lambda x: pd.datetime.strptime(x, '%y %m %d %H %M %S %f')

    df = pd.DataFrame()

    for a in list_adcp:

        print (a)

        df1 = pd.read_csv(a, sep=',', header=None, names=head, date_parser=dateparse, parse_dates=[['ano','mes','dia','hora','min','seg','dec_seg']],
                           index_col='ano_mes_dia_hora_min_seg_dec_seg')

        # coloca nome do index de date
        df1.index.name = 'date'

        df1 = quality_control(df1, head)

        df = pd.concat((df, df1))

    df.sort_index(inplace=True)

    # reamostra serie em minuto
    df = df.resample('T').mean()

    return df

def quality_control(df, head):

    # magnitudes
    for v in df.columns[9:84]:
        df[v].loc[(df[v] > 5) | (df[v] < 0)] = np.nan

    # direcoes
    for d in df.columns[84:]:
        df[d].loc[(df[d] > 360) | (df[d] < 0)] = np.nan

    df['temperatura'].loc[(df['temperatura'] > 360) | (df['temperatura'] < 0)] = np.nan

    df['depth'].loc[(df['depth'] > 20) | (df['depth'] < 7)] = np.nan

    df['pitch'].loc[(df['pitch'] > 5) | (df['pitch'] < -5)] = np.nan

    df['roll'].loc[(df['roll'] > 5) | (df['roll'] < -5)] = np.nan

    return df

def read_output(path_out, filename):

    # leitura dos dados
    df = pd.read_csv(path_out + filename, sep=',', header=0,
                     parse_dates=['date'], index_col='date')

    return df

def plot_mare(df, mare):
    """
    Plot comparacao da mare do ADCP e do Eco
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # plt.plot(mare.ALT, label='ALT')
    # plt.plot(mare.Media_Movel, label='Eco_Media_Movel')
    ax1.plot(mare.Mare1, label='Eco')
    ax1.plot(df.depth, label='ADCP')
    ax1.grid()
    ax1.set_title('Maré - Barra Norte')
    ax1.legend(ncol=5)
    plt.xticks(rotation=15)
    ax1.set_ylabel('Maré (m)')

    return fig

def plot_contour_adcp(df, mare, q, vert='ECO'):
    """
    Plot ADCP contour
    q = True - plota o quiver
    """

    depth = np.arange(21.62+0.25, 3.12, -0.25)# - 3.12 - .125
    depth1 = np.arange(21.62+0.25, 3.12, -0.25) - 3.12 - .125
    depth1 = depth1 + 1
    mag  = df[list(df.columns[9:84])].values.T# / 1000
    dire  = df[list(df.columns[84:])].values.T# / 1000
    u = mag * np.sin(np.deg2rad(dire))
    v = mag * np.cos(np.deg2rad(dire))

    # faz correcao vertical do perfil
    mag1 = np.zeros((len(depth), mag.shape[1])) * np.nan
    dire1 = np.zeros((len(depth), dire.shape[1])) * np.nan
    u1 = np.zeros((len(depth), u.shape[1])) * np.nan
    v1 = np.zeros((len(depth), v.shape[1])) * np.nan

    # correcao do eixo vertical (prof. ADCP)
    if vert == 'ADCP':
        for i in range(len(df.depth)):
            if np.isnan(df.depth[i]) == False:
                # acha o indice da celula correspondente a profundidade
                # medida pelo ADCP
                idx = (np.abs(depth - df.depth[i])).argmin()
                mag1[idx:,i] = mag[:-idx,i]
                dire1[idx:,i] = dire[:-idx,i]
                u1[idx:,i] = u[:-idx,i]
                v1[idx:,i] = v[:-idx,i]

    # correcao do eixo vertical (prof. Eco)
    if vert == 'ECO':
        for i in range(len(mare)):
            if np.isnan(mare.Mare1[i]) == False:
                # acha o indice da celula correspondente a profundidade
                # medida pelo ECO
                idx = (np.abs(depth - mare.Mare1[i])).argmin()
                mag1[idx:,i] = mag[:-idx,i]
                dire1[idx:,i] = dire[:-idx,i]
                u1[idx:,i] = u[:-idx,i]
                v1[idx:,i] = v[:-idx,i]

    gs = gridspec.GridSpec(2, 1)
    fig=plt.figure(figsize=(10,12),facecolor='w')

    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('Intensidade')
    lvls = np.arange(0, 3.25, .25)
    CF = ax1.contourf(df.index, depth1, mag1, levels=lvls, cmap=plt.cm.jet)
    ax1.plot(mare.Mare1, 'k', linewidth=2)
    ax1.plot(mare.index, np.ones(len(mare))*0, 'k', linewidth=0.7)
    cbar = plt.colorbar(CF, ticks=np.arange(0,3.5,0.25), format='%.2f', label=r'Intensidade da Corrente (m/s)',
                        orientation='horizontal', fraction=0.1, pad=0.11)
    ax1.set_ylabel('Profundidade (m)')
    ax1.set_ylim(-1, 15)
    ax1.set_yticklabels([])
    ax1.set_yticklabels(np.arange(-1,16).astype(str), minor=True)
    ax1.set_yticks(np.arange(-1,16), minor=True)
    ax1.set_xticks(df.resample('.5H').mean().index, minor=True)
    plt.xticks(rotation=10)
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_title('Direção')
    cmap= mpl.colors.ListedColormap(['#ff5700', '#e90100', '#800000', '#000080', '#0000e9', '#003aff', '#0097ff', '#ffad00'])
    cmap.set_under("crimson")
    cmap.set_over("w")
    lvls = np.arange(0, 360+45, 45)
    CF = ax2.contourf(df.index, depth1, dire1, levels=lvls, cmap=cmap, norm=None)
    ax2.plot(mare.Mare1, 'k', linewidth=2)
    ax2.plot(mare.index, np.ones(len(mare))*0, 'k', linewidth=0.7)
    if q:
        qwind = ax2.quiver(df.index[::60], depth1[::4], u1[::4, ::60], v1[::4,::60],
                       scale=50, pivot='tail', width=0.0017)#, headwidth=2)
    cbar = plt.colorbar(CF, ticks=np.arange(0,360+45,45), format='%i', label=r'Direção da Corrente (º)',
                        orientation='horizontal', fraction=0.1, pad=0.11)
    ax2.set_ylabel('Profundidade (m)')
    ax2.set_ylim(-1, 15)
    ax2.set_yticklabels([])
    ax2.set_yticklabels(np.arange(-1,16).astype(str), minor=True)
    ax2.set_yticks(np.arange(-1,16), minor=True)
    ax2.set_xticks(df.resample('.5H').mean().index, minor=True)
    plt.xticks(rotation=10)
    ax2.grid(which='minor', alpha=0.2)
    ax2.grid(which='major', alpha=0.5)

    return fig, depth, mag, dire, u, v


#if __name__ == '__main__':

do_concat_ascii = False
do_save_txt = False
do_read_output = True

# pathname dos arquivos
path_ascii = os.environ['HOME'] + '/gdrive/coppe/ldsc/barranorte_20190715/data/ldsc/adcp/winriver/ascii/ascii_v2/'
path_mare = os.environ['HOME'] + '/gdrive/coppe/ldsc/barranorte_20190715/data/ldsc/mare/'
path_out = os.environ['HOME'] + '/gdrive/coppe/ldsc/barranorte_20190715/out/'
path_fig = os.environ['HOME'] + '/gdrive/coppe/ldsc/barranorte_20190715/fig/'

mare = read_mare(path_mare, 'dados_mare_14a27jul2019_borba.csv')

head = winriver_header()

# concat TXT files
if do_concat_ascii == True:
    df = concat_ascii(path_ascii, path_out, head)

# save output file
if  do_save_txt == True:
    df.to_csv(path_out + 'output_adcp_barranorte_campanha_completa.csv', sep=',', na_rep=np.nan, index=True, index_label='date')

# read output file
if do_read_output == True:
    df = read_output(path_out, 'output_adcp_barranorte_campanha_completa.csv')

# deixa mare do eco com mesma data que o adcp
mare = mare[df.index[0]:df.index[-1]]

fig = plot_mare(df, mare)
fig.savefig('{}mare_eco_adcp.png'.format(path_fig))

# plotagem para todos os dias
print (50*'-')
print ('Campanha completa')
fig, depth, mag, dire, u, v = plot_contour_adcp(df, mare, q=False)
fig.savefig('{}adcp_campanha_completa.png'.format(path_fig))
plt.show()

# loop para os dias 15 a 27
for d in np.arange(15,27).astype(str):
    print (50*'-')
    print ('Dia {}'.format(d))
    dff = df['2019-07-{}'.format(d)]
    maref = mare['2019-07-{}'.format(d)]
    fig, depth, mag, dire, u, v = plot_contour_adcp(dff, maref, q=True)
    fig.savefig('{}adcp_campanha_dia_{}.png'.format(path_fig, d))
    plt.show()
    # stop

    # imprimir cores do jet com 12 tonalidades
    # cc = []
    # cmap = mpl.cm.get_cmap('jet', 12)    # PiYG
    # for i in range(cmap.N):
    #     rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    #     cc.append(mpl.colors.rgb2hex(rgb))


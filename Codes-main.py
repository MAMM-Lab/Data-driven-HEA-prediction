import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import read_excel
import math
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rcParams['font.sans-serif'] = 'simsun'

periodic_table = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


def Element_info():


    Element_data = {'He': [0.8, 2, 1, 3.89, 2],          'H':  [14.2, 1, 0.78, 2.2, 1],     'Ne': [24.7, 8, 1, 3.67, 10],        'O': [50.5, 6, 0.66, 3.44, 8],    'F': [53.6, 7, 0.64, 3.98, 9],
                     'N': [63.3, 5, 0.8, 3.04, 7],       'Ar': [84.0, 8, 1, 3.3, 18],       'Kr': [116, 18, 1.14, 3, 36],        'Xe': [161, 18, 1.33, 2.67, 54],  'Cl': [172, 7, 1.62, 3.16, 17],
                     'Rn': [202.1, 18, 1.53, 2.2, 86],   'Hg': [234, 12, 1.62, 2, 80],      'Br': [266, 17, 1.11, 2.96, 35],     'Fr': [300, 1, 1.53, 0.7, 87],    'Cs': [301.7, 1, 2.71, 0.79, 55],
                     'Ga': [302.91, 13, 1.4, 1.81, 31],  'Rb': [312.79, 1, 2.5, 0.82, 37],  'P': [317.3, 5, 1.3, 2.19, 15],      'K': [336.5, 1, 2.34, 0.82, 19],  'Na': [371, 1, 1.91, 0.93, 11],
                     'I': [386.7, 17, 1.95, 2.66, 53],   'S': [388.51, 6, 1.04, 2.58, 16],  'In': [429.91, 13, 1.66, 1.78, 49],  'Li': [453.9, 1, 1.54, 0.98, 3],  'Se': [494, 16, 1.6, 2.55, 34],
                     'Sn': [505.21, 14, 1.58, 1.96, 50], 'Po': [527, 16, 1.7, 2, 84],       'Bi': [544.67, 15, 1.7, 2.02, 83],   'At': [575, 17, 1.53, 2.2, 85],   'Tl': [577, 13, 1.73, 1.62, 81],
                     'Cd': [594.33, 12, 1.57, 1.69, 48], 'Pb': [600.8, 14, 1.75, 2.33, 82], 'Zn': [692.88, 12, 1.39, 1.65, 30],  'Te': [722.8, 16, 1.7, 2.1, 52],  'Sb': [904.1, 15, 1.6, 2.05, 51],
                     'Np': [913, 3, 1.58, 1.36, 93],     'Pu': [913, 3, 1.64, 1.28, 94],    'Mg': [923, 2, 1.6, 1.31, 12],       'Al': [933.4, 3, 1.43, 1.61, 13], 'Ra': [973, 2, 1.53, 0.9, 88],
                     'Ba': [1002, 2, 2.24, 0.89, 56],    'Sr': [1042, 2, 2.15, 0.95, 38],   'Ce': [1071, 3, 1.82, 1.12, 58],     'As': [1090, 15, 1.5, 2.18, 33],  'Eu': [1095, 3, 2.04, 1.2, 63],
                     'Yb': [1097, 3, 1.94, 1.1, 70],     'Ca': [1112, 2, 1.97, 1, 20],      'Es': [1130, 3, 1.42, 1.3, 99],      'La': [1190, 3, 1.87, 1.1, 57],   'Pr': [1204, 3, 1.83, 1.13, 59],
                     'Pm': [1204, 3, 1.63, 1.13, 61],    'Ge': [1211.4, 14, 1.4, 2.01, 32], 'Ag': [1234, 11, 1.44, 1.93, 47],    'Bk': [1259, 3, 1.42, 1.3, 97],   'Am': [1267, 3, 1.73, 1.13, 95],
                     'Nd': [1289, 3, 1.82, 1.14, 60],    'Ac': [1320, 3, 1.88, 1.1, 89],    'Au': [1337.73, 11, 1.44, 2.54, 79], 'Cm': [1340, 3, 1.42, 1.28, 96],  'Sm': [1345, 3, 1.8, 1.17, 62],
                     'Cu': [1357.8, 11, 1.28, 1.9, 29],  'U': [1405, 3, 1.55, 1.38, 92],    'Mn': [1519, 7, 1.32, 1.55, 25],     'Be': [1551, 2, 1.13, 1.57, 4],   'Gd': [1585, 3, 1.8, 1.2, 64],
                     'Tb': [1630, 3, 1.78, 1.1, 65],     'Dy': [1680, 3, 1.77, 1.22, 66],   'Si': [1680, 4, 1.34, 1.9, 14],      'Ni': [1726, 10, 1.24, 1.91, 28], 'Ho': [1740, 3, 1.77, 1.23, 67],
                     'Co': [1768, 9, 1.26, 1.88, 27],    'Md': [1794, 3, 1.42, 1.3, 101],   'No': [1794, 3, 1.42, 1.3, 102],     'Er': [1795, 3, 1.76, 1.24, 68],  'Y': [1799, 3, 1.8, 1.22, 39],
                     'Fe': [1808, 8, 1.27, 1.83, 26],    'Sc': [1812, 3, 1.64, 1.36, 21],   'Tm': [1818, 3, 1.75, 1.25, 69],     'Pd': [1825, 10, 1.37, 2.2, 46],  'Pa': [1870, 3, 1.6, 1.5, 91],
                     'Cf': [1925, 3, 1.42, 1.3, 98],     'Ti': [1930, 4, 1.45, 1.54, 22],   'Lu': [1936, 3, 1.73, 1.27, 71],     'Th': [2028, 3, 1.8, 1.3, 90],    'Pt': [2045, 10, 1.39, 2.28, 78],
                     'Zr': [2125, 4, 1.6, 1.33, 40],     'Cr': [2130, 6, 1.27, 1.66, 24],   'V': [2175, 5, 1.35, 1.63, 23],      'Rh': [2239, 9, 1.34, 2.28, 45],  'Tc': [2470, 7, 1.35, 1.9, 43],
                     'Hf': [2500, 4, 1.59, 1.3, 72],     'Ru': [2520, 8, 1.32, 2.2, 44],    'B': [2570, 3, 0.95, 2.04, 5],       'Ir': [2716, 9, 1.36, 2.2, 77],   'Nb': [2741, 5, 1.48, 1.6, 41],
                     'Mo': [2890, 6, 1.4, 2.16, 42],     'Fm': [3054, 3, 1.42, 1.3, 100],   'Lr': [3234, 3, 1.42, 1.291, 103],   'Ta': [3269, 5, 1.48, 1.5, 73],   'Os': [3300, 8, 1.34, 2.2, 76],
                     'Re': [3450, 7, 1.46, 1.9, 75],     'W': [3680, 6, 1.41, 2.36, 74],    'C': [3948, 4, 0.86, 2.55, 6]}
    #columns = ['melting_point', 'VEC', 'atom_radius', 'electronegativity']
    df = pd.DataFrame(Element_data).T
    df = df.reset_index()
    df.columns = ['element', 'melting_point', 'VEC', 'atom_radius', 'electronegativity', 'sequence']
    df = df.sort_values(by='sequence')
    df = df.reset_index(drop=True)
    print(df)
    df.to_csv(r'D:\study\sunshine\HEA\data\Element.csv')


def HEA_data():



    df_new = read_csv(r'D:\study\sunshine\HEA\data\PREML.csv', index_col=0)
    #print(len(df_new))
    #df_new = read_csv(r'D:\study\sunshine\HEA\data\ML.csv')
    for i in range(len(df_new)):
        name = df_new.loc[i, 'name']
        phase = df_new.loc[i, 'phase']
        Alloy_info(name, phase)


    # name = input('please input HEA: ')
    # print(name)


def Alloy_info(name, phase):

    element = []
    mole = []
    # element;mole;
    df_element = read_csv(r'D:\study\sunshine\HEA\data\Element.csv')
    element_list = list(df_element['element'])

    num = ''
    last = ''
    #print(name)
    i = 0   
    while i < len(name):
        if ord(name[i]) >= 65 and ord(name[i]) <= 90:   
            ele = name[i]    # (X)
            i += 1
            if i < len(name):    
                if ord(name[i]) >= 97 and ord(name[i]) <= 122:
                    ele += name[i]     # (Xx)
                    i += 1
        else:   
            #ord(name[i]) >= 48 and ord(name[i]) <= 57:
            ele = name[i]    # (X)
            i += 1
            if i < len(name):
                if ord(name[i]) >= 48 and ord(name[i]) <= 57:
                    ele += name[i]    # (XX)
                    i += 1
                    if i < len(name):
                        if ord(name[i]) == 46:
                            ele += name[i]   # (XX.)
                            i += 1
                            ele += name[i]   # (XX.X)
                            i += 1
                            if i < len(name):
                                if ord(name[i]) >= 48 and ord(name[i]) <= 57:
                                    ele += name[i]   # (XX.XX)
                                    i += 1
                elif ord(name[i]) == 46:
                    ele += name[i]    # (X.)
                    i += 1
                    ele += name[i]    # (X.X)
                    i += 1
                    if i < len(name):
                        if ord(name[i]) >= 48 and ord(name[i]) <= 57:
                            ele += name[i]
                            i += 1      # (X.XX)
        #print(ele)
        #print(i)
        if len(last) == 0:
            element.append(ele)
        elif last in element and ele in element_list:
            element.append(ele)
            mole.append(float(0))
        elif last in element and ele not in element_list:
            mole.append(float(ele))
        else:
            element.append(ele)

        last = ele

        num += ele
        if num == name:
            break

        ele = ''
    if len(element) != len(mole):
        mole.append(float(0))

    n = 0
    m = 0
    if 0 in mole:
        for j in mole:
            if j == 0:
                n += 1
            m += j
        k = (100 - m) / n
        for p in range(len(mole)):
            if mole[p] == 0:
                mole[p] = k

    alloy = dict(zip(element, mole))
    #print(alloy)

    new_alloy = {}
    index = []
    for i in list(alloy.keys()):
        index.append(periodic_table.index(i))
    index = sorted(index)
    for j in index:
        new_alloy[periodic_table[j]] = alloy[periodic_table[j]]

    new_name = ''
    for i in list(new_alloy.keys()):
        new_name += i
        new_name += str(new_alloy[i])
    #print(new_name)

    phase_class = {'BCC': 0, 'FCC': 1, 'HCP': 2, 'IM': 3, 'AM': 4,
                   'BCC+FCC': 5, 'BCC+HCP': 6, 'FCC+HCP': 7,
                   'BCC+IM': 8, 'FCC+IM': 9, 'HCP+IM': 10,
                   'BCC+FCC+HCP': 11, 'BCC+FCC+IM': 12, 'BCC+HCP+IM': 13, 'FCC+HCP+IM': 14,
                   'SS': 15, 'IM+SS': 16}
    phase_sort = ''
    phase_num = {}
    if phase.count('+') == 0:
        phase_sort = phase
        label = phase_class[phase_sort]
    elif phase.count('+') == 1:
        phase_1, phase_2 = phase.split('+')
        phase_num[phase_1] = phase_class[phase_1]
        phase_num[phase_2] = phase_class[phase_2]
        phase_list = sorted(phase_num.items(), key=lambda x: x[1], reverse=False) 
        j = 0
        for i in phase_list:
            phase_sort += i[0]
            if j+1 < len(phase_list):
                phase_sort += '+'
            j += 1
        label = phase_class[phase_sort]
    elif phase.count('+') == 2:
        phase_1, phase_2, phase_3 = phase.split('+')
        phase_num[phase_1] = phase_class[phase_1]
        phase_num[phase_2] = phase_class[phase_2]
        phase_num[phase_3] = phase_class[phase_3]
        phase_list = sorted(phase_num.items(), key=lambda x: x[1], reverse=False)
        j = 0
        for i in phase_list:
            phase_sort += i[0]
            if j+1 < len(phase_list):
                phase_sort += '+'
            j += 1
        label = phase_class[phase_sort]


    feature_column = ['mean_VEC', 'VEC_dif', 'mean_electronegativity', 'electronegativity_dif',
                      'atom_radius_dif', 'a2', 'y', 'melting_point_dif', 'Smix', 'aomiga', 'X',
                      'Hmix', 'Hmix_dif', 'Hmix_z', 'Hmix_p', 'Hmix_n',
                      'phase', 'label']

    feature_column = [r"${VEC}$", r"$\delta_{VEC}$", r"$\chi_{arg}$", r"$\delta_\chi$",
                      r"$\delta_r$", r"$\alpha_2$", r"$\gamma$", r"$\delta_T$",
                      r"$\Delta{S_{mix}}$", r"$\Omega$", r"$\lambda$",
                      r"$\Delta{H_{mix}}$", r"$\delta{H_{mix}}$", r"$\delta{H_{mix}^0}$",
                      r"$\delta{H_{mix}^+}$", r"$\delta{H_{mix}^-}$", 'phase', 'label']


    feature_column.insert(0, 'name')
    feature_column.insert(1, 'full_name')
    feature_list = feature(new_alloy)
    feature_list.insert(0, name)
    feature_list.insert(1, new_name)
    feature_list.append(phase)
    feature_list.append(label)

    dict_feature = dict(zip(feature_column, feature_list))
    df_feature = pd.DataFrame(dict_feature, index=[0])
    #print(df_feature)
    if os.path.exists('D:\study\sunshine\HEA\data\ML.csv'):
        df_alloy_exist = read_csv(r'D:\study\sunshine\HEA\data\ML.csv')
        df_alloy_name = list(df_alloy_exist['full_name'])
        if new_name in df_alloy_name:
            print('{}该合金型号已存在'.format(name))
        else:
            df_feature.to_csv(r'D:\study\sunshine\HEA\data\ML.csv', index=False, mode='a', header=False)
    else:
        df_feature.to_csv(r'D:\study\sunshine\HEA\data\ML.csv', index=False, header=True)


def feature(new_alloy):

    df_Element_data = read_csv(r'D:\study\sunshine\HEA\data\Element.csv', sep=',')
    df_Element_data = df_Element_data.set_index('element')

    # y,a2,X
    mean_atom_radius = 0
    mean_melting_point = 0
    mean_electronegativity = 0
    mean_VEC = 0
    Smix = 0
    R = 0.008314        # KJ/(mol*k)
    kB = 1.380649e-26   # KJ/K
    for i in list(new_alloy.keys()):
        mean_atom_radius += new_alloy[i] / 100 * (df_Element_data.loc[i, 'atom_radius'])
        mean_melting_point += new_alloy[i] / 100 * (df_Element_data.loc[i, 'melting_point'])
        mean_electronegativity += new_alloy[i] / 100 * (df_Element_data.loc[i, 'electronegativity'])
        mean_VEC += new_alloy[i] / 100 * (df_Element_data.loc[i, 'VEC'])
        Smix += -1 * new_alloy[i] / 100 * (math.log(new_alloy[i] / 100, math.e)) * R

    sum_radius = 0
    sum_melting = 0
    sum_electronegativity = 0
    sum_VEC = 0
    for i in list(new_alloy.keys()):
        radius = df_Element_data.loc[i, 'atom_radius']
        melting = df_Element_data.loc[i, 'melting_point']
        electronegativity = df_Element_data.loc[i, 'electronegativity']
        VEC = df_Element_data.loc[i, 'VEC']

        sum_radius += new_alloy[i] / 100 * (math.pow((1 - (radius / mean_atom_radius)), 2))
        sum_melting += new_alloy[i] / 100 * (math.pow((1 - (melting / mean_melting_point)), 2))
        sum_electronegativity += new_alloy[i] / 100 * (math.pow((1 - (electronegativity / mean_electronegativity)), 2))
        sum_VEC += new_alloy[i] / 100 * (math.pow((1 - (VEC / mean_VEC)), 2))


    atom_radius_dif = float('%.5f' % (math.sqrt(sum_radius)))
    melting_point_dif = float('%.5f' % (math.sqrt(sum_melting)))
    electronegativity_dif = float('%.5f' % (math.sqrt(sum_electronegativity)))
    VEC_dif = float('%.5f' % (math.sqrt(sum_VEC)))
    mean_VEC = float('%.5f' % (mean_VEC))
    mean_electronegativity = float('%.5f' % (mean_electronegativity))
    Smix = float('%.5f' % (Smix))
    X = float('%.5f' % (Smix / math.pow(atom_radius_dif, 2)))
    #mean_melting_point = float('%.5f' % (mean_melting_point))


    # Hmix
    df_Hmix = read_excel(r'D:\study\sunshine\HEA\data\Hmix.xlsx', sheet_name='Sheet1')
    df_Hmix = df_Hmix.set_index('element')

    alloy_element = list(new_alloy.keys())
    Hmix = 0
    for i in range(len(alloy_element)):
        if i+1 < len(alloy_element):
            for j in range(i+1, len(alloy_element)):
                Hmix += 4 * df_Hmix.loc[alloy_element[i], alloy_element[j]] * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / 10000

    Hmix_dif = 0
    Hmix_z = 0
    Hmix_p = 0
    Hmix_n = 0
    a2 = 0
    for i in range(len(alloy_element)):
        if i+1 < len(alloy_element):
            for j in range(i+1, len(alloy_element)):
                #Hmix_dif += math.pow((1-(df_Hmix.loc[alloy_element[i], alloy_element[j]]/Hmix)), 2) * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / 10000
                Hmix_dif += math.pow((df_Hmix.loc[alloy_element[i], alloy_element[j]] - Hmix), 2) * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / 10000
                a2 += abs(df_Element_data.loc[alloy_element[i], 'atom_radius']+df_Element_data.loc[alloy_element[j], 'atom_radius']-2*mean_atom_radius) * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / (10000*2*mean_atom_radius)
                Hmix_z += math.pow(df_Hmix.loc[alloy_element[i], alloy_element[j]], 2) * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / 10000
                if df_Hmix.loc[alloy_element[i], alloy_element[j]] > 0:
                    Hmix_p += math.pow(df_Hmix.loc[alloy_element[i], alloy_element[j]], 2) * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / 10000
                else:
                    Hmix_n += math.pow(df_Hmix.loc[alloy_element[i], alloy_element[j]], 2) * new_alloy[alloy_element[i]] * new_alloy[alloy_element[j]] / 10000

    Hmix = float('%.5f' % (Hmix))
    Hmix_dif = float('%.5f' % ((math.sqrt(Hmix_dif)) / (kB * mean_melting_point)))
    a2 = float('%.5f' % (a2))
    Hmix_z = float('%.5f' % ((math.sqrt(Hmix_z)) / (kB * mean_melting_point)))
    Hmix_p = float('%.5f' % ((math.sqrt(Hmix_p)) / (kB * mean_melting_point)))
    Hmix_n = float('%.5f' % ((math.sqrt(Hmix_n)) / (kB * mean_melting_point)))

    #
    aomiga = abs(mean_melting_point * Smix / Hmix)
    aomiga = float('%.5f' % (aomiga))


    atom_radius_list = []
    for i in list(new_alloy.keys()):
        atom_radius_list.append(df_Element_data.loc[i, 'atom_radius'])
    largest_radius = max(atom_radius_list)
    smallest_radius = min(atom_radius_list)
    Ws = 1 - math.sqrt((math.pow((largest_radius+mean_atom_radius), 2)-math.pow(mean_atom_radius, 2))/math.pow((largest_radius+mean_atom_radius), 2))
    Wl = 1 - math.sqrt((math.pow((smallest_radius + mean_atom_radius), 2) - math.pow(mean_atom_radius, 2)) / math.pow((smallest_radius + mean_atom_radius), 2))
    y = float('%.5f' % (Ws / Wl))

    feature_list = [mean_VEC, VEC_dif, mean_electronegativity, electronegativity_dif,
                    atom_radius_dif, a2, y, melting_point_dif, Smix, aomiga, X,
                    Hmix, Hmix_dif, Hmix_z, Hmix_p, Hmix_n]

    return feature_list


# 分类SS:0;IM:1;AM:2;SS+IM:3
def data_analyze():

    df_alloy = read_csv(r'D:\study\sunshine\HEA\data\ML_last.csv', index_col=0)
    df_alloy = df_alloy.reset_index(drop=True)

    phase_ss = []
    label_ss = []

    #print(df_alloy['Hmix_n'].describe())
    '''
    for t in range(40):
        df_alloy = df_alloy[df_alloy['aomiga'] != df_alloy['aomiga'].max()]
        df_alloy = df_alloy[df_alloy['X'] != df_alloy['X'].max()]
        df_alloy = df_alloy[df_alloy['Hmix_n'] != df_alloy['Hmix_n'].max()]
        df_alloy = df_alloy[df_alloy['y'] != df_alloy['y'].max()]
    '''
    #df_alloy.to_csv(r'D:\study\sunshine\HEA\data\ML_last.csv')


    for i in list(df_alloy['label']):
        if i == 0 or i == 1 or i == 2 or i == 6 or i == 7 or i == 15 or i == 5 or i == 11:
            phase_ss.append('SS')
            label_ss.append(0)
        elif i == 3:
            phase_ss.append('IM')
            label_ss.append(1)
        elif i == 4:
            phase_ss.append('AM')
            label_ss.append(2)
        else:
            phase_ss.append('SS+IM')
            label_ss.append(3)


    '''
    df_alloy = df_alloy[(df_alloy['label'] == 0) | (df_alloy['label'] == 1) | (df_alloy['label'] == 5)]
    for i in list(df_alloy['label']):
        if i == 0:
            phase_ss.append('BCC')
            label_ss.append(0)
        elif i == 1:
            phase_ss.append('FCC')
            label_ss.append(1)
        else:
            phase_ss.append('DP')
            label_ss.append(2)
    '''

    '''
    df_alloy = df_alloy[(df_alloy['label'] == 0) | (df_alloy['label'] == 1) | (df_alloy['label'] == 3) | (df_alloy['label'] == 4)]

    for i in list(df_alloy['label']):
        if i == 0:
            phase_ss.append('BCC')
            label_ss.append(0)
        elif i == 1:
            phase_ss.append('FCC')
            label_ss.append(1)
        elif i == 3:
            phase_ss.append('IM')
            label_ss.append(2)
        elif i == 4:
            phase_ss.append('AM')
            label_ss.append(3)
    '''

    df_alloy['phase_ss'] = phase_ss
    df_alloy['label_ss'] = label_ss

    #print(df_alloy['phase_ss'].value_counts())
    sklearn_clas(df_alloy)


def sklearn_clas(df_alloy):
    df_alloy = df_alloy[(df_alloy['label_ss'] == 0) | (df_alloy['label_ss'] == 1) | (df_alloy['label_ss'] == 2)]
    #print(len(df_alloy))
    df_alloy = df_alloy.reset_index(drop=True)

    #predict_alloy_Al = df_alloy.loc[[350]]
    #predict_alloy_Fe = df_alloy.loc[[698]]
    #print(predict_alloy_Al)
    #print(predict_alloy_Fe)

    #df_alloy.to_csv(r'C:/Users/lzlfly/Desktop/HEA_data.csv', index=False, header=True)

    #X = df_alloy.drop(['name', 'full_name', 'electronegativity_dif', 'atom_radius_dif', 'Hmix_z', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    #X = df_alloy.drop(['name', 'full_name', 'electronegativity_dif', 'VEC_dif', 'Hmix_z', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    #X = df_alloy.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    X = df_alloy.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss', 'atom_radius_dif', 'electronegativity_dif', 'Hmix_z'], axis=1)

    #Al_X = predict_alloy_Al.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss', 'atom_radius_dif', 'electronegativity_dif', 'Hmix_z'], axis=1)
    #Fe_X = predict_alloy_Fe.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss', 'atom_radius_dif', 'electronegativity_dif', 'Hmix_z'], axis=1)

    y = df_alloy['label_ss']

    # feature_enginer(X, y)

    X = X[['y', 'Hmix']]
    #df_new = aug_feature(X, y)
    #X = pd.concat([X, df_new], axis=1)

    X = np.array(X)
    y = np.array(y)

    #Al_X = np.array(Al_X)
    #Fe_X = np.array(Fe_X)

    # X = X[:, [1, 15]]

    scaler = MinMaxScaler()

    '''
    from sklearn.decomposition import PCA
    pca = PCA(n_components=16)
    X= pca.fit_transform(X)
    '''
    '''
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)

    plt.scatter(X[:, 0], X[:, 1], marker='o', s=1)
    plt.show()
    '''

    X = scaler.fit_transform(X)
    #Al_X = scaler.fit_transform(Al_X)
    #Fe_X = scaler.fit_transform(Fe_X)

    #selected_feature_num(X, y)

    # 分割训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # X_train-(910, ), X_test-(228, )

    #knn(X_train, X_test, y_train, y_test, Al_X, Fe_X)
    #knn(X_train, y_train, Al_X, Fe_X)
    #svm_class(X_train, X_test, y_train, y_test, Al_X, Fe_X)
    #svm_class(X_train, y_train, Al_X, Fe_X)
    #Decisiontree(X_train, X_test, y_train, y_test)
    #Randomforest(X_train, X_test, y_train, y_test, Al_X, Fe_X)
    xgb(X_train, X_test, y_train, y_test)
    #GBDT(X_train, X_test, y_train, y_test)
    #lgb(X_train, X_test, y_train, y_test)

    #roc(X, y)
    # FCNN # 0.74783  0.75217

def feature_enginer(data, label):
    # filter
    # 方差选择法，返回值为特征选择后的数据  # 参数threshold为方差的阈值
    from sklearn.feature_selection import VarianceThreshold
    df = VarianceThreshold(threshold=5).fit_transform(data)
    print(df.shape)

    # 相关系数法
    from sklearn.feature_selection import SelectKBest
    #from scipy.stats import pearsonr
    #df = SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=5).fit_transform(data, label)
    #print(df)



    #from sklearn.feature_selection import chi2
    #df = SelectKBest(chi2, k=5).fit_transform(data, label)
    #print(df)


    # Wrapper
    from sklearn.feature_selection import RFE
    from xgboost import XGBClassifier
    df = RFE(estimator=XGBClassifier(), n_features_to_select=5).fit_transform(data, label)
    print(df)
    print('Selected Features: %s' % (df.support_))
    print('Feature ranking: %s' % (df.ranking_))


    # Embedded
    from sklearn.feature_selection import SelectFromModel
    from xgboost import XGBClassifier
    df = SelectFromModel(XGBClassifier(penalty="l1", C=0.1)).fit_transform(data, label)
    print(df)

    class LR(XGBClassifier):
        def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                     fit_intercept=True, intercept_scaling=1, class_weight=None,
                     random_state=None, solver='liblinear', max_iter=100,
                     multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

            self.threshold = threshold
            XGBClassifier.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                        fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                        class_weight=class_weight,
                                        random_state=random_state, solver=solver, max_iter=max_iter,
                                        multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
            self.l2 = XGBClassifier(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                         intercept_scaling=intercept_scaling, class_weight=class_weight,
                                         random_state=random_state, solver=solver, max_iter=max_iter,
                                         multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        def fit(self, X, y, sample_weight=None):
            super(LR, self).fit(X, y, sample_weight=sample_weight)
            self.coef_old_ = self.coef_.copy()
            self.l2.fit(X, y, sample_weight=sample_weight)

            cntOfRow, cntOfCol = self.coef_.shape
            for i in range(cntOfRow):
                for j in range(cntOfCol):
                    coef = self.coef_[i][j]
                    if coef != 0:
                        idx = [j]
                        coef1 = self.l2.coef_[i][j]
                        for k in range(cntOfCol):
                            coef2 = self.l2.coef_[i][k]
                            if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                                idx.append(k)
                        mean = coef / len(idx)
                        self.coef_[i][idx] = mean
            return self


    df = SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(data, label)
    print(df)


def aug_feature(data, label):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    #data = data[['X', 'y', 'Hmix_n', 'Hmix_p', 'mean_VEC']]
    x_poly = poly.fit_transform(data)
    df = pd.DataFrame(x_poly, columns=poly.get_feature_names())
    print(df.shape)

    return df


def selected_feature_num(data, label):

    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV

    model_xgb = XGBClassifier(learning_rate=0.2, max_depth=4, n_estimators=90)
    #model_knn = KNeighborsClassifier(leaf_size=1, n_jobs=1, n_neighbors=3, p=1, weights='distance')
    #model_svm = svm.SVC(C=100, decision_function_shape='ovo', degree=1, gamma='scale', kernel='rbf')
    model_rf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features='auto', n_estimators=73)

    #model.fit(data, label)
    #param_Grid = [{'learning_rate': [0.0001, 0.0003, 0.0002, 0.0005, 0.0007, 0.001, 0.01, 0.1, 0.2, 0.3],
                   #'max_depth': [1, 2, 3, 4, 5]}]
    #kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    #grid_search = GridSearchCV(model, param_grid=param_Grid, scoring='accuracy', n_jobs=-1, cv=kfold)

    #model = RandomForestClassifier()
    #param_grid = [{'criterion': ['gini', 'entropy'],
                   #'n_estimators': [i for i in range(10, 100)],
                   #'max_features': ['auto', 'sqrt', 'log2']}
                  #]
    #kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    #grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=kfold)

    # Instantiate RFECV visualizer with a random forest regressor
    #rfecv_knn = RFECV(estimator=model_knn, cv=5, scoring='accuracy')
    #rfecv_svm = RFECV(estimator=model_svm, cv=5, scoring='accuracy')
    rfecv_rf = RFECV(estimator=model_rf, cv=5, scoring='accuracy')
    rfecv_xgb = RFECV(estimator=model_xgb, cv=5, scoring='accuracy')

    #rfecv_knn.fit(data, label)  # Fit the data to the visualizer
    #rfecv_svm.fit(data, label)
    rfecv_rf.fit(data, label)
    rfecv_xgb.fit(data, label)

    #print("Optimal number of features : %d" % rfecv.n_features_)
    #print('Feature ranking: {}'.format(rfecv.ranking_))
    #print('Feature support: {}'.format(rfecv.support_))
    #print('Grid Scores: {}'.format(rfecv.grid_scores_))

    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(7, 7))
    plt.xlabel("选择的特征数量", fontsize=15)
    plt.ylabel("交叉验证准确率", fontsize=15)
    #plt.plot(range(1, len(rfecv_knn.grid_scores_) + 1), rfecv_knn.grid_scores_, linewidth=2, c='r')
    #plt.scatter(range(1, len(rfecv_knn.grid_scores_) + 1), rfecv_knn.grid_scores_, s=15, c='r')

    #plt.plot(range(1, len(rfecv_svm.grid_scores_) + 1), rfecv_svm.grid_scores_, linewidth=2, c='g')
    #plt.scatter(range(1, len(rfecv_svm.grid_scores_) + 1), rfecv_svm.grid_scores_, s=15, c='g')

    plt.plot(range(1, len(rfecv_rf.grid_scores_) + 1), rfecv_rf.grid_scores_, linewidth=2, c='r', label='RF')
    plt.scatter(range(1, len(rfecv_rf.grid_scores_) + 1), rfecv_rf.grid_scores_, s=15, c='r')

    plt.plot(range(1, len(rfecv_xgb.grid_scores_) + 1), rfecv_xgb.grid_scores_, linewidth=2, c='g', label='XGBoost')
    plt.scatter(range(1, len(rfecv_xgb.grid_scores_) + 1), rfecv_xgb.grid_scores_, s=15, c='g')

    plt.xticks(np.arange(0, 14), fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc='lower right', fontsize=13)
    plt.savefig('C:/Users/lzlfly/Desktop/map/feature_num_all.jpg', dpi=1500)
    # plt.show()


def roc(X, y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold
    model_list = ['KNN', 'SVM', 'DT', 'RF', 'XGBoost', 'GBDT']

    '''
    # BCC/FCC
    model_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=7, p=1, weights='distance')
    model_svm = svm.SVC(C=100, decision_function_shape='ovo', gamma='auto', kernel='rbf', probability=True)
    model_DT = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
    model_RF = RandomForestClassifier(criterion='entropy', max_features='log2', n_estimators=44)
    model_XGB = XGBClassifier(learning_rate=0.1, max_depth=5)
    model_GBDT = GradientBoostingClassifier(learning_rate=0.2, max_depth=5)
    '''

    # SS/IM
    model_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=4, p=1, weights='distance')
    model_svm = svm.SVC(C=1000, decision_function_shape='ovo', gamma='scale', kernel='rbf', probability=True)
    model_DT = tree.DecisionTreeClassifier(criterion='gini', splitter='random')
    model_RF = RandomForestClassifier(criterion='entropy', max_features='sqrt', n_estimators=79)
    model_XGB = XGBClassifier(learning_rate=0.2, max_depth=3)
    model_GBDT = GradientBoostingClassifier(learning_rate=0.3, max_depth=3)

    '''
    # SS/AM
    model_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1, weights='uniform')
    model_svm = svm.SVC(C=10, decision_function_shape='ovo', gamma='scale', kernel='rbf', probability=True)
    model_DT = tree.DecisionTreeClassifier(criterion='entropy', splitter='random')
    model_RF = RandomForestClassifier(criterion='gini', max_features='auto', n_estimators=29)
    model_XGB = XGBClassifier(learning_rate=0.2, max_depth=2)
    model_GBDT = GradientBoostingClassifier(learning_rate=0.1, max_depth=1)
    '''
    '''
    # IM/AM
    model_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=4, p=1, weights='distance')
    model_svm = svm.SVC(C=10, decision_function_shape='ovo', gamma='scale', kernel='rbf', probability=True)
    model_DT = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
    model_RF = RandomForestClassifier(criterion='gini', max_features='auto', n_estimators=21)
    model_XGB = XGBClassifier(learning_rate=0.2, max_depth=3)
    model_GBDT = GradientBoostingClassifier(learning_rate=0.1, max_depth=3)
    '''

    cv = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    mean_tpr_knn = 0
    mean_tpr_svm = 0
    mean_tpr_DT = 0
    mean_tpr_RF = 0
    mean_tpr_XGB = 0
    mean_tpr_GBDT = 0

    mean_fpr = np.linspace(0, 1, 100)
    cnt = 0
    for i, (train, test) in enumerate(cv.split(X, y)):
        cnt += 1
        probas_knn = model_knn.fit(X[train], y[train]).predict_proba(X[test])
        probas_svm = model_svm.fit(X[train], y[train]).predict_proba(X[test])
        probas_DT = model_DT.fit(X[train], y[train]).predict_proba(X[test])
        probas_RF = model_RF.fit(X[train], y[train]).predict_proba(X[test])
        probas_XGB = model_XGB.fit(X[train], y[train]).predict_proba(X[test])
        probas_GBDT = model_GBDT.fit(X[train], y[train]).predict_proba(X[test])

        fpr_knn, tpr_knn, thresholds_knn = metrics.roc_curve(y[test], probas_knn[:, 1])
        fpr_svm, tpr_svm, thresholds_svm = metrics.roc_curve(y[test], probas_svm[:, 1])
        fpr_DT, tpr_DT, thresholds_DT = metrics.roc_curve(y[test], probas_DT[:, 1])
        fpr_RF, tpr_RF, thresholds_RF = metrics.roc_curve(y[test], probas_RF[:, 1])
        fpr_XGB, tpr_XGB, thresholds_XGB = metrics.roc_curve(y[test], probas_XGB[:, 1])
        fpr_GBDT, tpr_GBDT, thresholds_GBDT = metrics.roc_curve(y[test], probas_GBDT[:, 1])

        mean_tpr_knn += np.interp(mean_fpr, fpr_knn, tpr_knn)
        mean_tpr_svm += np.interp(mean_fpr, fpr_svm, tpr_svm)
        mean_tpr_DT += np.interp(mean_fpr, fpr_DT, tpr_DT)
        mean_tpr_RF += np.interp(mean_fpr, fpr_RF, tpr_RF)
        mean_tpr_XGB += np.interp(mean_fpr, fpr_XGB, tpr_XGB)
        mean_tpr_GBDT += np.interp(mean_fpr, fpr_GBDT, tpr_GBDT)
        mean_tpr_knn[0] = 0
        mean_tpr_svm[0] = 0
        mean_tpr_DT[0] = 0
        mean_tpr_RF[0] = 0
        mean_tpr_XGB[0] = 0
        mean_tpr_GBDT[0] = 0

        #roc_auc = metrics.auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))
    from matplotlib.gridspec import GridSpec
    fig, ax1 = plt.subplots(figsize=(5, 5))
    #gs= GridSpec(4, 8)
    #plt.subplot(gs[0:, 0:5])
    ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

    mean_tpr_knn /= cnt  
    mean_tpr_svm /= cnt
    mean_tpr_DT /= cnt
    mean_tpr_RF /= cnt
    mean_tpr_XGB /= cnt
    mean_tpr_GBDT /= cnt

    mean_tpr_knn[-1] = 1.0  
    mean_tpr_svm[-1] = 1.0
    mean_tpr_DT[-1] = 1.0
    mean_tpr_RF[-1] = 1.0
    mean_tpr_XGB[-1] = 1.0
    mean_tpr_GBDT[-1] = 1.0
    mean_auc_knn = metrics.auc(mean_fpr, mean_tpr_knn)
    mean_auc_svm = metrics.auc(mean_fpr, mean_tpr_svm)
    mean_auc_DT = metrics.auc(mean_fpr, mean_tpr_DT)
    mean_auc_RF = metrics.auc(mean_fpr, mean_tpr_RF)
    mean_auc_XGB = metrics.auc(mean_fpr, mean_tpr_XGB)
    mean_auc_GBDT = metrics.auc(mean_fpr, mean_tpr_GBDT)

    ax1.plot(mean_fpr, mean_tpr_knn, color='r', label='KNN', lw=1)
    ax1.plot(mean_fpr, mean_tpr_svm, color='darkgreen', label='SVM', lw=1)
    ax1.plot(mean_fpr, mean_tpr_DT, color='b', label='DT', lw=1)
    ax1.plot(mean_fpr, mean_tpr_RF, color='darkorange', label='RF', lw=1)
    ax1.plot(mean_fpr, mean_tpr_XGB, color='dodgerblue', label='XGBoost', lw=1)
    ax1.plot(mean_fpr, mean_tpr_GBDT, color='black', label='GBDT', lw=1)

    ax1.set_xlim([-0.05, 1.05])  
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.text(0.0, 1.01, '(c)', fontsize=11)

    #plt.subplot(gs[2:, 6:])
    #ax2 = fig.add_axes([0.55, 0.15, 0.3, 0.3])

    #bar_width = 0.3
    #ax2.bar(x=np.arange(len(model_list)), height=height, width=bar_width)
    #ax2.set_ylim([0.8, 1.0])
    #ax2.set_xticks(np.arange(len(model_list)))
    #ax2.set_xticklabels(model_list)
    #ax2.tick_params(labelsize=5)
    #ax2.set_xlabel('Model', fontsize=6)
    #ax2.set_ylabel('AUC Value', fontsize=7)

    fig.savefig('C:/Users/lzlfly/Desktop/map/roc-SS_IM.jpg', dpi=1500)



def plot_decision_boundary(clf, X_t):
    import matplotlib as mpl
    #xp = np.linspace(axes[0], axes[1], 300) 
    #yp = np.linspace(axes[2], axes[3], 300) 

    x0p = np.linspace(X_t[:, 0].min()-0.05, X_t[:, 0].max()+0.05, 300)
    x1p = np.linspace(X_t[:, 1].min()-0.05, X_t[:, 1].max()+0.05, 300)
    '''
    x2p = np.linspace(X_t[:, 2].min()-0.05, X_t[:, 2].max()+0.05, 300)
    x3p = np.linspace(X_t[:, 3].min()-0.05, X_t[:, 3].max()+0.05, 300)
    x4p = np.linspace(X_t[:, 4].min()-0.05, X_t[:, 4].max()+0.05, 300)
    x5p = np.linspace(X_t[:, 5].min()-0.05, X_t[:, 5].max()+0.05, 300)
    x6p = np.linspace(X_t[:, 6].min()-0.05, X_t[:, 6].max()+0.05, 300)
    x7p = np.linspace(X_t[:, 7].min()-0.05, X_t[:, 7].max()+0.05, 300)
    x8p = np.linspace(X_t[:, 8].min()-0.05, X_t[:, 8].max()+0.05, 300)
    x9p = np.linspace(X_t[:, 9].min()-0.05, X_t[:, 9].max()+0.05, 300)
    x10p = np.linspace(X_t[:, 10].min()-0.05, X_t[:, 10].max()+0.05, 300)
    x11p = np.linspace(X_t[:, 11].min()-0.05, X_t[:, 11].max()+0.05, 300)
    x12p = np.linspace(X_t[:, 12].min()-0.05, X_t[:, 12].max()+0.05, 300)
    x13p = np.linspace(X_t[:, 13].min()-0.05, X_t[:, 13].max()+0.05, 300)
    x14p = np.linspace(X_t[:, 14].min()-0.05, X_t[:, 14].max()+0.05, 300)
    x15p = np.linspace(X_t[:, 15].min()-0.05, X_t[:, 15].max()+0.05, 300)
    '''
    x1, y1 = np.meshgrid(x0p, x1p) 
    xy1 = np.c_[x1.ravel(), y1.ravel()] 
    '''
    x2, y2 = np.meshgrid(x2p, x3p)
    xy2 = np.c_[x2.ravel(), y2.ravel()]

    x3, y3 = np.meshgrid(x4p, x5p)
    xy3 = np.c_[x3.ravel(), y3.ravel()]

    x4, y4 = np.meshgrid(x6p, x7p)
    xy4 = np.c_[x4.ravel(), y4.ravel()]

    x5, y5 = np.meshgrid(x8p, x9p)
    xy5 = np.c_[x5.ravel(), y5.ravel()]

    x6, y6 = np.meshgrid(x10p, x11p)
    xy6 = np.c_[x6.ravel(), y6.ravel()]

    x7, y7 = np.meshgrid(x12p, x13p)
    xy7 = np.c_[x7.ravel(), y7.ravel()]

    x8, y8 = np.meshgrid(x14p, x15p)
    xy8 = np.c_[x8.ravel(), y8.ravel()]
    
    xy = np.c_[xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8]
    '''
    y_pred = clf.predict(xy1).reshape(x1.shape)  
    #custom_cmap = mpl.colors.ListedColormap(['#FF0000', '#006400', '#FF8C00', '#4169E1'])
    custom_cmap = mpl.colors.ListedColormap(['#FF0000', '#006400', '#FF8C00'])
    plt.figure(figsize=(4, 4))
    plt.contourf(x1, y1, y_pred, alpha=0.3, cmap=custom_cmap)



# knn


def knn(X_train, y_train, Al_X, Fe_X):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold

    param_grid = [
        {'weights': ['uniform', 'distance'],
         'n_neighbors': [i for i in range(1, 10)],
         'p': [i for i in range(1, 6)],
         'leaf_size': [i for i in range(1, 10)],
         'n_jobs': [i for i in range(1, 10)]
         }
    ]

    model = KNeighborsClassifier()
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=kfold, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    #print(grid_search.best_estimator_)
    #print(grid_search.best_score_)
    print(grid_search.best_params_)

    #y_pred_knn = grid_search.predict(X_test)


    y_pred_Al = grid_search.predict(Al_X)
    y_pred_Fe = grid_search.predict(Fe_X)
    print('Al: {},Fe: {}'.format(y_pred_Al, y_pred_Fe))

    '''
    model = KNeighborsClassifier(leaf_size=1, n_jobs=1, n_neighbors=3, p=1, weights='distance')
    model.fit(X_train, y_train)
    plot_decision_boundary(model, X_t=X_test)

    p1 = plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='r', s=7)
    p2 = plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='darkgreen', s=7)
    p3 = plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], color='darkorange', s=7)
    p4 = plt.scatter(X_test[y_test == 3, 0], X_test[y_test == 3, 1], color='royalblue', s=7)

    plt.legend([p1, p2, p3, p4], ['SS', 'IM', 'AM', 'SS+IM'], loc='lower left', fontsize=8, markerscale=3)
    #plt.legend([p1, p2, p3], ['SS', 'IM', 'AM'], loc='lower left', fontsize=8, markerscale=3)
    plt.xlabel(r"$\gamma$", fontsize=10)
    plt.ylabel(r"$\Delta{H_{mix}}$", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    #plt.text(0, 0.98, '(a)', fontsize=9)
    plt.savefig('C:/Users/lzlfly/Desktop/map/knn_boundary.jpg', dpi=1500)
    #plt.show()
    '''


# {'C': 1000, 'decision_function_shape': 'ovo', 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
# {'C': 100, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}  16 features-4-class-0.6974
# {'C': 100, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}  13 features-4-class-0.6930
def svm_class(X_train, y_train, Al_X, Fe_X):
    # SVM
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold

    model = svm.SVC()
    param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                   'kernel': ['poly', 'rbf', 'sigmoid'],
                   'gamma': ['scale', 'auto'],
                   'decision_function_shape': ['ovo', 'ovr'],
                   'degree': [i for i in range(1, 10)]}]
    #C:1000, decision:ovo, gamma:auto, kernel: rbf
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=kfold, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    #print(grid_search.best_estimator_)
    #print(grid_search.best_score_)
    print(grid_search.best_params_)

    #y_pred = grid_search.predict(X_test)

    y_pred_Al = grid_search.predict(Al_X)
    y_pred_Fe = grid_search.predict(Fe_X)
    print('Al: {},Fe: {}'.format(y_pred_Al, y_pred_Fe))


    '''
    model = svm.SVC(C=100, decision_function_shape='ovo', degree=1, gamma='scale', kernel='rbf')
    model.fit(X_train, y_train)
    plot_decision_boundary(model, X_t=X_test)

    p1 = plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='r', s=7)
    p2 = plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='darkgreen', s=7)
    p3 = plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], color='darkorange', s=7)
    p4 = plt.scatter(X_test[y_test == 3, 0], X_test[y_test == 3, 1], color='royalblue', s=7)

    plt.legend([p1, p2, p3, p4], ['SS', 'IM', 'AM', 'SS+IM'], loc='lower left', fontsize=8, markerscale=3)
    #plt.legend([p1, p2, p3], ['SS', 'IM', 'AM'], loc='lower left', fontsize=8, markerscale=3)
    plt.xlabel(r"$\gamma$", fontsize=10)
    plt.ylabel(r"$\Delta{H_{mix}}$", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    #plt.text(0, 0.98, '(b)', fontsize=9)
    plt.savefig('C:/Users/lzlfly/Desktop/map/svm_boundary.jpg', dpi=1500)
    # plt.show()
    '''


# {'criterion': 'gini', 'max_depth': 8, 'splitter': 'best'}
def Decisiontree(X_train, X_test, y_train, y_test):
    # 决策树
    from sklearn import tree
    from sklearn.model_selection import StratifiedKFold
    '''
    model = tree.DecisionTreeClassifier()
    param_grid = [{'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_depth': [i for i in range(1, 16)]}
    ]
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=kfold, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    #print(grid_search.best_estimator_)
    #print(grid_search.best_score_)
    print(grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    print('Decisiontree准确率：{:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))
    '''

    model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', random_state=10, max_depth=3)
    model.fit(X_train, y_train)

    feature_names = ['lambda', 'gamma', 'VEC', 'VEC_dif', 'Electronegativity', 'Electronegativity_dif',
                     'Atom_radius_dif', 'Alpha_2', 'Melting_point_dif', 'Smix', 'Omega', 'Hmix', 'Hmix_dif',
                     'Hmix_z', 'Hmix_p', 'Hmix_n']


    import graphviz
    import pydotplus
    from IPython.display import Image, display
    dot_data = tree.export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            filled=True,
            impurity=False,
            rounded=True
        )
    #graph = graphviz.Source(dot_data)
    #print(graph)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #graph.get_nodes()[4].set_fillcolor("#FFF2DD")
    graph.write_png('C:/Users/lzlfly/Desktop/map/DT_depth.jpg')


    '''
    plt.figure(figsize=(10, 10))
    from sklearn.tree import plot_tree
    plot_tree(model, filled=True)
    plt.savefig('C:/Users/lzlfly/Desktop/map/DT_depth.pdf', dpi=1500)
    plt.show()
    '''


# {'criterion': 'entropy', 'max_depth': 15, 'max_features': 'auto', 'n_estimators': 85}
# {'criterion': 'gini', 'max_depth': 13, 'max_features': 'sqrt', 'n_estimators': 48}  16 features-4-class-0.7675
# {'criterion': 'entropy', 'max_depth': 15, 'max_features': 'auto', 'n_estimators': 73}  13 features-4-class-0.7807
def Randomforest(X_train, X_test, y_train, y_test, Al_X, Fe_X):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    model = RandomForestClassifier()
    param_grid = [{'criterion': ['gini', 'entropy'],
                   'n_estimators': [i for i in range(10, 100)],
                   'max_features': ['auto', 'sqrt', 'log2'],
                   'max_depth': [i for i in range(1, 16)]}
                  ]
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=kfold)
    grid_search.fit(X_train, y_train)
    #print(grid_search.best_estimator_)
    #print(grid_search.best_score_)
    print(grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    print('Randomforest准确率： {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))

    y_pred_Al = grid_search.predict(Al_X)
    y_pred_Fe = grid_search.predict(Fe_X)
    print('Al: {},Fe: {}'.format(y_pred_Al, y_pred_Fe))




# {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 30}
# {'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 20}  16 features-4-class-0.7412
# {'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 90}  13 features-4-class-0.7807
def xgb(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold
    

    model = XGBClassifier(learning_rate=0.2, max_depth=12, n_estimators=90)
    model.fit(X_train, y_train)
    plot_decision_boundary(model, X_t=X_test)

    p1 = plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='r', s=7)
    p2 = plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='darkgreen', s=7)
    p3 = plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], color='darkorange', s=7)   # turquoise  darkorange
    #p4 = plt.scatter(X_test[y_test == 3, 0], X_test[y_test == 3, 1], color='royalblue', s=7)

    plt.legend([p1, p2, p3], ['固溶相', '金属间化合物', '非晶态'], loc='lower left', fontsize=6, markerscale=2)
    #plt.legend([p1, p2, p3], ['SS', 'IM', 'AM'], loc='lower left', fontsize=8, markerscale=3)
    plt.xlabel(r"$\gamma$", fontsize=10)
    plt.ylabel(r"$\Delta{H_{mix}}$", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    #plt.text(0, 0.98, '(d)', fontsize=9)
    plt.savefig('C:/Users/lzlfly/Desktop/map/xgb_boundary_zl.jpg', dpi=1500)
    # plt.show()


    '''
    print(metrics.confusion_matrix(y_test, y_pred))
    target_names = ['BCC', 'FCC']
    print(metrics.classification_report(y_test, y_pred, target_names=target_names, digits=3))
    '''


def lgb(X_train, X_test, y_train, y_test):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    param_Grid = [{'learning_rate': [0.0001, 0.0003, 0.0002, 0.0005, 0.0007, 0.001, 0.01, 0.1, 0.2, 0.3],
                   'max_depth': [1, 2, 3, 4, 5]}]
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    model = LGBMClassifier()
    grid_search = GridSearchCV(model, param_grid=param_Grid, scoring='accuracy', n_jobs=-1, cv=kfold)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_)
    print(grid_search.best_score_)
    print(grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print('LGBM准确率： {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))
    


# {'learning_rate': 0.3, 'max_depth': 8, 'n_estimators': 50}
def GBDT(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    param_grid = [{'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                   'max_depth': [i for i in range(1, 16)],
                   'n_estimators': [i for i in range(10, 100, 10)]}]
    kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)

    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=kfold)
    grid_search.fit(X_train, y_train)
    #print(grid_search.best_estimator_)
    #print(grid_search.best_score_)
    print(grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print('GBDT准确率： {:.4f}'.format(metrics.accuracy_score(y_test, y_pred)))

   



def connection():
    df_ML = read_csv(r'D:\study\sunshine\HEA\data\ML_last.csv', index_col=0)
    df_ML = df_ML.reset_index(drop=True)

    phase_ss = []
    label_ss = []

    for i in list(df_ML['label']):
        if i == 0 or i == 1 or i == 2 or i == 5 or i == 6 or i == 7 or i == 11 or i == 15:
            phase_ss.append('SS')
            label_ss.append(0)
        elif i == 3:
            phase_ss.append('IM')
            label_ss.append(1)
        elif i == 4:
            phase_ss.append('AM')
            label_ss.append(2)
        else:
            phase_ss.append('SS+IM')
            label_ss.append(3)

    df_ML['phase_ss'] = phase_ss
    df_ML['label_ss'] = label_ss

    #three_D(df_ML)
    #pierson(df_ML)
    #feature_f_classif(df_ML)
    #importance_merge(df_ML)
    #between_class(df_ML)
    importance_zl(df_ML)



def between_class(df_ML):
    df_ML = df_ML[(df_ML['label_ss'] == 0) | (df_ML['label_ss'] == 1) | (df_ML['label_ss'] == 2)]
    df_ML = df_ML.reset_index(drop=True)

    X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    y = df_ML['label_ss']
    
    print(len(df_ML))

    df_ML_SS = df_ML[df_ML['label_ss'] == 0]
    df_ML_IM = df_ML[df_ML['label_ss'] == 1]
    df_ML_AM = df_ML[df_ML['label_ss'] == 2]
    #df_ML_MP = df_ML[df_ML['label_ss'] == 3]


    select_feature = ['X', 'y', 'aomiga', 'Hmix', 'mean_VEC']
    x_data = [r"$\lambda$", r"$\gamma$", r"$\Omega$", r"$\Delta{H_{mix}}$", r"${VEC}$"]
    plt.figure(figsize=(18, 20))
    
    n = 1
    for i in select_feature:
        for j in select_feature:
            if i == j:
                continue
            else:
                plt.subplot(5, 4, n)
                plt.scatter(df_ML_SS[j], df_ML_SS[i], c='r', marker='o', s=9)
                plt.scatter(df_ML_IM[j], df_ML_IM[i], c='darkgreen', marker='o', s=9)
                plt.scatter(df_ML_AM[j], df_ML_AM[i], c='turquoise', marker='o', s=9)
                #plt.scatter(df_ML_MP[j], df_ML_MP[i], c='royalblue', marker='o', s=9)
                # FF0000  #006400   #FF8C00  #4169E1
                plt.xlabel(x_data[select_feature.index(j)], fontsize=25)
                if n == 1 or n == 5 or n == 9 or n == 13 or n == 17:
                    plt.ylabel(x_data[select_feature.index(i)], fontsize=25)
                plt.xticks([])
                plt.yticks([])
                if n == 1:
                    plt.legend(['SS', 'IM', 'AM'], loc='upper left', fontsize=17, markerscale=6)
                n += 1


    plt.subplots_adjust(wspace=0.05)
    plt.savefig('C:/Users/lzlfly/Desktop/map/between_SM.jpg', dpi=1500)



def three_D(df_ML):
    from mpl_toolkits.mplot3d import Axes3D
    df_3D = df_ML.loc[:, ['Hmix', 'y', 'a2', 'label_ss']]
    df_3D_SS = df_3D[df_3D['label_ss'] == 0]  
    df_3D_IM = df_3D[df_3D['label_ss'] == 1]   
    df_3D_AM = df_3D[df_3D['label_ss'] == 2]  
    df_3D_SI = df_3D[df_3D['label_ss'] == 3]   
    X_SS = np.array(df_3D_SS)
    X_IM = np.array(df_3D_IM)
    X_AM = np.array(df_3D_AM)
    X_SI = np.array(df_3D_SI)

    ax = plt.subplot(111, projection='3d')
    ax.scatter(X_SS[:, 0], X_SS[:, 1], X_SS[:, 2], c='r', s=1)
    ax.scatter(X_IM[:, 0], X_IM[:, 1], X_IM[:, 2], c='g', s=1)
    ax.scatter(X_AM[:, 0], X_AM[:, 1], X_AM[:, 2], c='y', s=1)
    ax.scatter(X_SI[:, 0], X_SI[:, 1], X_SI[:, 2], c='b', s=1)
    ax.set_xlabel('Hmix')
    ax.set_ylabel('y')
    ax.set_zlabel('a2')
    ax.legend(['SS', 'IM', 'AM', 'SS+IM'])
    plt.show()


def pierson(df_ML):
    df_ML= df_ML[(df_ML['label_ss'] == 0) | (df_ML['label_ss'] == 1) | (df_ML['label_ss'] == 2)]
    df_ML = df_ML.reset_index(drop=True)

    X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    corr_feature = X.corr().round(2)

    print(corr_feature)
    #corr_feature.to_csv(r'D:\study\sunshine\HEA\data\pierson.csv', index=False, header=True)
    import seaborn
    #plt.figure(figsize=(12, 10), dpi=600)
    #seaborn.set(font_scale=0.5)
    h = seaborn.heatmap(corr_feature, center=0, annot=True, cmap='YlGnBu',
                    annot_kws={'size': 6}, cbar=False)
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=6)
    
    plt.xticks(np.arange(0, 16) + 0.5, [r"$\lambda$", r"$\gamma$", r"${VEC}$", r"$\delta_{VEC}$", r"$\chi_{arg}$",
                                        r"$\delta_\chi$", r"$\delta_r$", r"$\alpha_2$", r"$\delta_T$",
                                        r"$\Delta{S_{mix}}$", r"$\Omega$", r"$\Delta{H_{mix}}$", r"$\delta{H_{mix}}$",
                                        r"$\delta{H_{mix}^0}$", r"$\delta{H_{mix}^+}$", r"$\delta{H_{mix}^-}$"], fontsize=7, rotation=360)

    plt.yticks(np.arange(0, 16) + 0.5, [r"$\lambda$", r"$\gamma$", r"${VEC}$", r"$\delta_{VEC}$", r"$\chi_{arg}$",
                                        r"$\delta_\chi$", r"$\delta_r$", r"$\alpha_2$", r"$\delta_T$",
                                        r"$\Delta{S_{mix}}$", r"$\Omega$", r"$\Delta{H_{mix}}$", r"$\delta{H_{mix}}$",
                                        r"$\delta{H_{mix}^0}$", r"$\delta{H_{mix}^+}$", r"$\delta{H_{mix}^-}$"], fontsize=7)

    #plt.yticks(np.arange(0, 15)+0.5, ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
               # 'f10', 'f11', 'f12', 'f13', 'f14', 'f15'], fontsize=6)
    plt.savefig('C:/Users/lzlfly/Desktop/map/heatmap.jpg', dpi=1500)
    #plt.show()


def feature_f_classif(df_ML):
    df_ML = df_ML[(df_ML['label_ss'] == 0) | (df_ML['label_ss'] == 1) | (df_ML['label_ss'] == 2)]
    df_ML = df_ML.reset_index(drop=True)

    X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    y = df_ML['label_ss']
  

    plt.figure(figsize=(10, 10))
    bar_width = 0.3

    plt.subplot(2, 2, 1)
    from sklearn.ensemble import RandomForestClassifier
    # SS/IM/AM-param
    model = RandomForestClassifier(criterion='entropy', max_features='auto', n_estimators=85, max_depth=15, random_state=10, )
    # BCC/FCC/DP-param
    #model = RandomForestClassifier(criterion='entropy', max_features='log2', n_estimators=12, random_state=10)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print(importance)
    #x_data = [r"$\lambda$", r"$\Omega$", r"$\delta_r$", r"$\gamma$", r"$\delta{H_{mix}^-}$",
              #r"$\delta_{VEC}$", r"$VEC$", r"$\Delta{H_{mix}}$", r"$\Delta{S_{mix}}$",
              #r"$\chi_{arg}$", r"$\delta_T$", r"$\delta{H_{mix}^+}$", r"$\delta{H_{mix}^0}$",
              #r"$\chi_{arg}$", r"$\delta_\chi$", r"$\alpha_2$"]   # SS/IM/AM/MP

    x_data = [r"$\lambda$", r"$\gamma$", r"$\Omega$", r"$\Delta{H_{mix}}$", r"$\delta_r$",
              r"$\delta_{VEC}$", r"$\delta{H_{mix}^{0-}}$", r"$\Delta{S_{mix}}$", r"$VEC$",
              r"$\delta{H_{mix}^0}$", r"$\delta{H_{mix}^{0+}}$", r"$\chi_{arg}$",
              r"$\delta{H_{mix}}$", r"$\delta_T$", r"$\delta_\chi$", r"$\alpha_2$"]   # SS/IM/AM

    #x_data = [r"$VEC$", r"$\delta{H_{mix}^-}$", r"$\delta_{VEC}$", r"$\delta{H_{mix}^+}$",
              #r"$\chi_{arg}$", r"$\lambda$", r"$\delta_\chi$", r"$\Delta{S_{mix}}$",
              #r"$\alpha_2$", r"$\delta_r$", r"$\delta{H_{mix}^-}$", r"$\delta{H_{mix}}$",
              #r"$\Omega$", r"$\delta_T$", r"$\gamma$", r"$\Delta{H_{mix}}$"]  # BCC/FCC/DP

    x_data.reverse()
    y_data = list(importance)
    y_data.reverse()
    plt.barh(y=range(len(x_data)), width=y_data, height=bar_width)
    plt.yticks(np.arange(len(x_data)), x_data, fontsize=10)
    plt.xticks(fontsize=10)
    #plt.xlabel('Rank of importance of feature in RandomForest Classifier', fontsize=10)
    plt.text(0.0, 16.2, '(a)', fontsize=12)  # SM
    #plt.text(0.23, 0.1, '(a)', fontsize=8)   # BF


    plt.subplot(2, 2, 2)
    from sklearn.tree import DecisionTreeClassifier
    # SS/IM/AM-params
    model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=8, random_state=10)
    # BCC/FCC/MP-params
    #model = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=10)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print(importance)
    #x_data = [r"$\delta{H_{mix}^-}$", r"$\gamma$", r"$\delta{H_{mix}^0}$", r"$\Delta{S_{mix}}$",
              #r"$\delta{H_{mix}^+}$", r"$\delta_{VEC}$", r"$VEC$", r"$\lambda$", r"$\chi_{arg}$",
              #r"$\alpha_2$", r"$\delta_T$", r"$\Omega$", r"$\delta_r$", r"$\Delta{H_{mix}}$",
              #r"$\delta{H_{mix}}$", r"$\delta_\chi$"]  # SS/IM/AM/MP

    x_data = [r"$\lambda$", r"$\delta_T$", r"$\Delta{H_{mix}}$", r"$\delta{H_{mix}^{0+}}$",
              r"$\Delta{S_{mix}}$", r"$\delta_{VEC}$", r"$\gamma$", r"$\delta{H_{mix}^0}$",
              r"$\delta_\chi$", r"$VEC$", r"$\alpha_2$", r"$\chi_{arg}$", r"$\delta{H_{mix}}$",
              r"$\Omega$", r"$\delta_r$", r"$\delta{H_{mix}^{0-}}$"]

    #x_data = [r"$\chi_{arg}$", r"$\delta_\chi$", r"$VEC$", r"$\delta{H_{mix}^-}$", r"$\delta_r$",
              #r"$\alpha_2$", r"$\Delta{S_{mix}}$", r"$\delta{H_{mix}^+}$", r"$\delta_{VEC}$",
              #r"$\delta{H_{mix}^0}$", r"$\lambda$", r"$\delta{H_{mix}}$", r"$\gamma$",
              #r"$\Omega$", r"$\delta_T$", r"$\Delta{H_{mix}}$"]   # BCC/FCC/DP

    x_data.reverse()
    y_data = list(importance)
    y_data.reverse()
    plt.barh(y=range(len(x_data)), width=y_data, height=bar_width)
    plt.yticks(np.arange(len(x_data)), x_data, fontsize=10)
    plt.xticks(fontsize=10)
    #plt.xlabel('Feature importance ranking by DecisionTree Classifier', fontsize=10)
    plt.text(0.0, 16.2, '(b)', fontsize=12)   # SM
    #plt.text(0.21, 0.1, '(b)', fontsize=8)    # BF


    plt.subplot(2, 2, 3)
    from xgboost import XGBClassifier
    # SS/IM/AM-params
    model = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=30, random_state=10)
    # BCC/FCC/DP-params
    #model = XGBClassifier(learning_rate=0.3, max_depth=5, random_state=10)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print(importance)
    #x_data = [r"$\lambda$", r"$\gamma$", r"$\delta{H_{mix}^-}$", r"$\delta{H_{mix}^+}$",
              #r"$VEC$", r"$\Delta{H_{mix}}$", r"$\Omega$", r"$\chi_{arg}$", r"$\delta_{VEC}$",
              #r"$\Delta{S_{mix}}$", r"$\delta{H_{mix}}$", r"$\delta{H_{mix}^0}$", r"$\delta_T$",
              #r"$\alpha_2$", r"$\delta_r$", r"$\delta_\chi$"]   # SS/IM/AM/MP

    x_data = [r"$\lambda$", r"$\delta{H_{mix}^0}$", r"$\gamma$", r"$\delta{H_{mix}^{0+}}$",
              r"$\Omega$", r"$VEC$", r"$\delta{H_{mix}^{0-}}$", r"$\Delta{H_{mix}}$",
              r"$\Delta{S_{mix}}$", r"$\chi_{arg}$", r"$\delta_{VEC}$", r"$\delta_r$",
              r"$\alpha_2$", r"$\delta_T$", r"$\delta_\chi$", r"$\delta{H_{mix}}$"]    # SS/IM/AM

    #x_data = [r"$VEC$", r"$\delta_{VEC}$", r"$\delta{H_{mix}^+}$", r"$\delta_r$", r"$\chi_{arg}$",
              #r"$\gamma$", r"$\Delta{H_{mix}}$", r"$\delta{H_{mix}}$", r"$\alpha_2$",
              #r"$\delta{H_{mix}^-}$", r"$\Delta{S_{mix}}$", r"$\Omega$", r"$\lambda$", r"$\delta_T$",
              #r"$\delta{H_{mix}^0}$", r"$\delta_\chi$"]

    x_data.reverse()
    y_data = list(importance)
    y_data.reverse()
    plt.barh(y=range(len(x_data)), width=y_data, height=bar_width)
    plt.yticks(np.arange(len(x_data)), x_data, fontsize=10)
    plt.xticks(fontsize=10)
    #plt.xlabel('Feature importance by XGBoost Classifier', fontsize=10)
    plt.text(0.0, 16.2, '(c)', fontsize=12)   # SM
    #plt.text(0.27, 0.1, '(c)', fontsize=8)   # BF


    plt.subplot(2, 2, 4)
    from sklearn.ensemble import GradientBoostingClassifier
    # SS/IM/AM-params
    model = GradientBoostingClassifier(learning_rate=0.3, max_depth=8, n_estimators=50, random_state=10)
    # BCC/FCC/MP-params
    #model = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, random_state=10)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    print(importance)
    #x_data = [r"$\gamma$", r"$\lambda$", r"$VEC$", r"$\Omega$", r"$\delta_{VEC}$",
              #r"$\delta{H_{mix}^+}$", r"$\Delta{S_{mix}}$", r"$\delta_T$", r"$\Delta{H_{mix}}$",
              #r"$\chi_{arg}$", r"$\delta{H_{mix}^-}$", r"$\alpha_2$", r"$\delta_\chi$",
              #r"$\delta{H_{mix}}$", r"$\delta_r$", r"$\delta{H_{mix}^0}$"]  # SS/IM/AM/MP

    x_data = [r"$\gamma$", r"$\Omega$", r"$\delta{H_{mix}^{0+}}$", r"$\lambda$", r"$\Delta{S_{mix}}$",
              r"$VEC$", r"$\Delta{H_{mix}}$", r"$\delta_{VEC}$", r"$\delta{H_{mix}^{0-}}$",
              r"$\chi_{arg}$", r"$\delta_T$", r"$\delta_\chi$", r"$\delta_r$", r"$\alpha_2$",
              r"$\delta{H_{mix}^0}$", r"$\delta{H_{mix}}$"]   # SS/IM/AM

    #x_data = [r"$VEC$", r"$\delta_{VEC}$", r"$\delta{H_{mix}^+}$", r"$\Delta{H_{mix}}$",
              #r"$\chi_{arg}$", r"$\gamma$", r"$\Delta{S_{mix}}$", r"$\lambda$", r"$\delta_T$",
              #r"$\delta_r$", r"$\alpha_2$", r"$\delta{H_{mix}^-}$", r"$\delta{H_{mix}}$",
              #r"$\delta{H_{mix}^0}$", r"$\delta_\chi$", r"$\Omega$"]   # SM

    x_data.reverse()
    y_data = list(importance)
    y_data.reverse()
    plt.barh(y=range(len(x_data)), width=y_data, height=bar_width)
    plt.yticks(np.arange(len(x_data)), x_data, fontsize=10)
    plt.xticks(fontsize=10)
    #plt.xlabel('Feature importance by GBDT Classifier', fontsize=10)
    plt.text(0.0, 16.2, '(d)', fontsize=12)    # SM
    #plt.text(0.40, 0.1, '(d)', fontsize=8)    # BF


    plt.savefig('C:/Users/lzlfly/Desktop/map/importance.jpg', dpi=1500)
    #plt.show()


def importance_merge(df_ML):
    df_ML = df_ML[(df_ML['label_ss'] == 0) | (df_ML['label_ss'] == 1) | (df_ML['label_ss'] == 2)]
    df_ML = df_ML.reset_index(drop=True)

    X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    #X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss', 'atom_radius_dif', 'electronegativity_dif', 'Hmix_z'], axis=1)

    y = df_ML['label_ss']

    x = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    x= scaler.fit_transform(x)

    from sklearn.ensemble import RandomForestClassifier
    # SS/IM/AM-param
    model_RF = RandomForestClassifier(criterion='entropy', max_features='auto', n_estimators=73, max_depth=15,
                                      random_state=10, )
    model_RF.fit(x, y)
    importance_RF = pd.Series(model_RF.feature_importances_, index=X.columns)
    importance_RF = importance_RF.sort_values(ascending=False)
    print(dict(importance_RF))
    print('--------')

    from sklearn.tree import DecisionTreeClassifier
    # SS/IM/AM-params
    model_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=8, random_state=10)
    model_DT.fit(X, y)
    importance_DT = pd.Series(model_DT.feature_importances_, index=X.columns)
    importance_DT = importance_DT.sort_values(ascending=False)
    print(dict(importance_DT))
    print('------')

    from xgboost import XGBClassifier
    # SS/IM/AM-params
    model_XGB = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=30, random_state=10)
    model_XGB.fit(X, y)
    importance_XGB = pd.Series(model_XGB.feature_importances_, index=X.columns)
    importance_XGB = importance_XGB.sort_values(ascending=False)
    print(dict(importance_XGB))

    from sklearn.ensemble import GradientBoostingClassifier
    # SS/IM/AM-params
    model_GBDT = GradientBoostingClassifier(learning_rate=0.3, max_depth=8, n_estimators=50, random_state=10)
    model_GBDT.fit(X, y)
    importance_GBDT = pd.Series(model_GBDT.feature_importances_, index=X.columns)
    importance_GBDT = importance_GBDT.sort_values(ascending=False)
    print(dict(importance_GBDT))


    plt.figure(figsize=(9, 9))
    bar_width = 0.3
    
    x_data = [r"$\lambda$", r"$\gamma$", r"$\Omega$", r"$\delta_{VEC}$", r"$\delta{H_{mix}^{0-}}$",
              r"$\Delta{H_{mix}}$", r"$VEC$", r"$\Delta{S_{mix}}$", r"$\delta_T$", r"$\delta{H_{mix}}$",
              r"$\chi_{arg}$", r"$\alpha_2$", r"$\delta{H_{mix}^{0+}}$"]

    x_data = [r"$\lambda$", r"$\delta{H_{mix}^{0}}$", r"$\gamma$", r"$\delta{H_{mix}^{0+}}$",
              r"$\Omega$", r"$VEC$", r"$\delta{H_{mix}^{0-}}$", r"$\Delta{H_{mix}}$",
              r"$\Delta{S_{mix}}$", r"$\chi_{arg}$", r"$\delta_{VEC}$", r"$\delta_{r}", r"$\alpha_2$",
              r"$\delta_T$", r"\delta_{chi}", r"$\delta{H_{mix}}$"]

    y_data1 = list(dict(importance_XGB).values())
    y_data2 = []
    y_data3 = []
    y_data4 = []
    for i in list(dict(importance_XGB).keys()):
        y_data2.append(dict(importance_DT)[i])
        y_data3.append(dict(importance_XGB)[i])
        y_data4.append(dict(importance_GBDT)[i])

    print(y_data1)
    print(y_data2)
    print(y_data3)
    print(y_data4)

    plt.barh(y=np.arange(len(x_data)), width=y_data1, label='RF', color='r', alpha=0.8, height=bar_width)
    plt.barh(y=np.arange(len(x_data))+bar_width, width=y_data2, label='DT', color='darkgreen', alpha=0.8, height=bar_width)
    plt.barh(y=np.arange(len(x_data))+2*bar_width, width=y_data3, label='XGBoost', color='turquois', alpha=0.8, height=bar_width)  #turquoise
    plt.barh(y=np.arange(len(x_data))+3*bar_width, width=y_data4, label='GBDT', color='mediumblue', alpha=0.8, height=bar_width)

    plt.yticks(np.arange(len(x_data))+0.5*bar_width, x_data, fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel('Feature importance value', fontsize=15)
    y_num = np.arange(len(x_data))
    plt.ylim(min(y_num)-0.4, max(y_num)+1)
    plt.legend(loc='upper right', fontsize=15)
    plt.savefig('C:/Users/lzlfly/Desktop/map/impor_all.jpg', dpi=1500)


def importance_zl(df_ML):
    df_ML = df_ML[(df_ML['label_ss'] == 0) | (df_ML['label_ss'] == 1) | (df_ML['label_ss'] == 2)]
    df_ML = df_ML.reset_index(drop=True)

    X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss'], axis=1)
    #X = df_ML.drop(['name', 'full_name', 'phase', 'label', 'phase_ss', 'label_ss', 'atom_radius_dif', 'electronegativity_dif', 'Hmix_z'], axis=1)

    y = df_ML['label_ss']

    x = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler()
    x= scaler.fit_transform(x)

    from xgboost import XGBClassifier
    # SS/IM/AM-params
    model_XGB = XGBClassifier(learning_rate=0.2, max_depth=12, n_estimators=90, random_state=10)
    model_XGB.fit(X, y)
    importance_XGB = pd.Series(model_XGB.feature_importances_, index=X.columns)
    importance_XGB = importance_XGB.sort_values(ascending=False)
    print(dict(importance_XGB))

    plt.figure(figsize=(9, 9))
    bar_width = 0.3

    x_data = [r"$\lambda$", r"$\delta{H_{mix}^{0}}$", r"$\gamma$", r"$\delta{H_{mix}^{0+}}$",
              r"$\Omega$", r"$VEC$", r"$\delta{H_{mix}^{0-}}$", r"$\Delta{H_{mix}}$",
              r"$\Delta{S_{mix}}$", r"$\chi_{arg}$", r"$\delta_{VEC}$", r"$\delta_{r}", r"$\alpha_2$",
              r"$\delta_T$", r"\delta_{chi}", r"$\delta{H_{mix}}$"]

    x_data = [r"$\lambda$", r"$\gamma$", r"$\delta{H_{mix}^{0}}$", r"$\Omega$", r"$\delta{H_{mix}^{0+}}$",
              r"$VEC$", r"$\Delta{H_{mix}}$", r"$\Delta{S_{mix}}$", r"$\delta{H_{mix}^{0-}}$", r"$\chi_{arg}$",
              r"$\delta_{VEC}$", r"$\alpha_2$", r"$\delta_{r}$", r"$\delta_{chi}$", r"$\delta_T$", r"$\delta{H_{mix}}$"]

    y_data1 = list(dict(importance_XGB).values())
    plt.barh(y=np.arange(len(x_data)), width=y_data1, label='RF', color='r', alpha=0.8, height=bar_width)

    plt.yticks(np.arange(len(x_data)), x_data, fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel('特征重要性', fontsize=15)
    y_num = np.arange(len(x_data))
    plt.ylim(min(y_num) - 0.4, max(y_num) + 1)
    #plt.legend(loc='upper right', fontsize=15)
    plt.savefig('C:/Users/lzlfly/Desktop/map/impor_zl.jpg', dpi=1500)



def import_all():
    y_data1 = [0.17260036, 0.09256767, 0.087601475, 0.07889306, 0.07368347, 0.0723952, 0.06637937, 0.061088737, 0.047465038,
     0.04664154, 0.043545768, 0.041177362, 0.032325115, 0.031385977, 0.031289916, 0.020959944]
    y_data2 = [0.34143424764401414, 0.042861163353650165, 0.04565817864563158, 0.0838423292959449, 0.009764461581297517,
     0.03133314346060839, 0.03, 0.09574253633099668, 0.06594880773644406, 0.01702607599470779, 0.06227360884481308,
     0.008683867581320073, 0.030578229068808846, 0.11683669825718965, 0.03171017357653253, 0.01630647862804055]
    y_data3 = [0.17260036, 0.09256767, 0.087601475, 0.07889306, 0.07368347, 0.0723952, 0.06637937, 0.061088737, 0.047465038,
     0.04664154, 0.043545768, 0.041177362, 0.032325115, 0.031385977, 0.031289916, 0.020959944]
    y_data4 = [0.08718472403802016, 0.02790817717281054, 0.20610819738484573, 0.08212184901525985, 0.1403059316804899,
     0.07234433694757081, 0.03891417789893571, 0.058279506258004946, 0.07131050569458612, 0.035754522494044316,
     0.04878181985434855, 0.02337492951849516, 0.02216289095815026, 0.03601413609124697, 0.03386151744467759,
     0.015572777548513496]

    plt.figure(figsize=(9, 9))
    bar_width = 0.2

    x_data = [r"$\lambda$", r"$\gamma$", r"$\Omega$", r"$\delta_{VEC}$", r"$\delta{H_{mix}^{0-}}$",
              r"$\Delta{H_{mix}}$", r"$VEC$", r"$\Delta{S_{mix}}$", r"$\delta_T$", r"$\delta{H_{mix}}$",
              r"$\chi_{arg}$", r"$\alpha_2$", r"$\delta{H_{mix}^{0+}}$", r"$\delta"]

    x_data = [r"$\lambda$", r"$\delta{H_{mix}^{0}}$", r"$\gamma$", r"$\delta{H_{mix}^{0+}}$",
              r"$\Omega$", r"$VEC$", r"$\delta{H_{mix}^{0-}}$", r"$\Delta{H_{mix}}$",
              r"$\Delta{S_{mix}}$", r"$\chi_{arg}$", r"$\delta_{VEC}$", r"$\delta_{r}$", r"$\alpha_2$",
              r"$\delta_T$", r"$\delta_\chi$", r"$\delta{H_{mix}}$"]


    x_data = [r"$\lambda$", r"$\gamma$", r"$\Omega$", r"$\Delta{H_{mix}}$", r"$\delta{H_{mix}^{0}}$",
              r"$\delta{H_{mix}^{0+}}$", r"$VEC$", r"$\delta{H_{mix}^{0-}}$",
              r"$\Delta{S_{mix}}$", r"$\chi_{arg}$", r"$\delta_{VEC}$", r"$\delta_{r}$", r"$\alpha_2$",
              r"$\delta_T$", r"$\delta_\chi$", r"$\delta{H_{mix}}$"]

    plt.barh(y=np.arange(len(x_data)), width=y_data1, label='RF', color='r', alpha=0.8, height=bar_width)
    plt.barh(y=np.arange(len(x_data)) + bar_width, width=y_data2, label='DT', color='darkgreen', alpha=0.8, height=bar_width)
    plt.barh(y=np.arange(len(x_data)) + 2 * bar_width, width=y_data3, label='XGBoost', color='turquoise', alpha=0.8, height=bar_width)  # turquoise
    plt.barh(y=np.arange(len(x_data)) + 3 * bar_width, width=y_data4, label='GBDT', color='mediumblue', alpha=0.8, height=bar_width)

    plt.yticks(np.arange(len(x_data)) + 1 * bar_width, x_data, fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel('Feature importance value', fontsize=15)
    y_num = np.arange(len(x_data))
    plt.ylim(min(y_num) - 0.4, max(y_num) + 1)
    plt.legend(loc='upper right', fontsize=15)
    plt.savefig('C:/Users/lzlfly/Desktop/map/impor_all.jpg', dpi=1500)


def main():
    y_data_one_m = [0.7368, 0.6974, 0.6272, 0.7719, 0.7588, 0.7500]   # SS/IM/AM/MP

    y_data_one_s = [0.8161, 0.7874, 0.7874, 0.8391, 0.8506, 0.8161]   # SS/IM/AM   # 16 features
    y_data_one_s1 = [0.8218, 0.7471, 0.7931, 0.8333, 0.8506, 0.8333]
    y_data_one_s2 = [0.8218, 0.8046, 0.7586, 0.8448, 0.8276, 0.8046]

    y_data_two_m = [0.8286, 0.7714, 0.8000, 0.8476, 0.8286, 0.8476]   # BCC/FCC/DP
    y_data_two_s = [0.9425, 0.8966, 0.9195, 0.9425, 0.9540, 0.9540]   # BCC/FCC

    x_data = ['KNN', 'SVM', 'DT', 'RF', 'XGBoost', 'GBDT']

    plt.figure(figsize=(6, 6))

    #plt.subplot(1, 2, 1)
    bar_width = 0.25
    #plt.bar(x=range(len(x_data)), height=y_data_one_m, label='SS/IM/AM/SS+IM-Classification', width=bar_width, color='c')
    plt.bar(x=np.arange(len(x_data)), height=y_data_one_s, label='original 16 features', width=bar_width, color='m')
    plt.bar(x=np.arange(len(x_data))+bar_width, height=y_data_one_s1, label='selected 13 features', width=bar_width, color='lightseagreen')
    plt.axis([-0.5, 5.5, 0.6, 0.90])
    plt.xticks(np.arange(len(x_data))+bar_width/2, x_data, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Model', fontsize=13)
    plt.ylabel('Accuracy in Validation Set', fontsize=13)
    #plt.text(0.0005, 0.98, '(a)', fontsize=13)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig('C:/Users/lzlfly/Desktop/map/acc.jpg', dpi=1500)

    

def main_zl():
    y_data_one_m = [0.7368, 0.6974, 0.6272, 0.7719, 0.7588, 0.7500]  # SS/IM/AM/MP

    y_data_one_s = [0.8161, 0.7874, 0.7874, 0.8391, 0.8506]  # SS/IM/AM   # 16 features
    y_data_one_s1 = [0.8218, 0.7471, 0.7931, 0.8333, 0.8506]
    y_data_one_s2 = [0.8218, 0.8046, 0.7586, 0.8448, 0.8276, 0.8046]

    y_data_two_m = [0.8286, 0.7714, 0.8000, 0.8476, 0.8286, 0.8476]  # BCC/FCC/DP
    y_data_two_s = [0.9425, 0.8966, 0.9195, 0.9425, 0.9540, 0.9540]  # BCC/FCC

    x_data = ['KNN', 'SVM', 'DT', 'RF', 'XGBoost']

    plt.figure(figsize=(6, 6))

    # plt.subplot(1, 2, 1)
    bar_width = 0.25
    # plt.bar(x=range(len(x_data)), height=y_data_one_m, label='SS/IM/AM/SS+IM-Classification', width=bar_width, color='c')
    plt.bar(x=np.arange(len(x_data)), height=y_data_one_s, label='初始16个特征训练', width=bar_width, color='m')
    #plt.bar(x=np.arange(len(x_data)) + bar_width, height=y_data_one_s1, label='选择13个特征训练', width=bar_width,
            #color='lightseagreen')
    plt.axis([-0.5, 4.5, 0.6, 0.90])
    plt.xticks(np.arange(len(x_data)), x_data, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('机器学习模型', fontsize=13)
    plt.ylabel('在验证集上的准确率', fontsize=13)
    # plt.text(0.0005, 0.98, '(a)', fontsize=13)
    #plt.legend(loc='upper right', fontsize=12)
    plt.savefig('C:/Users/lzlfly/Desktop/map/acc_zl.jpg', dpi=1500)


def main_paper():
    y_data_1 = [0.7368, 0.6974, 0.7675, 0.7412, 0.7478]
    y_data_2 = [0.7544, 0.6930, 0.7807, 0.7807, 0.7522]

    x_data = ['KNN', 'SVM', 'RF', 'XGBoost', 'FCNN']

    plt.figure(figsize=(6, 6))
    bar_width = 0.25
    plt.bar(x=range(len(x_data)), height=y_data_1, label='初始16个特征训练', width=bar_width, color='m')
    plt.bar(x=np.arange(len(x_data)) + bar_width, height=y_data_2, label='选择13个特征训练', width=bar_width, color='lightseagreen')

    plt.axis([-0.5, 5, 0.5, 0.85])
    plt.xticks(np.arange(len(x_data)) + 0.5*bar_width, x_data, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('机器学习模型', fontsize=14)
    plt.ylabel('在验证集上的准确率', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)

    plt.savefig('C:/Users/lzlfly/Desktop/map/acc.jpg', dpi=1500)



def pca_relationship():
    y_data_knn = [0.6031, 0.5992, 0.584, 0.6489, 0.6832, 0.645,	0.6527,	0.6489,	0.7061,	0.7252,	0.7328,	0.6985,	0.7328,	0.7214,	0.7252]
    y_data_knn = [0.5570, ]

    y_data_svm = [0.5611, 0.5573, 0.542, 0.6069, 0.6069, 0.6183, 0.6069, 0.6336, 0.6298, 0.645,	0.6641,	0.7099,	0.6947,	0.7176,	0.7137]
    y_data_svm = [0.5570, ]

    y_data_DT = [0.5496, 0.6069, 0.5649, 0.6107, 0.5954, 0.6489, 0.6412, 0.6031, 0.6336, 0.6374, 0.6565, 0.6527, 0.6794, 0.6374, 0.6832]
    y_data_DT = [0.5439, ]

    y_data_RF = [0.5614, 0.5833, 0.614,	0.693, 0.6667, 0.6491, 0.6535, 0.6842, 0.7149, 0.7105, 0.7105, 0.7368, 0.7061, 0.7237, 0.7325]
    y_data_RF = [0.5965, ]

    y_data_XGB = [0.5992, 0.5687, 0.6183, 0.6679, 0.6832, 0.6565, 0.6679, 0.6832, 0.6794, 0.7023, 0.6908, 0.6947, 0.7137, 0.7099, 0.729]
    y_data_XGB = [0.7126, 0.7069, 0.7011, 0.7184, 0.7701, 0.7529, 0.8218, 0.8161, 0.8046, 0.8103, 0.8161, 0.8218, 0.8276, 0.8103, 0.8678]  # SS/IM/AM

    y_data_GBDT = [0.5687, 0.5573, 0.5458, 0.6679, 0.6603, 0.6527, 0.6374, 0.6527, 0.6565, 0.6832, 0.6947, 0.7023, 0.7023, 0.6908, 0.7137]
    y_data_GBDT = []

    x_data = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    #f_knn = np.polyfit(x_data, y_data_knn, 3)
    #f_svm = np.polyfit(x_data, y_data_svm, 3)
    #f_DT = np.polyfit(x_data, y_data_DT, 3)
    #f_RF = np.polyfit(x_data, y_data_RF, 3)
    f_XGB = np.polyfit(x_data, y_data_XGB, 3)
    #f_GBDT = np.polyfit(x_data, y_data_GBDT, 3)

    #y_knn = np.polyval(f_knn, x_data)
    #y_svm = np.polyval(f_svm, x_data)
    #y_DT = np.polyval(f_DT, x_data)
    #y_RF = np.polyval(f_RF, x_data)
    y_XGB = np.polyval(f_XGB, x_data)
    #y_GBDT = np.polyval(f_GBDT, x_data)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()
    #plt.plot(x_data, y_knn, color='r', label='KNN')
    #plt.plot(x_data, y_svm, color='g', label='SVM')
    #plt.plot(x_data, y_DT, color='b', label='DT')
    #plt.plot(x_data, y_RF)  #color='orange', label='RF'
    plt.plot(x_data, y_XGB, color='dodgerblue', label='XGBoost', linewidth=2)
    #plt.plot(x_data, y_GBDT, color='purple', label='GBDT')

    #plt.scatter(x_data, y_data_knn, color='r', marker='x', s=5)
    #plt.scatter(x_data, y_data_svm, color='g', marker='o', s=5)
    #plt.scatter(x_data, y_data_DT, color='b', marker='s', s=5)
    #plt.scatter(x_data, y_data_RF, s=3)  #color='orange', marker='d', s=5
    plt.scatter(x_data, y_data_XGB, color='dodgerblue', marker='v', s=15)
    #plt.scatter(x_data, y_data_GBDT, color='purple', marker='p', s=5)

    plt.text(6, 0.715, 'stage 1', fontsize=13)
    plt.text(8, 0.765, 'stage 2', fontsize=13)
    plt.text(5, 0.820, 'stage 3', fontsize=13)

    #plt.legend(loc='upper left')
    plt.ylabel('Accuracy in Validation Set', fontsize=15)
    plt.xlabel('Principal Components', fontsize=15)
    plt.xticks(np.arange(0, 18, 2), fontsize=13)
    plt.yticks(fontsize=13)

    plt.savefig('C:/Users/lzlfly/Desktop/map/5/pca-XGB.jpg', dpi=1500)
    #plt.show()


def roc_bar():
    IM_AM = [0.97, 0.96, 0.91, 0.98, 0.98, 0.98]
    SS_AM = [0.98, 0.99, 0.95, 0.99, 0.99, 0.99]
    SS_IM = [0.89, 0.89, 0.82, 0.92, 0.91, 0.90]

    x_data = ['KNN', 'SVM', 'DT', 'RF', 'XGBoost', 'GBDT']

    plt.figure(figsize=(5, 5))
    bar_width = 0.25

    plt.bar(x=range(len(x_data)), height=IM_AM, label='IM/AM-Classification', width=bar_width, color='turquoise')
    plt.bar(x=np.arange(len(x_data)) + bar_width, height=SS_AM, label='SS/AM-Classification', width=bar_width,
            color='r')
    plt.bar(x=np.arange(len(x_data)) + 2 * bar_width, height=SS_IM, label='SS/IM-Classification', width=bar_width,
            color='g')
    plt.axis([-0.5, 6, 0.75, 1.04])
    plt.xticks(np.arange(len(x_data)) + bar_width, x_data, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('ML Model', fontsize=11)
    plt.ylabel('AUC Value', fontsize=11)
    plt.text(-0.4, 1.03, '(d)', fontsize=11)
    plt.legend(loc='upper right', fontsize=9)

    plt.savefig('C:/Users/lzlfly/Desktop/map/roc_bar.jpg', dpi=1500)



def abc():
    d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    for i in list(d.keys()):
        print(d[i])


if __name__ == '__main__':
    data_analyze()




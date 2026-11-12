# -*- coding: utf-8 -*-
"""
Data preparation
Spatial Machine Learning - Indonesia
------------------------------
version: March 2022
Updates: New dataset that included 2021 hh information (March 2022)
         I added spatial datasets based on GIS .shp files (Jan 2022)
         I added outlier treatment (winsorizing), when needed (Nov 2021)
         I removed perfect collinear variables (Nov 2021)
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
# ------------------------------
import pandas as pd
from scipy.stats.mstats import winsorize
import seaborn as sns
import numpy as np
# data source route
df_source = 'Z:/UpWork/UpWork2022/spatialML_march2022/db/'
# ---------------------------------------------------------------------------
# Loading survey data:
df = pd.read_stata(df_source + 'susenas1621pooled.dta')
df.count()
# 1512887
# ---------------------------------------------------------------------------
# %% Data preparation
    # id of kabupaten and per-capita expenditure pcexp
    # In the new dataset: id_kabu == district 
    # kapita_ori: Total pengeluaran per kapita sebulan (total expenditure per capita a month)
    # kapita_adjust: Total pengeluaran per kapita sebulan - adjusted
# Renaming columns:
df.rename(columns={"district": "id_kabu"},inplace=True)
# Dependent variable:
df['lrpcexp'] = np.log(df['kapita_ori'])
# Note: kapita_adjust does not have data of 2021

#%% Dropping variables which should not be explanatory variables:
# urut: household id (Nomor urut rumah tangga)
# r101: pronvice name
# r102: district id
# r105: Status wilayah: Kota/Desa = Area status: City/Village, redundant, it is already coded as urban 
df.drop(columns = ['r101','r102','r105'], inplace = True)
# Keeping only one target (depedent) variable: lrpcexp, and removing the others:
df.drop(columns = ['kapita_ori','kapita_adjust'], inplace = True)

#%% Some descriptive data analysis:
# Dependent variable:
sns.histplot(data=df, x="lrpcexp", hue="year")

# Potential explanatory variables:
# Continuous covariates:
sns.boxplot(y="h_pc_floor", x='year',data=df)       # With outliers (upper tail)
sns.boxplot(y="ikg", x='year',data=df)              # With outliers (upper tail)
sns.boxplot(y="pelayanan_dasar", x='year',data=df)  # With outliers (upper tail)
sns.boxplot(y="infrastruktur", x='year',data=df) 
sns.boxplot(y="aksesbilitas", x='year',data=df)     # With outliers (upper tail)

# Discrete (integer) covariates with outliers (upper tail)
sns.boxplot(y='h_hhcount', x='year',data=df)
sns.boxplot(y='h_n_bekerja', x='year',data=df)
sns.boxplot(y='h_n_pt', x='year',data=df)
sns.boxplot(y='h_n_sma', x='year',data=df)
sns.boxplot(y='h_n_smp', x='year',data=df)
sns.boxplot(y='h_n_tidakbekerja', x='year',data=df)
sns.boxplot(y='h_n_tidaksekolah', x='year',data=df)
sns.boxplot(y='h_nage04', x='year',data=df)
sns.boxplot(y='h_nage0519', x='year',data=df)
sns.boxplot(y='h_nage2064', x='year',data=df)
sns.boxplot(y='h_nage65up', x='year',data=df)
sns.boxplot(y='h_nfemale', x='year',data=df)
sns.boxplot(y='h_nmale', x='year',data=df)
sns.boxplot(y='h_hhcount2', x='year',data=df)

# Binary variables:
# All v* 
sns.boxplot(y="h_hhmarried", x='year',data=df)         
sns.boxplot(y="h_dwater1", x='year',data=df)
sns.boxplot(y="h_dwater2", x='year',data=df)
sns.boxplot(y="h_dwater3", x='year',data=df)
sns.boxplot(y="h_dwater4", x='year',data=df)
sns.boxplot(y="h_dwater5", x='year',data=df)       
sns.boxplot(y="h_house1", x='year',data=df)
sns.boxplot(y="h_house2", x='year',data=df)
sns.boxplot(y="h_house3", x='year',data=df)
# sns.boxplot(y="h_house4", x='year',data=df)
sns.boxplot(y="h_aset", x='year',data=df)
sns.boxplot(y="h_floor1", x='year',data=df)
sns.boxplot(y="h_floor2", x='year',data=df)
sns.boxplot(y="h_floor3", x='year',data=df)
sns.boxplot(y="h_floor4", x='year',data=df)
sns.boxplot(y="h_wall1", x='year',data=df)
sns.boxplot(y="h_wall2", x='year',data=df)
sns.boxplot(y="h_wall3", x='year',data=df)
sns.boxplot(y="h_wall4", x='year',data=df)
sns.boxplot(y="h_roof1", x='year',data=df)
sns.boxplot(y="h_roof2", x='year',data=df)
sns.boxplot(y="h_roof3", x='year',data=df)
sns.boxplot(y="h_roof4", x='year',data=df)
sns.boxplot(y="h_lighting1", x='year',data=df)
sns.boxplot(y="h_lighting2", x='year',data=df)
sns.boxplot(y="h_lighting3", x='year',data=df)
sns.boxplot(y="h_cookingfuel1", x='year',data=df)
sns.boxplot(y="h_cookingfuel2", x='year',data=df)
sns.boxplot(y="h_toilet", x='year',data=df)
sns.boxplot(y="h_aset", x='year',data=df)
sns.boxplot(y="h_house1", x='year',data=df)
sns.boxplot(y="h_house2", x='year',data=df)
sns.boxplot(y="h_house3", x='year',data=df)
sns.boxplot(y="h_house4", x='year',data=df)


#%% Treatment of outliers: Winsorizing in the lower tail, upper tail or both
# discrete and continuous variables

# Winsorizing in the lower and upper tail:
df['lrpcexp'] = winsorize(df['lrpcexp'], limits=[0.001, 0.001])
sns.histplot(data=df, x="lrpcexp", hue="year")

# Winsorizing in the upper tail only:
df['aksesbilitas'] = winsorize(df['aksesbilitas'], limits=[0, 0.05])
df['h_pc_floor'] = winsorize(df['h_pc_floor'], limits=[0, 0.05])
df['ikg'] = winsorize(df['ikg'], limits=[0, 0.05])
df['infrastruktur'] = winsorize(df['infrastruktur'], limits=[0, 0.05])
df['pelayanan_dasar'] = winsorize(df['pelayanan_dasar'], limits=[0, 0.05])

df['h_hhcount'] = winsorize(df['h_hhcount'], limits=[0, 0.05])
df['h_n_bekerja'] = winsorize(df['h_n_bekerja'], limits=[0, 0.05])
df['h_n_pt'] = winsorize(df['h_n_pt'], limits=[0, 0.05])
df['h_n_sma'] = winsorize(df['h_n_sma'], limits=[0, 0.05])
df['h_n_smp'] = winsorize(df['h_n_smp'], limits=[0, 0.05])
df['h_n_tidakbekerja'] = winsorize(df['h_n_tidakbekerja'], limits=[0, 0.05])
df['h_n_tidaksekolah'] = winsorize(df['h_n_tidaksekolah'], limits=[0, 0.05])
df['h_nage04'] = winsorize(df['h_nage04'], limits=[0, 0.05])
df['h_nage0519'] = winsorize(df['h_nage0519'], limits=[0, 0.05])
df['h_nage2064'] = winsorize(df['h_nage2064'], limits=[0, 0.05])
df['h_nage65up'] = winsorize(df['h_nage65up'], limits=[0, 0.05])
df['h_nfemale'] = winsorize(df['h_nfemale'], limits=[0, 0.05])
df['h_nmale'] = winsorize(df['h_nmale'], limits=[0, 0.05])
df['h_hhcount2'] = winsorize(df['h_hhcount2'], limits=[0, 0.05])

#%% Binarization (recoding) of binomial variables:
df.h_healthpcfloor.value_counts()
df['h_healthpcfloor'] = np.where(df['h_healthpcfloor'] == "8+", 1, 0)
df.h_healthpcfloor.value_counts()

df.h_hhmarried.value_counts()
df['h_hhmarried'] = np.where(df['h_hhmarried'] == "Ya", 1, 0)
df.h_hhmarried.value_counts()

df['h_aset'] = np.where(df['h_aset'] == 'Ya', 1, 0)
df['h_cookingfuel1'] = np.where(df['h_cookingfuel1'] == 'Ya', 1, 0)
df['h_cookingfuel2'] = np.where(df['h_cookingfuel2'] == 'Ya', 1, 0)
df['h_dwater1'] = np.where(df['h_dwater1'] == 'Ya', 1, 0)
df['h_dwater2'] = np.where(df['h_dwater2'] == 'Ya', 1, 0)
df['h_dwater3'] = np.where(df['h_dwater3'] == 'Ya', 1, 0)
df['h_dwater4'] = np.where(df['h_dwater4'] == 'Ya', 1, 0)
df['h_dwater5'] = np.where(df['h_dwater5'] == 'Ya', 1, 0)
df['h_floor1'] = np.where(df['h_floor1'] == 'Ya', 1, 0)
df['h_floor2'] = np.where(df['h_floor2'] == 'Ya', 1, 0)
df['h_floor3'] = np.where(df['h_floor3'] == 'Ya', 1, 0)
df['h_floor4'] = np.where(df['h_floor4'] == 'Ya', 1, 0)
df['h_house1'] = np.where(df['h_house1'] == 'Ya', 1, 0)
df['h_house2'] = np.where(df['h_house2'] == 'Ya', 1, 0)
df['h_house3'] = np.where(df['h_house3'] == 'Ya', 1, 0)
df['h_house4'] = np.where(df['h_house4'] == 'Ya', 1, 0)
df['h_lighting1'] = np.where(df['h_lighting1'] == 'Ya', 1, 0)
df['h_lighting2'] = np.where(df['h_lighting2'] == 'Ya', 1, 0)
df['h_lighting3'] = np.where(df['h_lighting3'] == 'Ya', 1, 0)
df['h_roof1'] = np.where(df['h_roof1'] == 'Ya', 1, 0)
df['h_roof2'] = np.where(df['h_roof2'] == 'Ya', 1, 0)
df['h_roof3'] = np.where(df['h_roof3'] == 'Ya', 1, 0)
df['h_roof4'] = np.where(df['h_roof4'] == 'Ya', 1, 0)
df['h_toilet'] = np.where(df['h_toilet'] == 'Ya', 1, 0)
df['h_wall1'] = np.where(df['h_wall1'] == 'Ya', 1, 0)
df['h_wall2'] = np.where(df['h_wall2'] == 'Ya', 1, 0)
df['h_wall3'] = np.where(df['h_wall3'] == 'Ya', 1, 0)
df['h_wall4'] = np.where(df['h_wall4'] == 'Ya', 1, 0)


#%% Filling na for 2021 
'''
# Filling based on average by kabu:
df.isna().any()[lambda x: x]
df.groupby(df.year)['ikg'].mean()
for kabu in np.unique(df.id_kabu):
    df.loc[(df.id_kabu==kabu) & (df.year==2021),'ikg'] = df.ikg[df.id_kabu == kabu].mean()
    df.loc[(df.id_kabu==kabu) & (df.year==2021),'pelayanan_dasar'] = df.pelayanan_dasar[df.id_kabu == kabu].mean()
    df.loc[(df.id_kabu==kabu) & (df.year==2021),'infrastruktur'] = df.infrastruktur[df.id_kabu == kabu].mean()
    df.loc[(df.id_kabu==kabu) & (df.year==2021),'aksesbilitas'] = df.aksesbilitas[df.id_kabu == kabu].mean()
    print('Imputation for kabupaten: ',kabu) 
df.groupby(df.year)['ikg'].mean()
df.groupby(df.year)['pelayanan_dasar'].mean()
df.groupby(df.year)['infrastruktur'].mean()
df.groupby(df.year)['aksesbilitas'].mean()
df.isna().any()[lambda x: x]
df.ikg.fillna(df.ikg.mean(),inplace=True)
df.pelayanan_dasar.fillna(df.pelayanan_dasar.mean(),inplace=True)
df.infrastruktur.fillna(df.infrastruktur.mean(),inplace=True)
df.aksesbilitas.fillna(df.aksesbilitas.mean(),inplace=True)
df.isna().any()[lambda x: x]

df = df.astype({"urut": 'str'}) # Changing urut (hhid) type from float64 to string

# Binary imputation for 2021:
df.groupby(df.year)['v1104_1'].sum()
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_1 == 1),'urut'])
df.v1104_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
df.groupby(df.year)['v1104_1'].sum()

mask = np.unique(df.loc[(df.year==2020) & (df.v1101_0== 1),'urut']); df.v1101_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1101_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1101_1== 1),'urut']); df.v1101_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1101_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1101_2== 1),'urut']); df.v1101_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1101_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1101_3== 1),'urut']); df.v1101_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1101_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1101_4== 1),'urut']); df.v1101_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1101_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1101_5== 1),'urut']); df.v1101_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1101_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1102_0== 1),'urut']); df.v1102_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1102_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1102_1== 1),'urut']); df.v1102_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1102_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1102_2== 1),'urut']); df.v1102_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1102_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1102_3== 1),'urut']); df.v1102_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1102_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1102_4== 1),'urut']); df.v1102_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1102_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1102_5== 1),'urut']); df.v1102_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1102_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1103_0== 1),'urut']); df.v1103_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1103_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1103_1== 1),'urut']); df.v1103_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1103_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1103_2== 1),'urut']); df.v1103_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1103_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1103_3== 1),'urut']); df.v1103_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1103_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1103_4== 1),'urut']); df.v1103_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1103_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1103_5== 1),'urut']); df.v1103_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1103_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_0== 1),'urut']); df.v1104_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_1== 1),'urut']); df.v1104_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_2== 1),'urut']); df.v1104_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_3== 1),'urut']); df.v1104_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_4== 1),'urut']); df.v1104_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1104_5== 1),'urut']); df.v1104_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1104_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1201_0== 1),'urut']); df.v1201_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1201_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1201_1== 1),'urut']); df.v1201_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1201_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1201_2== 1),'urut']); df.v1201_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1201_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1201_3== 1),'urut']); df.v1201_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1201_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1201_4== 1),'urut']); df.v1201_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1201_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1201_5== 1),'urut']); df.v1201_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1201_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1202_0== 1),'urut']); df.v1202_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1202_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1202_1== 1),'urut']); df.v1202_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1202_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1202_2== 1),'urut']); df.v1202_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1202_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1202_3== 1),'urut']); df.v1202_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1202_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1202_4== 1),'urut']); df.v1202_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1202_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1202_5== 1),'urut']); df.v1202_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1202_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1203_0== 1),'urut']); df.v1203_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1203_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1203_1== 1),'urut']); df.v1203_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1203_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1203_2== 1),'urut']); df.v1203_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1203_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1203_3== 1),'urut']); df.v1203_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1203_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1203_4== 1),'urut']); df.v1203_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1203_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1203_5== 1),'urut']); df.v1203_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1203_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1204_0== 1),'urut']); df.v1204_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1204_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1204_1== 1),'urut']); df.v1204_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1204_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1204_2== 1),'urut']); df.v1204_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1204_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1204_3== 1),'urut']); df.v1204_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1204_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1204_4== 1),'urut']); df.v1204_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1204_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1204_5== 1),'urut']); df.v1204_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1204_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1205_0== 1),'urut']); df.v1205_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1205_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1205_1== 1),'urut']); df.v1205_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1205_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1205_2== 1),'urut']); df.v1205_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1205_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1205_3== 1),'urut']); df.v1205_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1205_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1205_4== 1),'urut']); df.v1205_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1205_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1205_5== 1),'urut']); df.v1205_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1205_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1206_0== 1),'urut']); df.v1206_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1206_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1206_1== 1),'urut']); df.v1206_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1206_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1206_2== 1),'urut']); df.v1206_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1206_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1206_3== 1),'urut']); df.v1206_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1206_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1206_4== 1),'urut']); df.v1206_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1206_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1206_5== 1),'urut']); df.v1206_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1206_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1207_0== 1),'urut']); df.v1207_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1207_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1207_1== 1),'urut']); df.v1207_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1207_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1207_2== 1),'urut']); df.v1207_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1207_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1207_3== 1),'urut']); df.v1207_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1207_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1207_4== 1),'urut']); df.v1207_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1207_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1207_5== 1),'urut']); df.v1207_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1207_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1208_0== 1),'urut']); df.v1208_0[(df.urut.isin(mask))&(df.year==2021)] = df.v1208_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1208_1== 1),'urut']); df.v1208_1[(df.urut.isin(mask))&(df.year==2021)] = df.v1208_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1208_2== 1),'urut']); df.v1208_2[(df.urut.isin(mask))&(df.year==2021)] = df.v1208_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1208_3== 1),'urut']); df.v1208_3[(df.urut.isin(mask))&(df.year==2021)] = df.v1208_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1208_4== 1),'urut']); df.v1208_4[(df.urut.isin(mask))&(df.year==2021)] = df.v1208_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v1208_5== 1),'urut']); df.v1208_5[(df.urut.isin(mask))&(df.year==2021)] = df.v1208_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2101_0== 1),'urut']); df.v2101_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2101_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2101_1== 1),'urut']); df.v2101_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2101_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2101_2== 1),'urut']); df.v2101_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2101_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2101_3== 1),'urut']); df.v2101_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2101_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2101_4== 1),'urut']); df.v2101_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2101_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2101_5== 1),'urut']); df.v2101_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2101_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2102_0== 1),'urut']); df.v2102_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2102_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2102_1== 1),'urut']); df.v2102_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2102_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2102_2== 1),'urut']); df.v2102_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2102_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2102_3== 1),'urut']); df.v2102_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2102_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2102_4== 1),'urut']); df.v2102_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2102_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2102_5== 1),'urut']); df.v2102_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2102_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2103_0== 1),'urut']); df.v2103_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2103_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2103_1== 1),'urut']); df.v2103_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2103_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2103_2== 1),'urut']); df.v2103_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2103_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2103_3== 1),'urut']); df.v2103_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2103_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2103_4== 1),'urut']); df.v2103_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2103_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2103_5== 1),'urut']); df.v2103_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2103_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2104_0== 1),'urut']); df.v2104_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2104_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2104_1== 1),'urut']); df.v2104_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2104_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2104_2== 1),'urut']); df.v2104_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2104_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2104_3== 1),'urut']); df.v2104_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2104_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2104_4== 1),'urut']); df.v2104_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2104_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2104_5== 1),'urut']); df.v2104_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2104_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2106_0== 1),'urut']); df.v2106_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2106_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2106_1== 1),'urut']); df.v2106_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2106_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2106_2== 1),'urut']); df.v2106_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2106_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2106_3== 1),'urut']); df.v2106_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2106_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2106_4== 1),'urut']); df.v2106_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2106_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2106_5== 1),'urut']); df.v2106_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2106_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2201_0== 1),'urut']); df.v2201_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2201_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2201_1== 1),'urut']); df.v2201_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2201_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2201_2== 1),'urut']); df.v2201_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2201_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2201_3== 1),'urut']); df.v2201_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2201_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2201_4== 1),'urut']); df.v2201_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2201_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2201_5== 1),'urut']); df.v2201_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2201_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2202_0== 1),'urut']); df.v2202_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2202_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2202_1== 1),'urut']); df.v2202_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2202_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2202_2== 1),'urut']); df.v2202_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2202_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2202_3== 1),'urut']); df.v2202_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2202_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2202_4== 1),'urut']); df.v2202_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2202_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2202_5== 1),'urut']); df.v2202_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2202_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2203_0== 1),'urut']); df.v2203_0[(df.urut.isin(mask))&(df.year==2021)] = df.v2203_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2203_1== 1),'urut']); df.v2203_1[(df.urut.isin(mask))&(df.year==2021)] = df.v2203_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2203_2== 1),'urut']); df.v2203_2[(df.urut.isin(mask))&(df.year==2021)] = df.v2203_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2203_3== 1),'urut']); df.v2203_3[(df.urut.isin(mask))&(df.year==2021)] = df.v2203_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2203_4== 1),'urut']); df.v2203_4[(df.urut.isin(mask))&(df.year==2021)] = df.v2203_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v2203_5== 1),'urut']); df.v2203_5[(df.urut.isin(mask))&(df.year==2021)] = df.v2203_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3101_0== 1),'urut']); df.v3101_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3101_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3101_1== 1),'urut']); df.v3101_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3101_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3101_2== 1),'urut']); df.v3101_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3101_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3101_3== 1),'urut']); df.v3101_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3101_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3101_4== 1),'urut']); df.v3101_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3101_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3101_5== 1),'urut']); df.v3101_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3101_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3102_0== 1),'urut']); df.v3102_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3102_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3102_1== 1),'urut']); df.v3102_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3102_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3102_2== 1),'urut']); df.v3102_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3102_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3102_3== 1),'urut']); df.v3102_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3102_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3102_4== 1),'urut']); df.v3102_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3102_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3102_5== 1),'urut']); df.v3102_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3102_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3103_0== 1),'urut']); df.v3103_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3103_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3103_1== 1),'urut']); df.v3103_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3103_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3103_2== 1),'urut']); df.v3103_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3103_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3103_3== 1),'urut']); df.v3103_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3103_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3103_4== 1),'urut']); df.v3103_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3103_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3103_5== 1),'urut']); df.v3103_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3103_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3104_0== 1),'urut']); df.v3104_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3104_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3104_1== 1),'urut']); df.v3104_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3104_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3104_2== 1),'urut']); df.v3104_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3104_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3104_3== 1),'urut']); df.v3104_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3104_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3104_4== 1),'urut']); df.v3104_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3104_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3104_5== 1),'urut']); df.v3104_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3104_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3201_0== 1),'urut']); df.v3201_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3201_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3201_1== 1),'urut']); df.v3201_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3201_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3201_2== 1),'urut']); df.v3201_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3201_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3201_3== 1),'urut']); df.v3201_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3201_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3201_4== 1),'urut']); df.v3201_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3201_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3201_5== 1),'urut']); df.v3201_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3201_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3202_0== 1),'urut']); df.v3202_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3202_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3202_1== 1),'urut']); df.v3202_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3202_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3202_2== 1),'urut']); df.v3202_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3202_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3202_3== 1),'urut']); df.v3202_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3202_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3202_4== 1),'urut']); df.v3202_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3202_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3202_5== 1),'urut']); df.v3202_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3202_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3203_0== 1),'urut']); df.v3203_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3203_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3203_1== 1),'urut']); df.v3203_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3203_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3203_2== 1),'urut']); df.v3203_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3203_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3203_3== 1),'urut']); df.v3203_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3203_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3203_4== 1),'urut']); df.v3203_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3203_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3203_5== 1),'urut']); df.v3203_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3203_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3204_0== 1),'urut']); df.v3204_0[(df.urut.isin(mask))&(df.year==2021)] = df.v3204_0[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3204_1== 1),'urut']); df.v3204_1[(df.urut.isin(mask))&(df.year==2021)] = df.v3204_1[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3204_2== 1),'urut']); df.v3204_2[(df.urut.isin(mask))&(df.year==2021)] = df.v3204_2[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3204_3== 1),'urut']); df.v3204_3[(df.urut.isin(mask))&(df.year==2021)] = df.v3204_3[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3204_4== 1),'urut']); df.v3204_4[(df.urut.isin(mask))&(df.year==2021)] = df.v3204_4[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
mask = np.unique(df.loc[(df.year==2020) & (df.v3204_5== 1),'urut']); df.v3204_5[(df.urut.isin(mask))&(df.year==2021)] = df.v3204_5[(df.urut.isin(mask))&(df.year==2021)].replace(0,1)
df.groupby(df.year)['v3204_5'].sum()

# Categorical imputation for multinomial
for cat in range(6):
    mask = np.unique(df.loc[(df.year==2020) & (df.k1101== cat),'urut']); df.k1101[(df.urut.isin(mask))&(df.year==2021)] = df.k1101[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1102== cat),'urut']); df.k1102[(df.urut.isin(mask))&(df.year==2021)] = df.k1102[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1103== cat),'urut']); df.k1103[(df.urut.isin(mask))&(df.year==2021)] = df.k1103[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1104== cat),'urut']); df.k1104[(df.urut.isin(mask))&(df.year==2021)] = df.k1104[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1201== cat),'urut']); df.k1201[(df.urut.isin(mask))&(df.year==2021)] = df.k1201[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1202== cat),'urut']); df.k1202[(df.urut.isin(mask))&(df.year==2021)] = df.k1202[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1203== cat),'urut']); df.k1203[(df.urut.isin(mask))&(df.year==2021)] = df.k1203[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1204== cat),'urut']); df.k1204[(df.urut.isin(mask))&(df.year==2021)] = df.k1204[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1205== cat),'urut']); df.k1205[(df.urut.isin(mask))&(df.year==2021)] = df.k1205[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1206== cat),'urut']); df.k1206[(df.urut.isin(mask))&(df.year==2021)] = df.k1206[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1207== cat),'urut']); df.k1207[(df.urut.isin(mask))&(df.year==2021)] = df.k1207[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k1208== cat),'urut']); df.k1208[(df.urut.isin(mask))&(df.year==2021)] = df.k1208[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2101== cat),'urut']); df.k2101[(df.urut.isin(mask))&(df.year==2021)] = df.k2101[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2102== cat),'urut']); df.k2102[(df.urut.isin(mask))&(df.year==2021)] = df.k2102[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2103== cat),'urut']); df.k2103[(df.urut.isin(mask))&(df.year==2021)] = df.k2103[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2104== cat),'urut']); df.k2104[(df.urut.isin(mask))&(df.year==2021)] = df.k2104[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2106== cat),'urut']); df.k2106[(df.urut.isin(mask))&(df.year==2021)] = df.k2106[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2201== cat),'urut']); df.k2201[(df.urut.isin(mask))&(df.year==2021)] = df.k2201[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2202== cat),'urut']); df.k2202[(df.urut.isin(mask))&(df.year==2021)] = df.k2202[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k2203== cat),'urut']); df.k2203[(df.urut.isin(mask))&(df.year==2021)] = df.k2203[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3101== cat),'urut']); df.k3101[(df.urut.isin(mask))&(df.year==2021)] = df.k3101[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3102== cat),'urut']); df.k3102[(df.urut.isin(mask))&(df.year==2021)] = df.k3102[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3103== cat),'urut']); df.k3103[(df.urut.isin(mask))&(df.year==2021)] = df.k3103[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3104== cat),'urut']); df.k3104[(df.urut.isin(mask))&(df.year==2021)] = df.k3104[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3201== cat),'urut']); df.k3201[(df.urut.isin(mask))&(df.year==2021)] = df.k3201[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3202== cat),'urut']); df.k3202[(df.urut.isin(mask))&(df.year==2021)] = df.k3202[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3203== cat),'urut']); df.k3203[(df.urut.isin(mask))&(df.year==2021)] = df.k3203[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
    mask = np.unique(df.loc[(df.year==2020) & (df.k3204== cat),'urut']); df.k3204[(df.urut.isin(mask))&(df.year==2021)] = df.k3204[(df.urut.isin(mask))&(df.year==2021)].fillna(cat)
df.groupby(df.year)['k3204'].mean()

df.drop('urut', axis = 1, inplace = True)
'''
#%% One hot encoding of multinomial variables
# k*
one_hot = pd.get_dummies(df['k1101'],prefix='k1101',drop_first=True); df.drop('k1101',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1102'],prefix='k1102',drop_first=True); df.drop('k1102',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1103'],prefix='k1103',drop_first=True); df.drop('k1103',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1104'],prefix='k1104',drop_first=True); df.drop('k1104',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1201'],prefix='k1201',drop_first=True); df.drop('k1201',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1202'],prefix='k1202',drop_first=True); df.drop('k1202',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1203'],prefix='k1203',drop_first=True); df.drop('k1203',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1204'],prefix='k1204',drop_first=True); df.drop('k1204',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1205'],prefix='k1205',drop_first=True); df.drop('k1205',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1206'],prefix='k1206',drop_first=True); df.drop('k1206',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1207'],prefix='k1207',drop_first=True); df.drop('k1207',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k1208'],prefix='k1208',drop_first=True); df.drop('k1208',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2101'],prefix='k2101',drop_first=True); df.drop('k2101',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2102'],prefix='k2102',drop_first=True); df.drop('k2102',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2103'],prefix='k2103',drop_first=True); df.drop('k2103',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2104'],prefix='k2104',drop_first=True); df.drop('k2104',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2106'],prefix='k2106',drop_first=True); df.drop('k2106',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2201'],prefix='k2201',drop_first=True); df.drop('k2201',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2202'],prefix='k2202',drop_first=True); df.drop('k2202',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k2203'],prefix='k2203',drop_first=True); df.drop('k2203',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3101'],prefix='k3101',drop_first=True); df.drop('k3101',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3102'],prefix='k3102',drop_first=True); df.drop('k3102',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3103'],prefix='k3103',drop_first=True); df.drop('k3103',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3104'],prefix='k3104',drop_first=True); df.drop('k3104',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3201'],prefix='k3201',drop_first=True); df.drop('k3201',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3202'],prefix='k3202',drop_first=True); df.drop('k3202',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3203'],prefix='k3203',drop_first=True); df.drop('k3203',axis = 1, inplace=True); df = df.join(one_hot)
one_hot = pd.get_dummies(df['k3204'],prefix='k3204',drop_first=True); df.drop('k3204',axis = 1, inplace=True); df = df.join(one_hot)
one_hot.head()
one_hot.sum(axis=0)
del one_hot

#%% Final checkings of data consistency

# Checking that all variables are numeric:
dftypes = df.dtypes
print(dftypes.value_counts())
del dftypes

# checking for missing values (should not be any at this point):
if len(df.isna().any()[lambda x: x]) > 0:
    print('Missing values are still present')
    df.isna().any()[lambda x: x]
    df.ikg.fillna(df.ikg.mean(),inplace=True)
    df.pelayanan_dasar.fillna(df.pelayanan_dasar.mean(),inplace=True)
    df.infrastruktur.fillna(df.infrastruktur.mean(),inplace=True)
    df.aksesbilitas.fillna(df.aksesbilitas.mean(),inplace=True)
    df.isna().any()[lambda x: x]
else:
    print('No missing values')
    
#%% Processed spatial data from Geoda:
ds = pd.read_csv(df_source + 'spatialdata_indonesia_Feb2022.csv')  

#%% Merging into a single database:
db = df.merge(ds, left_on='id_kabu', right_on='id_ML')
db.drop(columns = ['id_ML'], inplace = True)  

#%% Saving data:
db.to_csv(df_source + 'SML_df_March2022.csv', index = False)
print('Results exported to .csv file')
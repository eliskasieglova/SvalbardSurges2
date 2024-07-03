label = 'svalbard'
products = ['ATL06', 'ATL08']
spatial_extents = {
    'heerland': [16.65, 77.65, 18.4, 78],
    'heerlandextended': [14, 76, 19, 78],
    'svalbard': [5, 70, 40, 89],
    'south': [5, 70, 19, 78]
}
spatial_extent = spatial_extents[label]
date_range = ['2018-11-01', '2023-10-31']
rerun = True
from datetime import date
date = date.today()

classification_method = 'RF'  # 'RF' for Random Forest, 'TH' for threshold method
dy = True

data_l_threshold_lower = 0.05
data_l_threshold_upper = 0.4
data_m_threshold_upper = 0.4


import os
os.chdir('C:/Users/eliss/Documents/diplomka')
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt



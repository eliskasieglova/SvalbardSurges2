import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import read_icesat, analysis, preprocessing, plotting, download, management, classify
import rasterio as rio
import numpy as np
from shapely.ops import unary_union
from sklearn import linear_model
from pyproj import Proj
import xarray as xr
import os
from pathlib import Path
from vars import label, spatial_extent, rerun, classification_method
from glacier_names import glacier_names
import polars as pl

#for i in range(10):
#    download.downloadSvalbard()

if rerun:
    # read ICESat-2 data
    data = read_icesat.readICESat(['ATL06'])

    # count elevation change
    dh_path = Path(f'data/data/{label}_dh.csv')
    if not dh_path.is_file():
        # read csv data points as pandas dataframe
        print('reading data')
        data = pd.read_csv(f'data/data/ATL06.csv')

        # drop points with unknown elevation
        data = data.dropna(subset=['h'])

        # subset ICESat-2 data to bounding box of selected area (if not for whole of svalbard)
        if label != 'svalbard':
            icesat = analysis.subsetICESat(data, spatial_extent)
        else:
            icesat = data

        # count dh
        icesat = preprocessing.dh(icesat)

    else:
        icesat = pd.read_csv(dh_path, engine="pyarrow")

    # select glaciers from RGI inside the bounding box of selected area
    rgi = analysis.selectGlaciers(spatial_extent)

    # list glacier IDs in the selected area
    glacier_ids = management.listGlacierIDs(rgi)
    total = len(glacier_ids)

    # create glacier subsets
    clipping_unsuccessful = []
    f = 1
    for glacier_id in glacier_ids:
        print(f'{f}/{total} ({glacier_id})')
        f = f+1

        # set output path
        outpath = Path(f'data/temp/glaciers/{glacier_id}_icesat_clipped.gpkg')

        # cache
        if outpath.is_file():
            continue

        # load glacier shapefile
        glacier = management.loadGlacierShapefile(glacier_id)
        glacier_name = glacier_names[glacier_id]

        # clip
        try:
            clipped = preprocessing.clipICESat(icesat, glacier, glacier_name)
            clipped.to_file(outpath)
        except:
            clipping_unsuccessful.append(glacier_id)

    # print glaciers where clipping was not successful
    print(clipping_unsuccessful)

    # filter and normalize
    filtering_unsuccessful = []
    f = 1
    for glacier_id in glacier_ids:
        print(f'{f}/{total} ({glacier_id})')
        f = f+1

        # set output path
        outpath = Path(f'data/temp/glaciers/{glacier_id}_filtered.gpkg')

        # cache
        if outpath.is_file():
            continue

        # open dataset
        try:
            data = gpd.read_file(f'data/temp/glaciers/{glacier_id}_icesat_clipped.gpkg')
        except:
            filtering_unsuccessful.append(glacier_id)
            continue

        # filter, normalize, save
        try:
            filtered = preprocessing.filterWithRANSAC(data, glacier_id)
            normalized = preprocessing.normalize(filtered)
            normalized.to_file(outpath)
        except:
            filtering_unsuccessful.append(glacier_id)
            continue

    print(f'filtering unsuccessful: {filtering_unsuccessful}')

    # group data by hydrological years
    print('grouping by hydro years')
    icesat['date'] = [str(x) for x in icesat['date']]
    icesat['date'] = [x[:10] for x in icesat['date']]

    years = [2018, 2019, 2020, 2021, 2022, 2023]
    i = 1  # start with 2018
    tot = len(glacier_ids)
    for glacier_id in glacier_ids:
        print(f'{i}/{tot} {glacier_id}')
        i = i+1

        # open pts for glacier
        try:
            data = gpd.read_file(f'data/temp/glaciers/{glacier_id}_filtered.gpkg')
        except:
            continue

        # loop through years and create subset by hydrological year
        for year in years:
            print(year)
            preprocessing.groupByHydroYear(data, year, glacier_id)

#analysis.evaluateTrainingDataset()
#analysis.runFeatureExtraction()
analysis.countYearlyChanges()
classify.classify(classification_method)

#plotting.plotTimeline()
#plotting.plotResultsByYear()
#plotting.plotResultsSubplots()
#plotting.plotSurgesByAreaSubplots()
#plotting.plotPointsFromTrainingDataset()

management.geoJsonToJS()

#plotting.ransacOnHDH()
#plotting.surgingGlacierVisualization()
#training_data = pd.read_csv('data/data/trainingdata_histograms.csv')
#data = gpd.read_file('data/results/RFresults_svalbard_2024-06-28.gpkg')
#glacier_ids = data['glacier_id'].to_list()
#years = [2019, 2020, 2021, 2022, 2023]
#features = gpd.read_file(f'data/temp/svalbard_features.gpkg')
#for glacier_id in glacier_ids:
#    for year in years:
#        try:
#            plotting.plotStatisticalMetrics(glacier_id, year, features)
#        except:
#            continue
#plotting.trainingDataHistograms(f'bin_max', f'maximum of bin averages (lower part)', 'average maximum elevation change (m)')

# todo after the important stuff is done:
#  - create def formatInputData() where I format the merged ATL06, ATL08 datasets
#    into the format I want to have it in analysis (see DP2) and I can put it on
#    github as "this is the data I used".
#  - create def assignHydroYear() where I assign the hydrological year to the datapoints


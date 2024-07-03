import pandas as pd
import preprocessing
import geopandas as gpd
import rasterio as rio
import numpy as np
from shapely import wkt
from pathlib import Path
import time
from pyproj import Proj
from vars import label, spatial_extent, date, data_l_threshold_lower, data_l_threshold_upper, data_m_threshold_upper, classification_method, dy
from sklearn import linear_model
import management
from glacier_names import glacier_names
import math
import matplotlib.pyplot as plt
glaciers_nodata = []

def subsetICESat(df, bbox):
    """
    Create subset of data points.

    :param dfs: dataframes in list
    :param bbox: bbox of subset area in format list[east, west, north, south]

    :return: DataFrame of points within the given bounding box. Merged from all the input DataFrames.
    """

    print(f'subsetting icesat to {label}')

    outpath = Path(f'data/data/ICESat_{label}.csv')

    # cache
    if outpath.is_file():
        return pd.read_csv(outpath)

    # extract bounds from bounding box
    west = bbox[0]
    south = bbox[1]
    east = bbox[2]
    north = bbox[3]

    # create subset based on conditions
    subset = df[df['longitude'] < east]
    subset = subset[subset['longitude'] > west]
    subset = subset[subset['latitude'] < north]
    subset = subset[subset['latitude'] > south]

    # save as .csv file
    subset.to_csv(outpath)

    return subset


def selectGlaciers(bbox):
    """
    Selects glaciers from Randolph Glacier Inventory (RGI) inside given bounding box.

    :param bbox: Bounding box of subset area in format list[east, west, north, south].

    :returns: Subset of RGI.
    """

    print('subsetting rgi')

    outpath = Path(f'data/data/rgi_{label}.gpkg')

    # cache
    if outpath.is_file():
        subset = gpd.read_file(outpath, engine='pyogrio', use_arrow=True)
        return subset

    # load Randolph Glacier Inventory
    rgi = gpd.read_file('data/data/rgi.gpkg', engine='pyogrio', use_arrow=True).to_crs('EPSG:32633')

    # create subset by bbox (not used bc it included glaciers that only overlapped and were not completely within)
    #subset = rgi.cx[bbox[1]:bbox[0], bbox[3]:bbox[2]]

    # convert bbox lat lon to easting northing
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")  # assign projection
    eastings, northings = myproj((bbox[0], bbox[2]), (bbox[1], bbox[3]))

    lon0, lat0, lon1, lat1 = bbox

    pt1 = myproj(lon0, lat0)
    pt2 = myproj(lon0, lat1)
    pt3 = myproj(lon1, lat1)
    pt4 = myproj(lon1, lat0)

    # convert
    from shapely.geometry import Polygon
    polygon = Polygon([(eastings[0], northings[0]), (eastings[1], northings[0]), (eastings[1], northings[1]), (eastings[0], northings[1]), (eastings[0], northings[0])])
    polygon = Polygon([pt1, pt2, pt3, pt4, pt1])
    polygon = Polygon([pt1, pt2, pt3, pt4, pt1])
    bbox = gpd.GeoDataFrame(gpd.GeoSeries(polygon), columns=['geometry'], crs='EPSG:32633')

    # go glacier by glacier and determine if it's in bbox or not
    mask = []
    for i, row in rgi.iterrows():
        if row['geometry'].within(bbox)['geometry'][0]:
            mask.append(True)
        else:
            mask.append(False)

    # mask it out!!!
    rgi['mask'] = mask
    subset = rgi[rgi['mask'] == True]

    # if glaciers are too small just throw them out
    #subset = subset[subset['geometry'].area / 1e6 > 1]

    # save it
    subset.to_file(outpath)

    return subset

def selectCenterLines(bbox, minvalue):
    """
    Selects glaciers from Randolph Glacier Inventory (RGI) inside given bounding box.

    :param bbox: Bounding box of subset area in format list[east, west, north, south].

    :returns: Subset of RGI.
    """

    outpath = Path(f'data/data/CL_{label}.gpkg')

    # cache
    if outpath.is_file():
        subset = gpd.read_file(outpath, engine='pyogrio', use_arrow=True)
        return subset

    # load Randolph Glacier Inventory
    rgi = gpd.read_file('data/data/centerlines_svalbard.gpkg', engine='pyogrio', use_arrow=True).to_crs('EPSG:32633')

    # create subset by bbox (not used bc it included glaciers that only overlapped and were not completely within)
    #subset = rgi.cx[bbox[1]:bbox[0], bbox[3]:bbox[2]]

    # convert bbox lat lon to easting northing
    myproj = Proj("+proj=utm +zone=33 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")  # assign projection
    eastings, northings = myproj((bbox[0], bbox[2]), (bbox[1], bbox[3]))

    # convert
    from shapely.geometry import Polygon
    polygon = Polygon([(eastings[0], northings[0]), (eastings[1], northings[0]), (eastings[1], northings[1]), (eastings[0], northings[1]), (eastings[0], northings[0])])
    bbox = gpd.GeoDataFrame(gpd.GeoSeries(polygon), columns=['geometry'], crs='EPSG:32633')

    # go glacier by glacier and determine if it's in bbox or not
    mask = []
    for i, row in rgi.iterrows():
        if row['geometry'].within(bbox)['geometry'][0]:
            mask.append(True)
        else:
            mask.append(False)

    # mask it out!!!
    rgi['mask'] = mask
    subset = rgi[rgi['mask'] == True]

    # if glaciers are too small just throw them out
    subset = subset[subset['geometry'].length > minvalue]

    # save it
    subset.to_file(outpath)

    return subset


def countTimeOfReadingData():
    start = time.time()
    csv = pd.read_csv('C:/Users/eliss/Documents/diplomka/data/temp/ATL08.csv')
    csv_duration = time.time() - start

    start2 = time.time()
    gpkg = gpd.read_file('C:/Users/eliss/Documents/diplomka/data/data/ATL08.gpkg', engine='pyogrio', use_arrow=True)
    gpkg_duration = time.time() - start2

    print('ATL08 csv:')
    print(csv_duration)
    print('ATL08 gpkg:')
    print(gpkg_duration)

    start = time.time()
    csv = pd.read_csv('C:/Users/eliss/Documents/diplomka/data/temp/ATL06.csv')
    csv_duration = time.time() - start

    start2 = time.time()
    gpkg = gpd.read_file('C:/Users/eliss/Documents/diplomka/data/data/ATL06.gpkg', engine='pyogrio', use_arrow=True)
    gpkg_duration = time.time() - start2

    print('ATL06 csv:')
    print(csv_duration)
    print('ATL06 gpkg:')
    print(gpkg_duration)


def linRegAlg(data, glacier_name):

    if data.index.size == 0:
        return 'nodata'

    # reshape arrays
    X = data.h.values.reshape(-1, 1)
    y = data.dh.values.reshape(-1, 1)

    # Fit line
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)

    # plot
    # plt.scatter(X, y, color="yellowgreen", marker='.')
    # plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")

    # coefficient
    coef = lr.coef_[0][0]
    return X, y, line_X, line_y, coef


def linreg(data):

    if data.index.size == 0:
        return np.nan

    # reshape arrays
    X = data.h.values.reshape(-1, 1)
    y = data.dh.values.reshape(-1, 1)

    # Fit line
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    #line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    #line_y = lr.predict(line_X)

    coef = lr.coef_[0][0]

    #plt.scatter(X, y, color="orange", marker='.', s=2)
    #plt.plot(line_X, line_y, color="navy", linewidth=2, label="Linear regressor")
    #ax.invert_xaxis()

    return coef


def RANSAC(data, glacier_id=None, year=None, l=None, surging=None):
    """
    Takes df/gdf as input with columns h, dh.
    Returns RANSAC coefficient, outliers and inliers.
    """

    # reshape arrays
    X = data.h_norm.values.reshape(-1, 1)
    y = data.dh.values.reshape(-1, 1)

    # Robustly fit data with ransac algorithm
    ransac = linear_model.RANSACRegressor(max_trials=100)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models (draw the line)
    line_X = np.arange(0, 1, 0.05).reshape(-1, 1)
    line_y_ransac = ransac.predict(line_X)

    import matplotlib.pyplot as plt
    # PLOT
    lw = 2  # linewidth
    plt.scatter(X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
    plt.scatter(X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
    plt.plot(line_X, line_y_ransac, color="cornflowerblue", linewidth=lw, label="RANSAC regressor")
    plt.xlim(0, 1)
    plt.ylim(-100, 100)
    plt.title(f'{glacier_id} {year}')
    plt.xlabel('easting')
    plt.ylabel('h')
    plt.legend()
    plt.savefig(f'data/temp/figs/{glacier_id}_{year}{l}_ransac.png')
    plt.close()

    # ransac coefficient
    coef = ransac.estimator_.coef_[0][0]

    return coef


def extractFeatures(glacier_ids, years):
    """
    Extract features relevant for detecting surging glaciers.
    :param data:
    :return:
    """

    glacier_ids_result = []
    years_result = []

    dh_means_l = []  # mean elevation change for the lower part
    dh_maxs_l = []  # 90th percentile elevation change for the lower part

    dh_stds = []   # standard deviation of dh for the whole glacier
    dh_stds_l = []   # standard deviation of dh for the lower part
    hdh_stds = []  # standard deviation of h/dh combined for the whole glacier
    hdh_stds_l = []  # standard deviation of h/dh combined for the lower part

    lin_coefs_binned = []  # linear coefficients for binned data over the whole glacier
    lin_coefs_l_binned = []  # linear coefficients for binned data over the whole glacier

    ransacs = []
    ransacs_l = []

    correlations = []
    correlations_l = []
    variances_dh = []
    variances_dh_l = []

    pts_above_15 = []

    # bins
    bins1 = []
    bins2 = []
    bins3 = []
    bins4 = []
    bins5 = []
    bins6 = []
    bins7 = []
    bins8 = []
    bins9 = []
    bins10 = []
    bins11 = []
    bins12 = []
    bins13 = []
    bins14 = []
    bins15 = []
    bins16 = []
    bins17 = []
    bins18 = []
    residuals = []
    residuals_l = []
    bin_maxs = []

    # quality flag
    QFs = []

    # loop through glaciers
    tot = len(glacier_ids)
    i = 1
    for glacier_id in glacier_ids:
        print(f'{i}/{tot}')
        i = i + 1
        for year in years:
            print(year)
            # read points file
            try:
                if dy:
                    data = gpd.read_file(f'data/temp/glaciers/{glacier_id}_{year}_fl.gpkg', engine='pyogrio', use_arrow=True)
                else:
                    data = gpd.read_file(f'data/temp/glaciers/{glacier_id}_{year}.gpkg', engine='pyogrio', use_arrow=True)
            except:
                print('points file doesnt exist')
                continue

            if len(data) < 15:
                print('points file empty')
                continue

            # lowest part of dataset
            data_l = data[data['h_norm'] <= data_l_threshold_upper]
            if len(data_l) < 5:
                print('lower points empty')
                continue

            # split the data into elevation bins
            # append means for each bin and linreg for the binned data
            bin_limits = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            X = np.array(binData(data, bin_limits)).reshape(-1, 1)
            y = np.array([0.025, 0.075, 0.125, 0.175,  0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875]).reshape(-1, 1)

            # covariance - make 2D array - dont run bin analysis if everuthing is nan
            covariance = np.cov(y, X)
            try:
                correlations.append(covariance[0][1])
                variances_dh.append(covariance[1][1])
            except:
                print('only nans')
                break

            # append glacier id and year
            glacier_ids_result.append(glacier_id)
            years_result.append(year)

            # add quality flag
            QFs.append(QF(data))


            bins = [bins1, bins2, bins3, bins4, bins5, bins6, bins7, bins8, bins9, bins10, bins11, bins12, bins13, bins14, bins15, bins16, bins17, bins18]
            m = 0
            for bin in bins:
                val = X[m][0]
                # if value for given bin is nan, make average of bins around
                if math.isnan(val):
                    # if its the lowest bin, find nearest upper value
                    if m == 0:
                        # find nearest upper value
                        n = m
                        while math.isnan(val):
                            n = n + 1
                            if n == 18:
                                val = float('nan')
                                break
                            else:
                                val = X[n][0]

                    # else if its the last bin
                    elif m == 17:
                        n = m
                        while math.isnan(val):
                            # go one bin down
                            n = n - 1
                            if n == 0:
                                val = float('nan')
                                break
                            else:
                                val = X[n][0]
                    # else if its one of the middle bins
                    else:
                        # find nearest lower value
                        lower_val = float('nan')
                        n = m
                        while math.isnan(lower_val):
                            n = n - 1
                            # catch if out of bounds
                            if n == -1:
                                lower_val = float('nan')
                                break
                            else:
                                lower_val = X[n][0]

                        # find nearest upper value
                        upper_val = float('nan')
                        n = m
                        while math.isnan(upper_val):
                            n = n + 1
                            if n == 17:
                                upper_val = float('nan')
                                break
                            else:
                                upper_val = X[n][0]

                        # make average out of that
                        if (lower_val == float('nan')) & (upper_val == float('nan')):
                            break
                        val = (lower_val + upper_val) / 2
                        X[m] = val

                # append value
                bin.append(val)
                m = m + 1

            # plot the bins to check if its ok
            bin_avgs = [x[-1] for x in bins]
            plt.scatter(data.h_norm, data.dh, alpha=0.4, s=1, marker='.')
            plt.scatter(y, bin_avgs)
            plt.xlim(0, 0.9)
            plt.ylim(-75, 75)
            plt.xlabel('normalized elevation')
            plt.ylabel('elevation change (m)')
            plt.title(f'{glacier_names[glacier_id]} ({year})')
            plt.savefig(f'eliskasieglova.github.io/imgs/{glacier_id}_{year}.png')
            plt.savefig(f'figs/linregs_binned/{glacier_id}_{year}.png')
            plt.close()
            # now i have data split into bins that i can do analysis on (X and y)

            # linear coefficient on binned data (whole glaciers)
            try:
                mask = ~np.isnan(X).reshape(-1, 1)  # inverse mask for nan values
                X = X[mask].reshape(-1, 1)
                y = y[mask].reshape(-1, 1)
                lr = linear_model.LinearRegression()
                lr.fit(X, y)

                # append to the lists
                lin_coefs_binned.append(lr.coef_[0][0])
                residuals.append(((y - lr.predict(X)) ** 2).sum())
            except:
                lin_coefs_binned.append(float('nan'))
                residuals.append(float('nan'))

            # linear coefficients on binned data for lower part
            try:
                X_lower = X[:8]
                y_lower = y[:8]

                # plot the bins to check if its ok
                plt.scatter(data.h_norm, data.dh, alpha=0.4, s=1, marker='.')
                plt.scatter(y_lower, X_lower)
                plt.xlim(0, 5)
                plt.ylim(-75, 75)
                plt.title(f'{glacier_names[glacier_id]} ({year})')
                plt.savefig(f'figs/linregs_binned/{glacier_id}_{year}_lower.png')
                plt.close()

                mask = ~np.isnan(X_lower).reshape(-1, 1)  # inverse mask for nan values
                X_lower = X_lower[mask].reshape(-1, 1)
                y_lower = y_lower[mask].reshape(-1, 1)
                lr = linear_model.LinearRegression()
                lr.fit(X_lower, y_lower)

                lin_coefs_l_binned.append(lr.coef_[0][0])
                residuals_l.append(((y_lower - lr.predict(X_lower))** 2).sum())

            except:
                lin_coefs_l_binned.append(float('nan'))
                residuals_l.append(float('nan'))

            # filter the complete terminus... ???
            # data = data[data['h_norm'] > data_l_threshold_lower]

            # 90th percentile (lower)
            try:
                dh_maxs_l.append(np.percentile(data_l['dh'], 95))
            except:
                print('cant append dh maxs_l')
                dh_maxs_l.append(np.nan)

            # average (lower, middle)
            dh_means_l.append(data_l['dh'].mean())

            # standard deviation of dh (binned data)
            dh_stds.append(X.std())
            dh_stds_l.append(X.std())

            # standard deviation of h/dh (binned data)
            std_h = y.std()
            std_dh = X.std()
            stddev = np.sqrt(std_h * std_h + std_dh * std_dh)
            hdh_stds.append(stddev)  # the whole glacier

            std_h = y.std()
            std_dh = X.std()
            stddev_l = np.sqrt(std_h * std_h + std_dh * std_dh)
            hdh_stds_l.append(stddev_l)  # lower part

            covariance_lower = np.cov(y_lower, X_lower)
            correlations_l.append(covariance_lower[0][1])
            variances_dh_l.append(covariance_lower[1][1])

            # append maximum from bin avgs
            bin_maxs.append(max(bin_avgs[:8]))

            # RANSAC
            try:
                ransacs.append(RANSAC(data, glacier_id, year))
            except:
                print('cant do ransac')
                ransacs.append(np.nan)
            try:
                ransacs_l.append(RANSAC(data_l, glacier_id, year, '_l'))
            except:
                print('cant do ransac l')
                ransacs_l.append(np.nan)

            # points above 15m
            lowest = data_l[data_l['h_norm'] < 0.18]
            pts_above_15.append(len(lowest[lowest['dh'] > 15]))

    variables = {
        'glacier_id': glacier_ids_result,
        'year': years_result,
        'dh_max_l': dh_maxs_l,
        'dh_mean_l': dh_means_l,
        'dh_std': dh_stds,
        'dh_std_l': dh_stds_l,
        'hdh_std': hdh_stds,
        'hdh_std_l': hdh_stds_l,
        'lin_coef_binned': lin_coefs_binned,
        'lin_coef_l_binned': lin_coefs_l_binned,
        'correlation': correlations,
        'correlation_l': correlations_l,
        'variance_dh': variances_dh,
        'variance_dh_l': variances_dh_l,
        'ransac': ransacs,
        'ransac_l':ransacs_l,
        'over15': pts_above_15,
        'quality_flag': QFs,
        'bin1': bins1,
        'bin2': bins2,
        'bin3': bins3,
        'bin4': bins4,
        'bin5': bins5,
        'bin6': bins6,
        'bin7': bins7,
        'bin8': bins8,
        'bin9': bins9,
        'bin10': bins10,
        'bin11': bins11,
        'bin12': bins12,
        'bin13': bins13,
        'bin14': bins14,
        'bin15': bins15,
        'bin16': bins16,
        'bin17': bins17,
        'bin18': bins18,
        'residuals': residuals,
        'residuals_l': residuals_l,
        'bin_max': bin_maxs
    }

    for var in variables:
        print(f'{var}, {type(variables[var])}, {len(variables[var])}')

    results = pd.DataFrame.from_dict(variables)


    return results


def runFeatureExtraction():
    rgi = selectGlaciers(spatial_extent)
    # list glacier ids to loop through
    glacier_ids = management.listGlacierIDs(rgi)

    years = [2019, 2020, 2021, 2022, 2023]

    # extract features (min, max, linreg coefs etc. for each glacier)
    print('extracting features')
    glacier_features = extractFeatures(glacier_ids, years)

    print('exporting')
    rgi = rgi.rename(columns={"glims_id" : "glacier_id"})
    c = glacier_features.merge(rgi, how='left', on='glacier_id')
    #c = c.drop(columns=['geometry_x'])
    gdf = gpd.GeoDataFrame(c, geometry='geometry')
    gdf.to_file(f'data/temp/{label}_features.gpkg', geometry='geometry')

    return


def decisionTree():
    # so now i extracted features from the datasets and have them all collected in the dataset 'glacier_features'
    data = gpd.read_file(f'data/temp/{label}_features.gpkg', engine='pyogrio', use_arrow=True)

    # new geodataframe with glacier_id, geometry
    gdf = gpd.GeoDataFrame()
    gdf['glacier_id'] = list(data['glacier_id'])
    gdf['year'] = list(data['year'])
    gdf['geometry'] = list(data['geometry'])

    dh_max_lower = [1 if i > 25 else 0 for i in data['dh_max_lower']]
    lin_coef = [1 if i < 0.025 else 0 for i in data['lin_coef']]
    dh_std = [1 if i > 10 else 0 for i in data['dh_std']]

    gdf['dh_max_lower'] = dh_max_lower
    gdf['lin_coef'] = lin_coef
    gdf['dh_std'] = dh_std

    gdf['surging'] = gdf['dh_max_lower'] * gdf['lin_coef'] * gdf['dh_std']
    gdf['surging_sum'] = gdf['dh_max_lower'] + gdf['lin_coef'] + gdf['dh_std']

    gdf = gdf.set_geometry('geometry')
    gdf = gdf.set_crs(crs='EPSG:32633')

    # split by years
    y1 = gdf[gdf['year'] == 2018]
    y2 = gdf[gdf['year'] == 2019]
    y3 = gdf[gdf['year'] == 2020]
    y4 = gdf[gdf['year'] == 2021]
    y5 = gdf[gdf['year'] == 2022]
    y6 = gdf[gdf['year'] == 2023]

    y1.to_file(f'data/temp/{label}_2020_thresholds.gpkg', geometry='geometry')
    y2.to_file(f'data/temp/{label}_2021_thresholds.gpkg', geometry='geometry')
    y3.to_file(f'data/temp/{label}_2022_thresholds.gpkg', geometry='geometry')

def addQualityFlag(shortcut):
    """
    Add quality flag to the data based on how many points there are 1) for the whole glacier and
    2) for its lower part.

    Input DF/GDF with results after RF.

    Returns the same dataset with an additional column 'quality_flag'.
    """
    results = gpd.read_file(f'data/results/{shortcut}results_{label}_{date}.gpkg')

    qfs = []
    l = len(results)
    # loop through rows
    for i in range(l):
        print(f'{i}/{l}')
        glacier_id = results['glacier_id'].iloc[i]
        year = results['year'].iloc[i]

        data = gpd.read_file(f'data/temp/glaciers/{glacier_id}_{year}.gpkg')
        data = data[data['h_norm'] > data_l_threshold_lower]
        # total number of points in dataset
        number_of_points = len(data)

        # number of points in lower part
        number_of_points_lower = len(data[data['h_norm'] < data_l_threshold_upper])

        # number of points in middle part
        data_m = data[data['h_norm'] > data_l_threshold_upper]
        number_of_points_middle = len(data_m[data_m['h_norm'] < data_m_threshold_upper])

        qf = 0
        # todo: figure out the thresholds for this and add it to the data
        if number_of_points > 1000:
            qf = qf + 1

        if number_of_points_lower > 250:
            qf = qf + 1

        if number_of_points_middle > 250:
            qf = qf + 1

        qfs.append(qf)

    results['QF'] = qfs

    results.to_file(f'data/results/RFresultsQF_{date}.gpkg')

    return

def QF(data):
    """
    Counts quality flag for glacier based on the number of points available in the
    whole area, middle part and lower part.

    :param data: Pandas DataFrame of points for glacier.
    :return: Quality Flag (Int 0/1/2/3) based on num of points for glaciers.
    """

    # delete the complete bottom of the dataset
    data = data[data['h_norm'] > data_l_threshold_lower]
    # total number of points in dataset
    number_of_points = len(data)

    # number of points in lower part
    number_of_points_lower = len(data[data['h_norm'] < 0.4])

    # number of points in middle part
    data_m = data[data['h_norm'] > data_l_threshold_upper]
    number_of_points_middle = len(data_m[data_m['h_norm'] < data_m_threshold_upper])

    qf = 0
    if number_of_points > 1000:
        qf = qf + 1

    if number_of_points_lower > 500:
        qf = qf + 1

    return qf


def evaluateTrainingDataset():
    # extract features for training dataset
    datapath = Path('data/data/trainingdata.csv')
    outpath = Path('data/temp/trainingdata/trainingdata_features.gpkg')
    outpath_csv = Path('data/temp/trainingdata/trainingdata_features.csv')

    data = pd.read_csv(datapath)

    features = gpd.GeoDataFrame()
    empty_glaciers = []

    # loop through the training dataset
    for i in range(len(data)):
        # extract glacier id and year and name
        glacier_id = [data['glacier_id'].iloc[i]]
        year = [data['year'].iloc[i]]
        name = data['name'].iloc[i]
        # extract features
        glacier_features = extractFeatures(glacier_id, year)  # returns df which i hope has one row
        if len(glacier_features) == 0:
            empty_glaciers.append((glacier_id, name, year))
        else:
            glacier_features['name'] = name  # append name
            # append the extracted features to the result GDF
            features = pd.concat([features, glacier_features])

    # there are some nodata in the dataframe so let's fix that
    for i in range(len(features)):
        if type(features['lin_coef'].iloc[i]) == np.nan:
            empty_glaciers.append((features['glacier_id'].iloc[i], features['year'].iloc[i], features['name'].iloc[i]))
    features = features.dropna()

    # add geometry to the glaciers
    geoms=[]
    for i in range(len(features)):
        glacier_id = features['glacier_id'].iloc[i]
        # open shapefile data
        shp = gpd.read_file(f'data/temp/glaciers/{glacier_id}.gpkg')
        # copy geometry
        geoms.append(shp['geometry'].iloc[0])
    features['geom'] = geoms

    # append if they are surging or not
    surging = []
    for i in range(len(features)):
        glacier_id = features['glacier_id'].iloc[i]
        year = features['year'].iloc[i]
        subset = data.where(data['glacier_id'] == glacier_id).dropna()
        subset = subset.where(subset['year'] == year).dropna()
        surging.append(subset['surging'].iloc[0])
    features['surging'] = surging

    features = gpd.GeoDataFrame(features, crs=32633, geometry='geom')
    features.to_file(outpath)
    features.to_csv(outpath_csv)

    # print which glaciers did not contain any data
    print(empty_glaciers)
    print(len(empty_glaciers))
    print(len(features))

    return features


def createFeaturesTimeSeries():
    """
    not using this function yet
    :return:
    """

    data = gpd.read_file(f'data/results/{classification_method}_{label}_{date}.gpkg')

    linregs = pd.DataFrame(columns=['glacier_id', 2010, 2018, 2019, 2020, 2021, 2022])

    # split data into linreg, dh_max
    for i in range(len(data)):
        glacier_id = data['glacier_id'].iloc[i]
        year = data['year'].iloc[i]

    return


def binData(data, bin_limits):
    "split data into bins"

    avgs = []

    for i in range(len(bin_limits) - 1):
        l = bin_limits[i]
        u = bin_limits[i+1]

        subset = data[data['h_norm'] > l]
        subset = subset[subset['h_norm'] < u]

        if len(subset) < 3:
            avgs.append(np.nan)
        else:
            avgs.append(subset.dh.mean())

    return avgs

def splitFeatureByYear():
    # do the same thing as extractFeatures but do it by year
    # lin reg coefficients need to be differences from the year before
    # max dh has to be from the year before as well
    # all has to be from the year before

    # load data - extracted features
    data = gpd.read_file(f'data/temp/{label}_features.gpkg')

    # features = columns from data
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    features = ['lin_coef_binned', 'dh_max']

    # create list of column labels
    columns = ['glacier_id']
    for feature in features:
        for year in years:
            column_label = f'{feature}_{year}'
            columns.append(column_label)

    # create new df for the yearly differences
    new_df = pd.DataFrame(columns=columns)

    # list glacier ids
    glacier_ids = np.unique(data.glacier_id)

    # loop through glaciers
    for glacier_id in glacier_ids:
        for year in years:
            # select the row in df for given glacier id and year
            row = data[data['glacier_id'] == glacier_id]
            row = row[row['year'] == year]

            # loop through the features, count differences between years, assign differences to new dataframe
            for feature in features:
                column_label = f'{feature}_{year}'
                if year == 2018:
                    # assign original differences
                    continue
                else:
                    # count new differences
                    continue

    return


def countYearlyChanges():

    data = gpd.read_file(f'data/temp/{label}_features.gpkg')

    glacier_ids = np.unique(data.glacier_id)
    years = [2018, 2019, 2020, 2021, 2022, 2023]

    # features
    #features = ['glacier_id',
    #            'year', 'geometry', 'quality_flag', 'glac_name', 'dh_max_l', 'dh_max_m', 'dh_mean_l', 'dh_mean_m', 'dh_std', 'dh_std_l', 'dh_std_m', 'hdh_std', 'hdh_std_l', 'hdh_std_m', 'lin_coef', 'lin_coef_l', 'lin_coef_m', 'lin_coef_l_binned', 'lin_coef_binned', 'correlation', 'correlation_l', 'correlation_m', 'variance_dh', 'variance_dh_l', 'variance_dh_m', 'ransac', 'ransac_l', 'ransac_m', 'over15', 'over15_m']


    features = ['glacier_id',
                    'year',
                    'geometry',
                    'quality_flag',
                    'glac_name',
                    'dh_max_l',
                    #'dh_mean_l',
                    #'dh_std',
                    'dh_std_l',
                    #'hdh_std',
                    'hdh_std_l',
                    'lin_coef_binned',
                    'lin_coef_l_binned',
                    'variance_dh',
                    'variance_dh_l',
                    'correlation',
                    'correlation_l',
                    'ransac',
                    'ransac_l',
                    'over15',
                    'bin1',
                    'bin2',
                    'bin3',
                    'bin4',
                    'bin5',
                    'bin6',
                    'bin7',
                    'bin8',
                    'bin9',
                    'bin10',
                    'bin11',
                    'bin12',
                    'bin13',
                    'bin14',
                    'bin15',
                    'bin16',
                    'bin17',
                    'bin18',
                    'residuals',
                    'bin_max']


    # create new dataframe with the same columns
    new_df = pd.DataFrame(columns=features)
    t = len(glacier_ids)
    m = 0
    # loop through dataset
    for glacier_id in glacier_ids:
        print(f'{m}/{t}')
        m = m + 1
        print(glacier_id)
        for year in years:
            print(year)
            # select subset for given glacier id and year
            rows_id = data[data['glacier_id'] == glacier_id]
            row = rows_id[rows_id['year'] == year]

            # if the row is empty (does not exist - not enough points), continue
            if len(row) == 0:
                continue

            # if year is 2018 keep value and append original row to new dataset
            if year == 2018:
                # select only the values that i want
                new_list = [glacier_id, year, row['geometry'].iloc[0], row['quality_flag'].iloc[0], row['glac_name'].iloc[0]]
                for feature in features[5:]:
                    new_list.append(row[feature].iloc[0])

                # append new row to dataframe
                new_row = pd.DataFrame(columns=features, data=[new_list])
                new_df = pd.concat([new_df, new_row])

            # else count the differences between the years
            else:
                # try select row for previous year - if it doesnt exist, continue
                previous_row = rows_id[rows_id['year'] == year - 1]

                # if the previous row doesnt exist - append current values
                if len(previous_row) == 0:
                    new_list = [glacier_id, year, row['geometry'].iloc[0], row['quality_flag'].iloc[0],
                                row['glac_name'].iloc[0]]
                    for feature in features[5:]:
                        # select values
                        value = row[feature].iloc[0]
                        new_list.append(value)

                    # convert list to series
                    new_row = pd.DataFrame(columns=features, data=[new_list])

                    # append to new dataframe
                    new_df = pd.concat([new_df, new_row])

                    continue

                # create new list
                new_list = [glacier_id, year, row['geometry'].iloc[0], row['quality_flag'].iloc[0], row['glac_name'].iloc[0]]
                # loop through features
                for feature in features[5:]:
                    # select values
                    value = row[feature].iloc[0]
                    previous_value = previous_row[feature].iloc[0]

                    # if previous year is nan --> keep current value
                    if not previous_value > -999999:  # == if previous value is nan
                        # if the previous year is 2017 or 2018 then there will be no values lower --> append current value
                        previous_year = year - 1
                        if previous_year <= 2018:
                            new_list.append(value)

                        # if the previous year is 2019 and more, go lower to find out if a different year has a value
                        else:
                            i = 1
                            while previous_year > 2017:
                                preprevious_row = rows_id[rows_id['year'] == previous_year - i]

                                try:
                                    preprevious_value = preprevious_row[feature].iloc[0]

                                    # if value is still none go one more year lower
                                    if not preprevious_value > -999999:
                                        if previous_year - i == 2018:
                                            new_list.append(value)
                                            break
                                        i = i + 1
                                    else:
                                        new_list.append(value - preprevious_value)
                                        break
                                except:
                                    new_list.append(value - previous_value)
                                    break

                    # else count difference between current year and year before and append
                    else:
                        new_list.append(value - previous_value)

                # convert list to series
                new_row = pd.DataFrame(columns=features, data=[new_list])

                # append to new dataframe
                new_df = pd.concat([new_df, new_row])

    # fix data types
    new_df['year'] = [int(x) for x in new_df['year']]
    new_df['quality_flag'] = [int(x) for x in new_df['quality_flag']]

    # once all this looping is done, convert to gdf and save
    gdf = gpd.GeoDataFrame(new_df)
    gdf = gdf.set_crs(32633)
    gdf.to_file(f'data/temp/{label}_features_dy.gpkg')
    return


def centerlineToPoints(glacier_id):
    import shapely

    # glacier id
    glacier_id = 'G016964E77694N'

    # load centerlines
    centerlines = gpd.read_file('data/data/centerlines/RGI2000-v7.0-L-07_svalbard_jan_mayen.shp')

    from shapely.geometry import LineString
    from shapely.ops import substring


    line = centerlines[centerlines['glims_id'] == glacier_id].geometry

    mp = shapely.geometry.MultiPoint()
    for i in np.arange(0, line.length, 0.2):
        s = substring(line, i, i + 0.2)
        mp = mp.union(s.boundary)


def fillNansBins(X):
    # fill nans in binned data

    for i in range(len(X)):
        val = X[i][0]
        # if value for given bin is nan, make average of bins around
        if math.isnan(val):
            # if its the lowest bin, find nearest upper value
            if i == 0:
                # find nearest upper value
                n = i
                while math.isnan(val):
                    n = n + 1
                    if n == 18:
                        val = float('nan')
                        break
                    else:
                        val = X[n][0]

            # else if its the last bin
            elif i == (len(X) - 1):
                n = i
                while math.isnan(val):
                    # go one bin down
                    n = n - 1
                    if n == 0:
                        val = float('nan')
                        break
                    else:
                        val = X[i][0]
            # else if its one of the middle bins
            else:
                # find nearest lower value
                lower_val = float('nan')
                n = i
                while math.isnan(lower_val):
                    n = n - 1
                    # catch if out of bounds
                    if n == -1:
                        lower_val = float('nan')
                        break
                    else:
                        lower_val = X[n][0]

                # find nearest upper value
                upper_val = float('nan')
                n = i
                while math.isnan(upper_val):
                    n = n + 1
                    if n == 17:
                        upper_val = float('nan')
                        break
                    else:
                        upper_val = X[n][0]

                # make average out of that
                if (lower_val == float('nan')) & (upper_val == float('nan')):
                    break
                val = (lower_val + upper_val) / 2
                X[i] = val

    return X





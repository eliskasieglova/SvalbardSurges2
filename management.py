import os
from vars import label, classification_method
import geopandas as gpd
import pandas as pd
from pathlib import Path
import geojson


def createFolder(path):
    """create new folder"""

    # check if folder already exists
    if not os.path.exists(path):
        # if not, make folder
        os.mkdir(path)

    return


# get glacier ids
def listGlacierIDs(glaciers):
    glacier_ids = []

    for i in range(len(glaciers)):
        row = glaciers.iloc[i]
        glacier_id = row['glims_id']
        glacier_ids.append(glacier_id)

    return glacier_ids


def loadGlacierShapefile(glacier_id):
    outpath = Path(f'data/temp/glaciers/{glacier_id}.gpkg')

    # cache
    if outpath.is_file():
        return gpd.read_file(outpath)

    # load the RGI shapefile
    rgi = gpd.read_file(f'data/data/rgi_{label}.gpkg')

    # select glacier with input glacier_id
    glacier = rgi[rgi['glims_id'] == glacier_id]

    # save this glacier
    glacier.to_file(outpath)

    return glacier


def geoJsonToJS():
    """
    Converts geoJSON with results to .js files for visualization purposes.
    """
    print('converting geojson to js')

    years = [2019, 2020, 2021, 2022, 2023]
    for year in years:
        import json

        src = Path(f"data/results/{classification_method}results_{year}.geojson")
        dst = Path(f"eliskasieglova.github.io/data/{classification_method}results{year}.js")

        #geojson_str = json.loads(src.read_text(encoding='iso-8859-1'))
        #dst.write_text(f"var results{year} = {geojson_str};", encoding='iso-8859-1')
        with open(src, encoding='iso-8859-1') as f:
            geojson_text = geojson.load(f)
            dst.write_text(f"var {classification_method}results{year}={geojson_text}", encoding='iso-8859-1')


def gpkg2shp(glacier_id='G013901E78579N'):

    # convert glacier polygon
    gpkg = gpd.read_file(f'data/temp/glaciers/{glacier_id}.gpkg')
    gpkg.to_file(f'data/temp/weloveshapefiles/{glacier_id}.shp')

    # convert icesat points for each year
    years = [2019, 2020, 2021, 2022, 2023]
    for year in years:
        gpkg = gpd.read_file(f'data/temp/glaciers/{glacier_id}_{year}.gpkg')
        gpkg.to_file(f'data/temp/weloveshapefiles/{glacier_id}_{year}.shp')



    return


def convertToIndividualSurges():
    # convert RF results to df with glacier ids and columns with info about in which years
    # glacier was detected as surging

    # read data
    data = gpd.read_file('data/results/RFresults_svalbard_2024-06-28.gpkg')

    # create columns
    glacier_ids = [x for x in data.glacier_id.unique()]
    glacier_names = [data[data['glacier_id'] == x].glacier_name.iloc[0] for x in glacier_ids]
    y2019 = [None if x == -999 else int(x) for x in data[data['year'] == 2019].surging]
    y2020 = [None if x == -999 else int(x) for x in data[data['year'] == 2020].surging]
    y2021 = [None if x == -999 else int(x) for x in data[data['year'] == 2021].surging]
    y2022 = [None if x == -999 else int(x) for x in data[data['year'] == 2022].surging]
    y2023 = [None if x == -999 else int(x) for x in data[data['year'] == 2023].surging]
    p_2019 = [x for x in data[data['year'] == 2019].probability]
    p_2020 = [x for x in data[data['year'] == 2020].probability]
    p_2021 = [x for x in data[data['year'] == 2021].probability]
    p_2022 = [x for x in data[data['year'] == 2022].probability]
    p_2023 = [x for x in data[data['year'] == 2023].probability]

    new_results = pd.DataFrame()
    new_results['glacier_id'] = glacier_ids
    new_results['glacier_name'] = glacier_names
    new_results['2019'] = y2019
    new_results['2020'] = y2020
    new_results['2021'] = y2021
    new_results['2022'] = y2022
    new_results['2023'] = y2023
    new_results['p2019'] = p_2019
    new_results['p2020'] = p_2020
    new_results['p2021'] = p_2021
    new_results['p2022'] = p_2022
    new_results['p2023'] = p_2023

    new_results = new_results.sort_values('p2019', ascending=False)




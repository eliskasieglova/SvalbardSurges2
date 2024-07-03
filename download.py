import icepyx as ipx
from pathlib import Path
import management
from vars import label, spatial_extent, date_range
import requests
import shutil
import tempfile
import os
from vars import products

def downloadICESat(product, outpath):
    """
    Downloades ICESat-2 granules from EarthData and saves them to the output path.

    Necessary to have EarthData login saved as environmental variables under EARTHDATA_USERNAME
    and EARTHDATA_PASSWORD.

    Params:
    - product
        short name for ICESat-2 product ('ATL03', 'ATL06', 'ATL08')
    - spatial_extent
        bounding box of area ([16.65, 77.65, 18.4, 78])
    - date_range
        list of tuple of begin and end dates (['2018-10-14', '2024-02-02'])
    - outpath
        Path object to where the downloaded data should be saved
    """

    #management.createFolder(Path(f'downloads/{label}/'))

    region_a = ipx.Query(product, spatial_extent, date_range)
    region_a.order_granules(verbose=True, subset=False, email=False)
    try:
        region_a.download_granules(outpath)
    except:
        print(region_a)

def downloadSvalbard():

    for product in products:

        print(product)

        download_icesat(
            data_product=product,
            spatial_extent=spatial_extent,
            date_range=date_range
        )

    return


def download_file(url, filename, directory):
    """
    Download a file from the requested URL.
    Function from Erik.

    Parameters
    ----------
    - url:
        The URL to download the file from.
    - filename:
        The output filename of the file. Defaults to the basename of the URL.
    - directory:
        The directory to save the file in. Defaults to `cache/`

    Returns
    -------
    A path to the downloaded file.
    """

    # If `directory` is defined, make sure it's a path. If it's not defined, default to `cache/`
    if isinstance(directory, (str, Path)):
        out_dir = Path(directory)
    else:
        out_dir = Path("data/downloads/")

    if filename is not None:
        out_path = out_dir.joinpath(filename)
    else:
        out_path = out_dir.joinpath(os.path.basename(url))

    # If the file already exists, skip downloading it.
    if not out_path.is_file():
        # Open a data stream from the URL. This means not everything has to be kept in memory.
        with requests.get(url, stream=True) as request:
            # Stop and raise an exception if there's a problem.
            request.raise_for_status()

            # Save the file to a temporary directory first. The file is constantly appended to due to the streaming
            # Therefore, if the stream is cancelled, the file is not complete and should be dropped.
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir).joinpath("temp.file")
                # Write the data stream as it's received
                with open(temp_path, "wb") as outfile:
                    shutil.copyfileobj(request.raw, outfile)

                # When the download is done, move the file to its ultimate location.
                shutil.move(temp_path, out_path)

    return out_path


def download_icesat(data_product, spatial_extent, date_range):
    """
    Download ICESat-2 data from earthdata.

    Reccomendation: have earthdata login details stored as environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)

    Params
    ------
    - spatial_extent
        bounding box of wanted area as list of coords [left, bottom, right, top] in decimal degrees
    - date_range
        list of beginning and end of time span ['yyyy-mm-dd', 'yyyy-mm-dd']
    - data_product
        'ATL03/6/8' as str.

    Return
    ------
    Output path to ICESat-2 data product.
    """

    # granules are saved to cache
    output_path = Path(f'data/downloads/{data_product}')

    # specifications for download
    region_a = ipx.Query(
        data_product,
        spatial_extent,
        date_range,
        start_time='00:00:00',
        end_time='23:59:59'
    )

    region_a.avail_granules()
    region_a.granules.avail

    print('ordering granules')
    # order and download granules, save them to cache/is2_{dataproduct}
    region_a.order_granules()
    print(f'downloading granules {data_product}')
    region_a.download_granules(output_path)

    return output_path

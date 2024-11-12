###################
## create_mosaics.py
## Written pre-CS230
## Downloads satellite data for area of interest and stitches it into
##      a GeoTiff file for later use.
import argparse
import csv
from datetime import date
import geopandas as gpd
import io
import mercantile
import os
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
import requests
from shapely import force_2d
from utils.geom_utils import reproject_geoframe, get_lat_long_bounds

base_url = os.getenv("TILE_SERVER")
api_key = os.getenv("PLANET_API_KEY")
# We use web mercator because it is the CRS used by both planet basemaps
# and by the mercantile library
epsg_code = 3857


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-prefix",
        required=True,
        help="file prefix of saved images",
    )
    parser.add_argument(
        "--mapping-area",
        required=True,
        help="Relative file path to shapefile",
    )
    parser.add_argument(
        "--dates",
        required=True,
        help="CSV of ISO format date string in form yyyy-mm-dd",
    )
    parser.add_argument(
        "--zoom-level",
        required=False,
        default=15,
        help="zoom level between 0 and 18 (optional)",
    )
    return parser.parse_args()


def create_mosaic(
    shp_file: str, target_date: date, outfile_prefix: str, zoom: int = 15
):
    # Read the shp file and parse the polygons
    geo_shape = gpd.read_file(shp_file)
    reprojected_geo_shape = reproject_geoframe(geo_shape, epsg_code)
    valid = reprojected_geo_shape.loc[reprojected_geo_shape.geometry.is_valid]
    hull = force_2d(valid.unary_union.convex_hull)
    west, south, east, north = get_lat_long_bounds(hull)

    # Get the tile numbers covering the bounding box
    tiles = list(mercantile.tiles(west, south, east, north, zooms=zoom))

    # Download the tiles and store them in a list
    year, month = target_date.year, target_date.month
    mosaic_name = f"global_monthly_{year}_{month:02d}_mosaic"
    images = []
    total_number = len(tiles)
    counter = 0
    for tile in tiles:
        x, y = tile.x, tile.y
        # use xy_bounds so that bounding box is in meters to match web mercator projection
        bounding_box = mercantile.xy_bounds(tile)

        url = f"https://{base_url}/basemaps/v1/planet-tiles/{mosaic_name}/gmap/{zoom}/{x}/{y}.png?api_key={api_key}"
        response = requests.get(url)

        with MemoryFile(io.BytesIO(response.content)) as memfile:
            with memfile.open() as src:
                image = src.read()
                if len(image.shape) == 2:
                    (height, width) = image.shape
                elif len(image.shape) == 3:
                    (_, height, width) = image.shape
                transform = from_bounds(
                    bounding_box.left,
                    bounding_box.bottom,
                    bounding_box.right,
                    bounding_box.top,
                    width,
                    height,
                )
                profile = {
                    "driver": "GTiff",
                    "width": width,
                    "height": height,
                    "count": src.count,
                    "dtype": src.dtypes[0],
                    "crs": CRS.from_epsg(epsg_code),
                    "transform": transform,
                }

        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(image)
        images.append(memfile)

        counter += 1
        if counter % 100 == 0:
            print(f"Completed {counter}/{total_number}")

    # Merge the images into a mosaic
    mosaic, mosaic_transform = merge([image.open() for image in images])

    with MemoryFile() as memfile:
        if len(mosaic.shape) == 2:
            (height, width) = mosaic.shape
        elif len(mosaic.shape) == 3:
            (_, height, width) = mosaic.shape

        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": CRS.from_epsg(epsg_code),
            "transform": mosaic_transform,
        }
        with memfile.open(**profile) as dst:
            dst.write(mosaic)
        with memfile.open() as ms:
            # Crop the mosaic to the bounding box of the hull
            cropped_mosaic, cropped_transform = mask(ms, [hull], crop=True)

            # Write the mosaic to a GeoTiff file
            profile.update(
                {
                    "driver": "GTiff",
                    "height": cropped_mosaic.shape[1],
                    "width": cropped_mosaic.shape[2],
                    "transform": cropped_transform,
                }
            )

            outfile = f"{outfile_prefix}_{year}_{month}.tif"
            with rasterio.open(outfile, "w", **profile) as out:
                out.write(cropped_mosaic)


if __name__ == "__main__":
    args = get_args()
    with open(args.dates) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            create_mosaic(
                args.mapping_area,
                date.fromisoformat(row[0]),
                args.file_prefix,
                int(args.zoom_level),
            )

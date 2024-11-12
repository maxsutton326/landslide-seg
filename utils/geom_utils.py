import mercantile
from geopandas import GeoDataFrame
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import shape

"""
This file contains utility functions used in the python-planet-tile respository. 
These functions are all used to help manipulate and analyze geographic data, such as geodataframes
and shapes.
"""


# reproject_geoframe
#
# arg to_project: geoframe to reproject
# arg dest_code: destination epsg code
#
# returns geopandas geoframe in dest_code crs
def reproject_geoframe(to_project: GeoDataFrame, dest_code):
    destination_crs = CRS.from_epsg(dest_code)

    geometry = transform_geom(
        src_crs=to_project.crs,
        dst_crs=destination_crs,
        geom=to_project.geometry.values,
    )
    return to_project.set_geometry(
        [shape(geom) for geom in geometry],
        crs=destination_crs,
    )


# get_lat_long_bounds
#
# arg geometry: geometry in meters to get bounds of
#
# returns bounds in decimal degrees in tuple format
# (west, south, east, north)
def get_lat_long_bounds(geometry):
    minx, miny, maxx, maxy = geometry.bounds
    # convert bounds to degrees
    minLatLong = mercantile.lnglat(minx, miny)
    maxLatLong = mercantile.lnglat(maxx, maxy)
    return minLatLong.lng, minLatLong.lat, maxLatLong.lng, maxLatLong.lat

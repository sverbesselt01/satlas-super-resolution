'''
import rasterio
from rasterio.warp import reproject

tci_jp2_path = 'path/to/TCI.jp2'
with rasterio.open(tci_jp2_path) as src:
    img_rep, meta_rep = reproject(
        img, meta, rasterio.crs.CRS.from_epsg(3857), resolution=(9.555, 9.555), resampling=rasterio.warp.Resampling.bilinear)

with rasterio.open(tci_jp2_path.replace('.jp2', '_rep.jp2', 'w', **meta_rep) as dst:
    dst.write(img_rep)
'''

import rasterio
import numpy as np
from PIL import Image
import os
from pyproj import Transformer


def convert_to_8bit(band_data):
    """Converts band data to 8-bit (0-255) range."""
    min_val = np.min(band_data)
    max_val = np.max(band_data)
    if max_val > min_val:
        scaled_data = ((band_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        scaled_data = np.full_like(band_data, 0, dtype=np.uint8)
    return scaled_data


def tile_to_mercator(tile_x, tile_y, zoom=17, tile_size=256):
    """Converts tile coordinates to Web Mercator bounding box."""
    n = 2.0 ** zoom
    lon_min = tile_x / n * 360.0 - 180.0
    lat_max = np.arctan(np.sinh(np.pi * (1 - tile_y / n))) * 180.0 / np.pi
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    lat_min = np.arctan(np.sinh(np.pi * (1 - (tile_y + 1) / n))) * 180.0 / np.pi
    return lon_min, lat_min, lon_max, lat_max


def pixel_to_mercator(raster, col, row):
    """Converts pixel coordinates in a raster to Web Mercator coordinates."""
    transformer = Transformer.from_crs(raster.crs, "EPSG:3857", always_xy=True)
    x, y = raster.xy(row, col)
    return transformer.transform(x, y)


def process_sentinel2_tiled(input_tiff, output_root):
    """
    Processes a Sentinel-2 L2A TIFF image:
    - Creates and saves 32x32 pixel tiles for TCI (RGB) and individual bands.
    - Saves tiles in a Web-Mercator grid structure.

    Args:
        input_tiff (str): Path to the input Sentinel-2 L2A TIFF file.
        output_root (str): Root directory for saving the tiled data (e.g., "./data/sentinel2").
    """
    try:
        with rasterio.open(input_tiff) as src:
            num_bands = src.count
            bounds = src.bounds
            width = src.width
            height = src.height
            crs = src.crs

            # Define the desired bands and their names
            desired_bands = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
            band_names_mapping = {
                2: "b02",
                3: "b03",
                4: "b04",
                5: "b05",
                6: "b06",
                7: "b07",
                8: "b08",
                9: "b8a",
                11: "b11",
                12: "b12",
            }

            bands_data = []
            band_names = []
            for band_index in desired_bands:
                if 1 <= band_index <= num_bands:
                    bands_data.append(src.read(band_index))
                    band_names.append(band_names_mapping[band_index])
                else:
                    print(f"Warning: Band {band_index} not found in the input TIFF.")

            # Create TCI if bands 2, 3, and 4 exist
            tci_data = None
            if num_bands >= 4:
                red_scaled = convert_to_8bit(bands_data[3])  # Band 4 is index 3
                green_scaled = convert_to_8bit(bands_data[2])  # Band 3 is index 2
                blue_scaled = convert_to_8bit(bands_data[1])  # Band 2 is index 1
                tci_data = np.stack([red_scaled, green_scaled, blue_scaled], axis=-1)
                band_names.insert(0, "tci")
                bands_data.insert(0, tci_data)

            tile_size = 32
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    x_end = min(x + tile_size, width)
                    y_end = min(y + tile_size, height)
                    tile_width = x_end - x
                    tile_height = y_end - y

                    # Get Web-Mercator coordinates of the top-left pixel of the tile
                    mercator_x, mercator_y = pixel_to_mercator(src, x, y)

                    # Calculate the tile indices in the 2^17 x 2^17 grid (assuming zoom level 17)
                    zoom = 17
                    # Define Web Mercator bounds and world extent at zoom 17
                    mercator_extent = 20037508.34 * 2
                    world_pixels = 2 ** zoom * 256
                    pixels_per_meter = world_pixels / mercator_extent
                    meters_per_tile = 256 / pixels_per_meter
                    meters_per_small_tile = tile_size / pixels_per_meter

                    # Calculate tile indices
                    tile_x_index = int((mercator_x + 20037508.34) / meters_per_small_tile)
                    tile_y_index = int((20037508.34 - mercator_y) / meters_per_small_tile)

                    tile_folder = os.path.join(output_root, os.path.splitext(os.path.basename(input_tiff))[0],
                                               f"sentinel2/{tile_x_index}_{tile_y_index}")
                    os.makedirs(tile_folder, exist_ok=True)

                    for i, band_data in enumerate(bands_data):
                        band_name = band_names[i]
                        tile = band_data[y:y_end, x:x_end]
                        if band_name == "tci":
                            img = Image.fromarray(tile)
                        else:
                            scaled_tile = convert_to_8bit(tile)
                            img = Image.fromarray(scaled_tile)

                        output_filename = os.path.join(tile_folder, f"{band_name}.png")
                        img.save(output_filename)
                        print(f"Saved tile {tile_x_index}_{tile_y_index}, band {band_name} to: {output_filename}")

    except rasterio.RasterioIOError:
        print(f"Error: Could not open or read the file: {input_tiff}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_file = "G:/Mijn Drive/Deep resolution/Test_UAV_VMM/Sentinel2_L2A_20230301.tiff"  # Replace with the actual path to your TIFF file
    output_root_dir = "G:/Mijn Drive/Deep resolution/satlas-super-resolution/data"  # Root directory for saving tiles

    process_sentinel2_tiled(input_file, output_root_dir)
    print(f"\nTiling complete. Tiles saved under: {output_root_dir}")
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


def pixel_to_mercator(raster, col, row):
    """Converts pixel coordinates in a raster to Web Mercator coordinates."""
    transformer = Transformer.from_crs(raster.crs, "EPSG:3857", always_xy=True)
    x, y = raster.xy(row, col)
    return transformer.transform(x, y)


def process_sentinel2_tiled_tci_only(input_tiff, output_root):
    """
    Processes a Sentinel-2 L2A TIFF image and saves only the TCI (RGB) tiles.
    Tiles are saved as 32x32 pixel images in a Web-Mercator grid structure.

    Args:
        input_tiff (str): Path to the input Sentinel-2 L2A TIFF file.
        output_root (str): Root directory for saving the tiled data (e.g., "./data/sentinel2").
    """
    try:
        with rasterio.open(input_tiff) as src:
            num_bands = src.count
            width = src.width
            height = src.height

            tci_data = None
            if num_bands >= 4:
                red_band = src.read(4)  # Band 4 (Red)
                green_band = src.read(3)  # Band 3 (Green)
                blue_band = src.read(2)  # Band 2 (Blue)
                red_scaled = convert_to_8bit(red_band)
                green_scaled = convert_to_8bit(green_band)
                blue_scaled = convert_to_8bit(blue_band)
                tci_data = np.stack([red_scaled, green_scaled, blue_scaled], axis=-1)
            else:
                print("Warning: Not enough bands to create TCI (requires at least 4 bands). Skipping TCI generation.")
                return

            tile_size = 32
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    x_end = min(x + tile_size, width)
                    y_end = min(y + tile_size, height)

                    # Get Web-Mercator coordinates of the top-left pixel of the tile
                    mercator_x, mercator_y = pixel_to_mercator(src, x, y)

                    # Calculate the tile indices in the 2^17 x 2^17 grid (assuming zoom level 17)
                    zoom = 17
                    mercator_extent = 20037508.34 * 2
                    world_pixels = 2 ** zoom * 256
                    pixels_per_meter = world_pixels / mercator_extent
                    meters_per_small_tile = tile_size / pixels_per_meter

                    tile_x_index = int((mercator_x + 20037508.34) / meters_per_small_tile)
                    tile_y_index = int((20037508.34 - mercator_y) / meters_per_small_tile)

                    tile_folder = os.path.join(output_root, os.path.splitext(os.path.basename(input_tiff))[0],
                                               f"sentinel2/{tile_x_index}_{tile_y_index}")
                    print(f"Creating folder: {tile_folder}")  # Debugging line
                    os.makedirs(tile_folder, exist_ok=True)

                    tile = tci_data[y:y_end, x:x_end]
                    print(f"Tile shape: {tile.shape}")  # Debugging line
                    img = Image.fromarray(tile)

                    output_filename = os.path.join(tile_folder, f"tci.png")
                    img.save(output_filename)
                    print(f"Saved TCI tile {tile_x_index}_{tile_y_index} to: {output_filename}")

    except rasterio.RasterioIOError:
        print(f"Error: Could not open or read the file: {input_tiff}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_file = "G:/Mijn Drive/Deep resolution/Test_UAV_VMM/Sentinel2_L2A_20230301.tiff"  # Replace with the actual path to your TIFF file
    output_root_dir = "G:/Mijn Drive/Deep resolution/satlas-super-resolution/data"  # Root directory for saving tiles

    process_sentinel2_tiled_tci_only(input_file, output_root_dir)
    print(f"\nTiling of TCI complete. Tiles saved under: {output_root_dir}")
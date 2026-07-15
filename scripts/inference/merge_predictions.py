import os
import glob
import pickle
import logging
import numpy as np
import xarray as xr
import dask.array as da
from collections import defaultdict
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_prediction_files(log_dir, num_ensemble=8):
    """
    This function is used to process pkl files produced by inference_fsdp.py.
    Merge prediction files in batches of 10, creating a zarr file with the first batch
    and appending subsequent batches.
    
    Args:
        log_dir (str): Path to the log directory containing the tmp_predictions folder
        num_ensemble (int): Number of ensemble members (default: 1). If 1, processes without ensemble dimension
    """
    tmp_dir = os.path.join(log_dir, 'tmp_predictions')
    if not os.path.exists(tmp_dir):
        raise ValueError(f"Temporary directory {tmp_dir} does not exist")

    # Get all prediction files
    pred_files = sorted(glob.glob(os.path.join(tmp_dir, 'pred_*.pkl')))
    if not pred_files:
        raise ValueError(f"No prediction files found in {tmp_dir}")

    # Process files in batches of 10
    batch_size = 10
    save_path = os.path.join(log_dir, 'predictions.zarr')
    
    for batch_idx in range(0, len(pred_files), batch_size):
        batch_files = pred_files[batch_idx:batch_idx + batch_size]
        logging.info(f"Processing batch {batch_idx//batch_size + 1}, files {batch_idx+1} to {batch_idx+len(batch_files)}")

        # Initialize containers for this batch
        combined_data = defaultdict(list)
        times = []
        
        # Get coordinates from first file (they're the same for all files)
        with open(batch_files[0], 'rb') as f:
            first_batch = pickle.load(f)
            lats = first_batch['latitude']
            lons = first_batch['longitude']
            levels = first_batch['level']

        # Load and combine predictions for this batch
        for pred_file in batch_files:
            logging.info(f"Processing {os.path.basename(pred_file)}")
            with open(pred_file, 'rb') as f:
                batch_preds = pickle.load(f)

            # Collect times only
            times.extend(batch_preds['time'])

            # Collect predictions
            for key, value in batch_preds.items():
                if key not in ['time', 'latitude', 'longitude', 'level']:
                    combined_data[key].append(value)

        # Convert to unique coordinates
        times = np.array(times, dtype='datetime64[ns]')
        lats = np.array(lats)
        lons = np.array(lons)
        levels = np.array(levels)

        # Create xarray dataset for this batch
        ds_dict = {}
        coords_dict = {
            'time': times,
            'latitude': lats,
            'longitude': lons,
            'level': levels
        }

        # Define surface variables list
        surface_vars = [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            'mean_sea_level_pressure',
            'sea_ice_cover',
            'sea_surface_temperature',
            'surface_pressure',
            'toa_incident_solar_radiation',
            'total_cloud_cover',
            'total_column_water_vapour',
            'total_precipitation']

        # Process each variable
        for var_name, data_list in combined_data.items():
            logging.info(f"Processing data for variable: {var_name}")
            data = np.concatenate(data_list, axis=0)
            
            if any(var in var_name for var in surface_vars):
                if num_ensemble > 1:
                    data = data.reshape(len(times), num_ensemble, len(lats), len(lons))
                    dims = ['time', 'ensemble', 'latitude', 'longitude']
                    chunks = (1, -1, -1, -1)
                else:
                    data = data.reshape(len(times), len(lats), len(lons))
                    dims = ['time', 'latitude', 'longitude']
                    chunks = (1, -1, -1)
            else:
                if num_ensemble > 1:
                    data = data.reshape(len(times), num_ensemble, len(levels), len(lats), len(lons))
                    dims = ['time', 'ensemble', 'level', 'latitude', 'longitude']
                    chunks = (1, -1, -1, -1, -1)
                else:
                    data = data.reshape(len(times), len(levels), len(lats), len(lons))
                    dims = ['time', 'level', 'latitude', 'longitude']
                    chunks = (1, -1, -1, -1)

            dask_array = da.from_array(data, chunks=chunks)

            coords = {
                'time': coords_dict['time'],
                'latitude': coords_dict['latitude'],
                'longitude': coords_dict['longitude']
            }
            
            if num_ensemble > 1:
                coords['ensemble'] = np.arange(num_ensemble)
            if 'level' in dims:
                coords['level'] = coords_dict['level']

            ds_dict[var_name] = xr.DataArray(
                data=dask_array,
                dims=dims,
                coords=coords
            )

        ds = xr.Dataset(ds_dict)
        
        # For first batch, create new zarr file
        if batch_idx == 0:
            logging.info(f"Creating new zarr file at {save_path}")
            ds.to_zarr(save_path, mode='w')
        # For subsequent batches, append to existing zarr file
        else:
            logging.info(f"Appending to existing zarr file at {save_path}")
            ds.to_zarr(save_path, append_dim='time')
    print(ds)
    logging.info("Merge completed successfully")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Merge prediction files into a single zarr dataset')
    parser.add_argument('--log_dir', type=str, help='Path to the log directory')
    parser.add_argument('--num_ensemble', type=int, default=1, help='Number of ensemble members (default: 8)')
    args = parser.parse_args()

    merge_prediction_files(args.log_dir, args.num_ensemble)
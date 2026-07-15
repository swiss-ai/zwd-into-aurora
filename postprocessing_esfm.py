import numpy as np
import torch
import xarray as xr
from datetime import datetime

from aurora.batch import Batch, Metadata
from utils.transform_data import transform_data


def transform_time(ds, start_time_test):
    ''' Add the initial time and the lead time when using several start dates '''
    # Since we already have init_time and lead_time, we just need to add the initial_time dimension
    # and convert lead_time to prediction_timedelta if needed
    ds2 = ds.expand_dims(dim={'initial_time':[np.datetime64(start_time_test, 'h')]})
    ds2 = ds2.rename({"lead_time": "prediction_timedelta"})
    # lead_time is already in hours, so we can convert it to timedelta directly
    ds2['prediction_timedelta'] = ds2.prediction_timedelta.astype('timedelta64[h]')
    return ds2


def make_target_batch(ds, times, d_srf, d_static, d_atmos, locations, scales, dict_ds_extended=None, lead_time_h=6, device='cuda'):
    accumulated = {'total_precipitation': True,
                            'total_precipitation_MSWEP': False,
                            'potential_evaporation': True,
                            'evaporation': True,
                            "surface_latent_heat_flux": True,
                            "surface_sensible_heat_flux": True,
                            "surface_net_solar_radiation": True,
                            "surface_net_thermal_radiation": True,
                            "surface_solar_radiation_downwards": True,
                            "surface_thermal_radiation_downwards": True,
                            "top_net_solar_radiation": True,
                            "top_net_thermal_radiation": True,
                            "toa_incident_solar_radiation": True,
    }
    surf_vars_y = {}
    for k in d_srf.keys():
        if d_srf[k] in ds.data_vars:
            surf_vars_y[k] = torch.tensor(ds.sel(time=times)[d_srf[k]].values, device=device)
            if d_srf[k] in accumulated.keys() and accumulated[d_srf[k]]:
                for h in range(1, lead_time_h):
                    surf_vars_y[k] += torch.tensor(ds.sel(time=np.datetime64(times) - np.timedelta64(h,'h'))[d_srf[k]].values, device=device)
        else:
            surf_vars_y[k] = torch.tensor(dict_ds_extended[d_srf[k]].sel(time=times).values, device=device)
            if d_srf[k] in accumulated.keys() and accumulated[d_srf[k]]:
                for h in range(1, lead_time_h):
                    surf_vars_y[k] += torch.tensor(dict_ds_extended[d_srf[k]].sel(time=np.datetime64(times) - np.timedelta64(h,'h')).values, device=device)
    atmos_vars_y = {k: torch.tensor(ds.sel(time=times)[d_atmos[k]].values, device=device) for k in d_atmos.keys()}
    time_metadata = times
    static_vars_y = {k: torch.tensor(ds.sel(time=times)[d_static[k]].values, device=device) for k in d_static.keys()}
    
    batch_y = Batch(
        surf_vars=surf_vars_y,
        static_vars=static_vars_y,
        atmos_vars=atmos_vars_y,
        metadata=Metadata(
            lat=torch.tensor(ds.latitude.values, device=device),
            lon=torch.tensor(ds.longitude.values, device=device),
            time=(time_metadata,),
            atmos_levels=ds.level.values,
            locations={k: v for k, v in locations.items()},
            scales={k: v for k, v in scales.items()},
        ),
    )

    return batch_y


def make_input_batch(ds, times, d_srf, d_static, d_atmos, locations, scales, dict_ds_extended=None, lead_time_h=6, device='cuda', accumulated=None):
    if accumulated is None:
        accumulated = {'total_precipitation': True,
                                'total_precipitation_MSWEP': False,
                                'potential_evaporation': True,
                                'evaporation': True,
                                "surface_latent_heat_flux": True,
                                "surface_sensible_heat_flux": True,
                                "surface_net_solar_radiation": True,
                                "surface_net_thermal_radiation": True,
                                "surface_solar_radiation_downwards": True,
                                "surface_thermal_radiation_downwards": True,
                                "top_net_solar_radiation": True,
                                "top_net_thermal_radiation": True,
                                "toa_incident_solar_radiation": True,
        }
        ### TO DO: check that the accumulation works as expected, for variables in ds and in dict_ds_extended
    
    surf_vars_y = {}
    for k in d_srf.keys():
        if d_srf[k] in ds.data_vars:
            if d_srf[k] in accumulated.keys() and accumulated[d_srf[k]]:
                ds_acc = ds.sel(time=slice(np.datetime64(times[0], 'h') - np.timedelta64(lead_time_h+1, 'h'), np.datetime64(times[1])))
                acc = ds_acc[d_srf[k]].rolling(time=lead_time_h, center=False).sum().sel(time=times)
                surf_vars_y[k] = acc.values
            else:
                surf_vars_y[k] = ds.sel(time=times)[d_srf[k]].values

        else:
            surf_vars_y[k] = dict_ds_extended[d_srf[k]].sel(time=times).values
            if d_srf[k] in accumulated.keys() and accumulated[d_srf[k]]:
                for h in range(1, lead_time_h):
                    surf_vars_y[k] += dict_ds_extended[d_srf[k]].sel(time=times - np.timedelta64(h,'h')).values

        surf_vars_y[k] = torch.tensor(surf_vars_y[k][None], device=device)
    atmos_vars_y = {k: torch.tensor(ds.sel(time=times)[d_atmos[k]].values[None], device=device) for k in ["z", "u", "v", "t", "q"]}
    time_metadata = times[1]
    static_vars_y = {k: torch.tensor(ds[d_static[k]].values, device=device) for k in ["lsm", "z", "slt"]}
    
    batch_y = Batch(
        surf_vars=surf_vars_y,
        static_vars=static_vars_y,
        atmos_vars=atmos_vars_y,
        metadata=Metadata(
            lat=torch.tensor(ds.latitude.values, device=device),
            lon=torch.tensor(ds.longitude.values, device=device),
            time=(time_metadata,),
            atmos_levels=ds.level.values,
            locations={k: v for k, v in locations.items()},
            scales={k: v for k, v in scales.items()},

        ),
    )

    return batch_y


def batch2xr(batch, target, pred, d_srf, d_atmos=None, with_atmos=False, init_time=None, lead_time_h=0):
    # Use init_time if provided, otherwise use target time
    if init_time is None:
        init_time = target.metadata.time[0]
    
    tmp = dict()
    for k in d_srf.keys():
        target_data = target.surf_vars[k].cpu().numpy()
        tmp[d_srf[k]] = (["init_time", "lead_time", "latitude", "longitude"], target_data[None, None])
    if with_atmos:
        for k in d_atmos.keys():
            # FIX: Move atmospheric targets to CPU as well
            atmos_target_data = target.atmos_vars[k].cpu().numpy()
            tmp[d_atmos[k]] = (["init_time", "lead_time", "level", "latitude", "longitude"], atmos_target_data[None, None])
        ds_target = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                level=batch.metadata.atmos_levels,
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )
    else:
        ds_target = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )

    tmp = dict()
    for k in d_srf.keys():
        pred_data = pred.surf_vars[k][0].cpu().numpy()
        if pred_data.ndim == 3:
            data_no_batch = pred_data[0]  # Remove batch dim if still present
        else:
            data_no_batch = pred_data
        tmp[d_srf[k]] = (["init_time", "lead_time", "latitude", "longitude"], data_no_batch[None, None])
            
    if with_atmos:
        for k in d_atmos.keys():
            raw_atmos = pred.atmos_vars[k][0].cpu().numpy()
            
            # raw_atmos is (1, 13, 720, 1440) -> we want (1, 1, 13, 720, 1440) for [init_time, lead_time, level, lat, lon]
            data_no_batch = raw_atmos[0]  # Remove batch dim: (13, 720, 1440)
            tmp[d_atmos[k]] = (["init_time", "lead_time", "level", "latitude", "longitude"], data_no_batch[None, None])

        ds_pred = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                level=batch.metadata.atmos_levels,
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )
    else:    
        ds_pred = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )

    return ds_target, ds_pred


def batch2xr_ensemble(batch, pred_ens, d_srf, num_ensemble, d_atmos=None, with_atmos=False, init_time=None, lead_time_h=0):
    '''  '''
    # Use init_time if provided, otherwise use pred_ens time
    if init_time is None:
        init_time = pred_ens.metadata.time[0]
    
    tmp = dict()
    for k in d_srf.keys():
        # Get ensemble data - predictions are already denormalized by the model
        ens_data = pred_ens.surf_vars[k][0].cpu().numpy()  # Should be [num_ensemble, lat, lon] or [batch, num_ensemble, lat, lon]
        
        # If there's a batch dimension, remove it
        if ens_data.ndim == 4:  # [batch, num_ensemble, lat, lon]
            ens_data = ens_data[0]  # -> [num_ensemble, lat, lon]
        
        # Add init_time and lead_time dimensions: [num_ensemble, lat, lon] -> [num_ensemble, init_time, lead_time, lat, lon]
        tmp[d_srf[k]] = (["number", "init_time", "lead_time", "latitude", "longitude"], ens_data[:, None, None])
        
    if with_atmos:
        for k in d_atmos.keys():
            # Ensemble atmospheric data
            ens_atmos = pred_ens.atmos_vars[k][0].cpu().numpy()  # Should be [num_ensemble, level, lat, lon] or [batch, num_ensemble, level, lat, lon]
            
            # If there's a batch dimension, remove it
            if ens_atmos.ndim == 5:  # [batch, num_ensemble, level, lat, lon]
                ens_atmos = ens_atmos[0]  # -> [num_ensemble, level, lat, lon]
            
            # Add init_time and lead_time dimensions: [num_ensemble, level, lat, lon] -> [num_ensemble, init_time, lead_time, level, lat, lon]
            tmp[d_atmos[k]] = (["number", "init_time", "lead_time", "level", "latitude", "longitude"], ens_atmos[:, None, None])

        ds_pred_ens = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                level=batch.metadata.atmos_levels,
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h]),
                number=np.arange(num_ensemble)
            ),
            data_vars = tmp,
        )
    else:    
        ds_pred_ens = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h]),
                number=np.arange(num_ensemble)
            ),
            data_vars = tmp,
        )

    return ds_pred_ens


def batch2xr_transform(batch, target, pred, d_srf, d_atmos=None, with_atmos=False, init_time=None, lead_time_h=0):
    ''' This is not exactly the same function as in postprocessing.py because target remains on cpu in this version.
    Also now, the dataset returns transformed data. Therefore, we need to remove the transformation on the target and prediction '''
    # Use init_time if provided, otherwise use target time
    if init_time is None:
        init_time = target.metadata.time[0]
    
    tmp = dict()
    for k in d_srf.keys():
        # Handle target data shape - target.surf_vars[k] might have batch dimension
        target_data = target.surf_vars[k]
        # Add None to make it 3D, then transform, then handle batch dimension
        transformed_data = transform_data(target_data[None], k, direct=False)
        
        # Remove any extra dimensions and add init_time, lead_time
        if transformed_data.ndim == 3:  # [1, lat, lon]
            data_no_batch = transformed_data[0]  # -> [lat, lon]
        else:
            data_no_batch = transformed_data.squeeze()  # Fallback
            
        tmp[k] = (["init_time", "lead_time", "latitude", "longitude"], data_no_batch[None, None])
        
    if with_atmos:
        for k in d_atmos.keys():
            atmos_data = target.atmos_vars[k]
            # Add init_time and lead_time dimensions
            if atmos_data.ndim == 4:  # [batch, level, lat, lon]
                data_no_batch = atmos_data[0]  # Remove batch
            else:  # [level, lat, lon]
                data_no_batch = atmos_data
            tmp[k] = (["init_time", "lead_time", "level", "latitude", "longitude"], data_no_batch[None, None])
        ds_target = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                level=batch.metadata.atmos_levels,
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )
    else:
        ds_target = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )

    tmp = dict()
    for k in d_srf.keys():
        # Handle prediction data shape similar to regular batch2xr
        raw_data = pred.surf_vars[k][0].cpu().numpy()
        transformed_data = transform_data(raw_data, k, direct=False)
        
        # Remove batch dimension and add init_time, lead_time dimensions
        if transformed_data.ndim == 3:  # [batch, lat, lon]
            data_no_batch = transformed_data[0]  # -> [lat, lon]
        else:
            data_no_batch = transformed_data
            
        tmp[k] = (["init_time", "lead_time", "latitude", "longitude"], data_no_batch[None, None])
        
    if with_atmos:
        for k in d_atmos.keys():
            raw_atmos = pred.atmos_vars[k][0].cpu().numpy()
            
            # Remove batch dimension and add init_time, lead_time dimensions
            if raw_atmos.ndim == 4:  # [batch, level, lat, lon]
                data_no_batch = raw_atmos[0]  # -> [level, lat, lon]
            else:
                data_no_batch = raw_atmos
                
            tmp[k] = (["init_time", "lead_time", "level", "latitude", "longitude"], data_no_batch[None, None])

        ds_pred = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                level=batch.metadata.atmos_levels,
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )
    else:    
        ds_pred = xr.Dataset(
            coords=dict(
                longitude=batch.metadata.lon.cpu().numpy(),
                latitude=batch.metadata.lat.cpu().numpy(),
                init_time=np.array([np.datetime64(init_time, 'ns')]),
                lead_time=np.array([lead_time_h])
            ),
            data_vars = tmp,
        )

    return ds_target, ds_pred

## various classes and utilities for dataset and their loaders

import os
import glob
import pickle
import random
import itertools
from typing import Optional, List, Callable, Any, Dict
from datetime import datetime, timedelta

import torch
import cftime
import mmnpz
import numpy as np
import xarray as xr
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, Subset
from torchdata.stateful_dataloader import StatefulDataLoader


from aurora.normalisation import load_normalization_stats
from utils.transform_data import transform_data, accumulate_data


d_srf_abr2full = {"2t": "2m_temperature", 
                  "10u": "10m_u_component_of_wind", 
                  "10v": "10m_v_component_of_wind", 
                  "msl": "mean_sea_level_pressure", 
                  "tp": "total_precipitation",
                  "tp_mswep": "total_precipitation_MSWEP",
                  "pe": "potential_evaporation",
                  "e": "evaporation",
                  "r": "runoff",
                  "swvl": "volumetric_soil_water_layer",
                  "swc": "soil_water_content",
                  "tws_gou": "terrestrial_water_storage_Gou",
                  "tws_itsg": "terrestrial_water_storage_ITSG",
                  "slhf": "surface_latent_heat_flux",
                  "sshf": "surface_sensible_heat_flux",
                  "ssr": "surface_net_solar_radiation", 
                  "str": "surface_net_thermal_radiation",
                  "ssrd": "surface_solar_radiation_downwards",
                  "strd": "surface_thermal_radiation_downwards",
                  "tsr": "top_net_solar_radiation",
                  "ttr": "top_net_thermal_radiation",
                  "tisr": "toa_incident_solar_radiation",
                  "sst": "sea_surface_temperature",
                  "ci": "sea_ice_cover",
                  "co2": "global_CO2",
                  "zwd": "zenith_wet_delay",
                  }
d_static_abr2full = dict(zip(("lsm", "z", "slt"), ("land_sea_mask", "geopotential_at_surface", "soil_type"))) ## nonexisting variable in wb2: "slt": "soil_type"
d_atmos_abr2full = dict(zip(("z", "u", "v", "t", "q", "w"), ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "vertical_velocity")))
d_srf_full2abr = {v: k for k, v in d_srf_abr2full.items()}
d_static_full2abr = {v: k for k, v in d_static_abr2full.items()}
d_atmos_full2abr = {v: k for k, v in d_atmos_abr2full.items()}


class ScalarCO2Mapper:
    def __init__(
            self,
            co2_path: str = '/path/to/data/global_annual_CO2.csv',
            co2_fullname: str='global_CO2',
            lead_time_h: int=6,
            inds = None,
            ):
        self.co2_fullname = co2_fullname

        co2_df = pd.read_csv(co2_path)
        co2_df.columns = ["year", "CO2"]
        time_base = np.array([np.datetime64(f'{year}-07-02T00:00:00.000000000') for year in co2_df["year"]]) # 2nd of July is the middle of the year
        # add timesteps at the ends of range (the need for this depends on the inds definition)
        inds = np.concatenate([[inds[0] - np.timedelta64(lead_time_h, 'h')], inds, [inds[-1] + np.timedelta64(lead_time_h, 'h')]])

        # Interpolate from yearly to hourly
        co2_df.index = time_base
        co2_df.drop('year', axis=1, inplace=True)
        co2_df = pd.concat([co2_df, pd.DataFrame(index=inds, columns=['CO2'], dtype=float)]).sort_index()
        co2_df.loc[:, 'time'] = co2_df.index
        co2_df = co2_df.drop_duplicates(subset='time').drop('time', axis=1)
        co2_df.interpolate(inplace=True)
        self.co2_df = co2_df
        

    def getitem(self, t: list = None, lat: list = None, lon: list = None):
        if not isinstance(t, list):
            t = [t]
        co2_data = self.co2_df.loc[t, "CO2"].to_numpy()[:, np.newaxis, np.newaxis].astype(np.float32)
        co2_data = np.broadcast_to(co2_data, (len(t), len(lat), len(lon)))
        co2 = xr.DataArray(
                co2_data,
                coords={
                    "time": t,
                    "latitude": lat,
                    "longitude": lon
                },
                dims=["time", "latitude", "longitude"],
                name=self.co2_fullname
            )
        return co2


class WeatherBench2Raw(Dataset):
    def __init__(
        self, 
        name='era5',
        path: str = '/path/to/data/weatherbench2_original', 
        extended_path: dict = None, # key=variable full name, value=path to dataset that contains this extra variable
        extended_vars: list = None, # list of the variables (full name) from extended_dataset to include in original dataset
        stats_path: str = 'aurora/normalization_stats_1979_2021.json',
        inds = None, 
        str_task: str = 'forecast', 
        dict_vars: dict = None, 
        surf_vars: list[str] = None,
        static_vars: list[str] = None,
        atmos_vars: list[str] = None,
        atmos_levels = np.asarray([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int32),
        dict_stats: Optional[dict[str, tuple[float, float]]] = None, 
        co2_path: str = '/path/to/data/global_annual_CO2.csv',
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        lead_time_h: int = 6, # lead time in hours for the forecast task
        **kwargs,
    ):
        '''Defining surf_vars, static_vars, atmos_vars will overwrite dict_vars. 
        variable_name_mapping is ignored since all other datasets must conform to ERA5 convention.'''
        self.name = name
        self.path = path
        self.inds = inds
        self.d_ind_pairs = {}
        self.str_task = str_task
        self.dict_vars = dict_vars
        self.atmos_levels = atmos_levels
        if isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)
        self.dict_stats = dict_stats
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution
        self.ds = xr.open_zarr(path, chunks=None)
        self.dict_ds_extended = dict()
        self.lead_time_h = lead_time_h
        if extended_path:
            for var in extended_vars:
                self.dict_ds_extended[var] = xr.open_zarr(extended_path[var], chunks=None)
            # self.ds = self.ds.assign({var: self.dict_ds_extended[var] for var in extended_vars}) # remove because not lazy load

            # Get common time steps across the main dataset AND all extended datasets
            common_times = self.ds.time.to_index()
            for ext_ds in self.dict_ds_extended.values():
                common_times = common_times.intersection(ext_ds.time.to_index())

            # Subset `self.ds` to keep only common time steps
            self.ds = self.ds.sel(time=common_times)


        if len(self.ds.latitude) == 721:
            self.lat = self.ds.latitude.values[:-1] ## get only 720 out of the 721 latitudes
        else:
            self.lat = self.ds.latitude.values
        self.lon = self.ds.longitude.values
        self.lead_time_h = lead_time_h
        self.lead_time_x_hist = kwargs.pop('lead_time_x_hist', lead_time_h) # lead time in hours for the reconstruction task

        if extended_path:
            for k in extended_vars:
                self.dict_ds_extended[d_srf_abr2full[k]] = xr.open_zarr(extended_path[k], chunks=None)[d_srf_abr2full[k]].sel(time=common_times, latitude=self.lat, longitude=self.lon)
            # self.ds = self.ds.assign({var: self.dict_ds_extended[var] for var in extended_vars}) # remove because not lazy load

        if surf_vars is not None and 'co2' in surf_vars: # must be executed after defining self.lat and self.lon
            da_co2 = self._read_co2(co2_path)
            self.dict_ds_extended[d_srf_abr2full['co2']] = da_co2
            #self.ds = self.ds.assign({d_srf_abr2full['co2'] : da_co2}) # returns arrayMemoryError

        self.locations, self.scales = load_normalization_stats(stats_path)
        

        ###### HARDCODED CROP TO FIT TO MEM FOR NOW ########
        # self.lat = self.lat[:64]
        # self.lon = self.lon[:128]
        ###################################################
        
        if self.inds is None: ##assuming training set (<2018)
            self.inds = self.ds.time[self.ds.time.values < np.datetime64(datetime(2018, 1, 1),)]
            # inds_val = self.ds.time[(self.ds.time.values >= np.datetime64(datetime(2018, 1, 1),)) & (self.ds.time.values < np.datetime64(datetime(2019, 1, 1),))]
            # inds_test = self.ds.time[self.ds.time.values >= np.datetime64(datetime(2019, 1, 1),)]
        self.len_dataobj = len(self.inds) ## will be later overwritten.
        if self.dict_vars is None:
            self.dict_vars = {
                'surf_vars': ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"),
                'static_vars': ("land_sea_mask", "geopotential_at_surface", "soil_type"),
                'atmos_vars': ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity")
            }
        self.ds = self.ds.sel(level=self.atmos_levels)
        # self.ds = self.ds.sel(time=self.inds) #commenting out because this creates an issue for selecting certain subsets of indices. ##TODO: select time interval from [min(self.inds), max(self.inds)]
        self.ds = self.ds.sel(latitude=self.lat)
        self.ds = self.ds.sel(longitude=self.lon)
        
        if surf_vars is not None:
            self.surf_vars = dict()
            for k in surf_vars:
                if d_srf_abr2full[k] in self.ds.data_vars:
                    self.surf_vars[k] = self.ds[d_srf_abr2full[k]]
                else:
                    if k != 'co2':
                        self.surf_vars[k] = self.dict_ds_extended[d_srf_abr2full[k]]

            self.dict_vars['surf_vars'] = tuple([d_srf_abr2full[k] for k in surf_vars])
        else:
            self.surf_vars = {d_srf_full2abr[var]: self.ds[var] for var in self.dict_vars['surf_vars']} ## respecting the abbreviations from Aurora implementation for dict keys
        if static_vars is not None:
            self.static_vars = {k: self.ds[d_static_abr2full[k]] for k in static_vars}
            self.dict_vars['static_vars'] = tuple([d_static_abr2full[k] for k in static_vars])
        else:
            self.static_vars = {d_static_full2abr[var]: self.ds[var] for var in self.dict_vars['static_vars']}
        if atmos_vars is not None:
            self.atmos_vars = {k: self.ds[d_atmos_abr2full[k]] for k in atmos_vars}
            self.dict_vars['atmos_vars'] = tuple([d_atmos_abr2full[k] for k in atmos_vars])
        else:
            self.atmos_vars = {d_atmos_full2abr[var]: self.ds[var] for var in self.dict_vars['atmos_vars']}
        if self.str_task == '6h-forecast':
            self._prepare_inds_for_forecast(lead_time_h=6) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)
        elif self.str_task == 'forecast' and lead_time_h != 0:
            self._prepare_inds_for_forecast(lead_time_h=lead_time_h) ## assumes only forecast task for the dataloader (overwrites length of dataset obj.)
        elif self.str_task == 'reconst' or (self.str_task == 'forecast' and lead_time_h == 0):
            self._prepare_inds_for_reconst(lead_time_x_hist=self.lead_time_x_hist)

        if 'co2' in surf_vars: # must be executed after defining self.lat and self.lon
            self.co2_mapper = ScalarCO2Mapper(co2_path = co2_path,
                                  co2_fullname = d_srf_abr2full['co2'],
                                  lead_time_h = self.lead_time_h,
                                  inds = self.inds
                                  )
        
        # timestamp → position map (built once per worker) 
        times_ns = self.ds.time.values.astype("datetime64[ns]")
        self._time2idx = {int(t): idx for idx, t in enumerate(times_ns)}

        # static variables (cache once)
        self._static_cache = {
            d_static_full2abr[v]: np.asarray(self.static_vars[d_static_full2abr[v]].data)
            for v in self.dict_vars["static_vars"]
        }


    def __len__(self):
        return self.len_dataobj
    
    def _prepare_inds_for_reconst(self, lead_time_x_hist=6):
        # Determine if this is training or validation data based on time range
        first_time = np.min(self.inds)
        last_time = np.max(self.inds)
        
        # Create a dataset identifier based on time range
        dataset_id = f"{first_time.astype('datetime64[D]')}_{last_time.astype('datetime64[D]')}"
        cache_file = os.path.join('utils', f'reconst_pairs_{lead_time_x_hist}h_{dataset_id}_lenInds{(len(self.inds))}.pkl')
        print(f'cache_file: {cache_file}')
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.d_ind_pairs = pickle.load(f)
                    ind_pair_key = f"{lead_time_x_hist}h_reconst"
                    if ind_pair_key in self.d_ind_pairs:
                        self.len_dataobj = len(self.d_ind_pairs[ind_pair_key])
                        return
            except (EOFError, pickle.UnpicklingError):
                # Handle corrupt cache file
                print(f"Warning: Cache file {cache_file} is corrupted. Recreating...")
        
        # If cache doesn't exist or is invalid, compute from scratch
        # Compute the indices
        x_t1 = self.inds
        x_t0 = x_t1 - np.timedelta64(lead_time_x_hist, 'h')
        l_pairs = []
        for i in range(len(x_t1)):
            if x_t0[i] in self.ds.time and x_t1[i] in self.ds.time:
                l_pairs.append((x_t0[i], x_t1[i]))
        pairs = tuple(l_pairs)
        ind_pair_key = f"{lead_time_x_hist}h_reconst"
        self.d_ind_pairs[ind_pair_key] = pairs
        self.len_dataobj = len(pairs)
        
        # Save to cache for future use - only from rank 0
        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0
        
        if is_rank_zero:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.d_ind_pairs, f)
    
    def _prepare_inds_for_forecast(self, lead_time_h=6):
        # Determine if this is training or validation data based on time range
        first_time = np.min(self.inds)
        last_time = np.max(self.inds)
        
        # Create a dataset identifier based on time range
        dataset_id = f"{first_time.astype('datetime64[D]')}_{last_time.astype('datetime64[D]')}"
        cache_file = os.path.join('utils', f'forecast_pairs_{lead_time_h}h_{dataset_id}_lenInds{(len(self.inds))}.pkl')

        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.d_ind_pairs = pickle.load(f)
                    ind_pair_key = f"{lead_time_h}h_forecast"
                    if ind_pair_key in self.d_ind_pairs:
                        self.len_dataobj = len(self.d_ind_pairs[ind_pair_key])
                        print(f"Loaded {self.len_dataobj} pairs from cache for {ind_pair_key} with lead time {lead_time_h}h.")
                        return
            except (EOFError, pickle.UnpicklingError):
                # Handle corrupt cache file
                print(f"Warning: Cache file {cache_file} is corrupted. Recreating...")
        
        # If cache doesn't exist or is invalid, compute from scratch
        # Compute the indices
        x_t1 = self.inds
        x_t0 = x_t1 - np.timedelta64(lead_time_h, 'h')
        y_t = x_t1 + np.timedelta64(lead_time_h, 'h')
        l_pairs = []
        for i in range(len(x_t1)):
            if x_t0[i] in self.ds.time and x_t1[i] in self.ds.time and y_t[i] in self.ds.time:
                l_pairs.append((x_t0[i], x_t1[i], y_t[i]))
        pairs = tuple(l_pairs)
        ind_pair_key = f"{lead_time_h}h_forecast"
        self.d_ind_pairs[ind_pair_key] = pairs
        self.len_dataobj = len(pairs)
        print(f"Prepared {len(pairs)} pairs for {ind_pair_key} with lead time {lead_time_h}h.")

        # Save to cache for future use - only from rank 0
        is_rank_zero = int(os.environ.get("GLOBAL_RANK", "0")) == 0
        
        if is_rank_zero:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.d_ind_pairs, f)
    
    def __getitem__(self, idx):
        if self.str_task == '6h-forecast':
            return self._get_forecast(idx, lead_time_h=6)
        elif self.str_task == 'forecast' and self.lead_time_h != 0:
            return self._get_forecast(idx, lead_time_h=self.lead_time_h)
        elif self.str_task == 'reconst' or (self.str_task == 'forecast' and self.lead_time_h == 0):
            return self._get_reconst(idx, lead_time_x_hist=self.lead_time_x_hist)
        else:
            raise ValueError(f"Invalid task: {self.str_task}")
        

    def _get_forecast_sel(self, idx, lead_time_h=6):
        ## get the forecast data: Returns data of the form (x=(x-6h, x), y=(x+6h), t=x_t)
        if f"{lead_time_h}h_forecast" not in self.d_ind_pairs:
            raise ValueError(f"Invalid lead time: {lead_time_h}.")
        x_ind0, x_ind1, y_ind = self.d_ind_pairs[f"{lead_time_h}h_forecast"][idx]

        if d_srf_abr2full['co2'] in self.dict_vars['surf_vars']:
            ds_co2 = self.co2_mapper.getitem([x_ind0, x_ind1, y_ind], lat=self.lat, lon=self.lon)

        x_srf = dict()
        for var in self.dict_vars['surf_vars']:
            if var != d_srf_abr2full['co2']:
                x_srf[d_srf_full2abr[var]] = np.stack((self.surf_vars[d_srf_full2abr[var]].sel(time=x_ind0).values, self.surf_vars[d_srf_full2abr[var]].sel(time=x_ind1).values),axis=-3)
            else:
                x_srf[d_srf_full2abr[var]] = np.stack((ds_co2.sel(time=x_ind0).values, ds_co2.sel(time=x_ind1).values),axis=-3)
        x_static = {d_static_full2abr[var]: self.static_vars[d_static_full2abr[var]].values for var in self.dict_vars['static_vars']} ## cache this in a next iteration
        x_atmos = {d_atmos_full2abr[var]: np.stack((self.atmos_vars[d_atmos_full2abr[var]].sel(time=x_ind0).values, self.atmos_vars[d_atmos_full2abr[var]].sel(time=x_ind1).values),axis=-4) for var in self.dict_vars['atmos_vars']}
        
        y_srf = dict()
        for var in self.dict_vars['surf_vars']:
            if var != d_srf_abr2full['co2']:
                y_srf[d_srf_full2abr[var]] = self.surf_vars[d_srf_full2abr[var]].sel(time=y_ind).values
            else:
                y_srf[d_srf_full2abr[var]] = ds_co2.sel(time=y_ind).values
        y_static = x_static.copy() #[self.static_vars[k].values for k in self.dict_vars['static_vars']] ## cache this
        y_atmos = {d_atmos_full2abr[var]: self.atmos_vars[d_atmos_full2abr[var]].sel(time=y_ind).values for var in self.dict_vars['atmos_vars']}

        x_time = str(x_ind1)
        y_time = str(y_ind)
        return {
            'name': self.name,
            'x_srf': x_srf,
            'x_static':x_static,
            'x_atmos': x_atmos,
            'y_srf':y_srf,
            'y_static':y_static,
            'y_atmos':y_atmos,
            'x_time':x_time,
            'y_time':y_time,
            'lat': self.lat,
            'lon': self.lon,
            'atmos_levels':self.atmos_levels,
            'locations': self.locations,
            'scales': self.scales,
            'grid_resolution': self.grid_resolution,
            'is_global_observation': self.is_global_observation,
            'lead_time_seconds': timedelta(hours=self.lead_time_h).total_seconds(),
        }

    def _get_forecast(self, idx, lead_time_h: int = 6):
            """
            Fast version that uses positional indexing (isel) and minimises the
            number of xarray reads.
            """
            ind_key = f"{lead_time_h}h_forecast"
            if ind_key not in self.d_ind_pairs:
                raise ValueError(f"Invalid lead time: {lead_time_h}")

            x_ind0, x_ind1, y_ind = self.d_ind_pairs[ind_key][idx]

            try:
                i0 = self._time2idx[int(x_ind0.astype("datetime64[ns]").astype(int))]
                i1 = self._time2idx[int(x_ind1.astype("datetime64[ns]").astype(int))]
                iy = self._time2idx[int(y_ind.astype("datetime64[ns]").astype(int))]
            except KeyError as e:
                missing_dt64 = np.datetime64(e.args[0], "ns")
                missing_ts = np.datetime_as_string(missing_dt64, unit="s")
                available_range = (
                    np.datetime64(min(self._time2idx.keys()), "ns"),
                    np.datetime64(max(self._time2idx.keys()), "ns")
                )
                raise KeyError(
                    f"Timestamp {missing_ts} not found. "
                    f"Available range: {available_range[0]} → {available_range[1]}"
                ) from None
            t_idx = [i0, i1, iy]                        

            # surface variables - batch load all vars at once 
            surf_vars_list = list(self.dict_vars["surf_vars"])
            surf_abr_list = [d_srf_full2abr[v] for v in surf_vars_list]

            # Load all surface variables in one operation
            surf_data = np.stack([
                np.asarray(self.surf_vars[abr].isel(time=t_idx).data)
                for abr in surf_abr_list
            ])  # shape: (N_vars, 3, H, W)

            # Split into x and y dictionaries
            x_srf = {
                abr: surf_data[i, :2]  # (2, H, W)
                for i, abr in enumerate(surf_abr_list)
            }
            y_srf = {
                abr: surf_data[i, 2]   # (H, W)
                for i, abr in enumerate(surf_abr_list)
            }

            # CO₂ (optional)
            if d_srf_abr2full["co2"] in self.dict_vars["surf_vars"]:
                # lazily create the CO₂ DataArray for the three timesteps
                ds_co2 = self.co2_mapper.getitem([x_ind0, x_ind1, y_ind],
                                                 lat=self.lat, lon=self.lon)
                x_srf["co2"] = np.stack(
                    (ds_co2.sel(time=x_ind0).values,
                     ds_co2.sel(time=x_ind1).values), axis=-3
                )                                                                     # (2,H,W)
                y_srf["co2"] = ds_co2.sel(time=y_ind).values                           # (H,W)

            x_static = self._static_cache
            y_static = self._static_cache

            # atmospheric variables - batch load all vars at once 
            atmos_vars_list = list(self.dict_vars["atmos_vars"])
            atmos_abr_list = [d_atmos_full2abr[v] for v in atmos_vars_list]
            
            # Load all atmospheric variables in one operation
            atmos_data = np.stack([
                np.asarray(self.atmos_vars[abr].isel(time=t_idx).data)
                for abr in atmos_abr_list
            ])  # shape: (N_vars, 3, L, H, W)

            # Split into x and y dictionaries
            x_atmos = {
                abr: atmos_data[i, :2]  # (2, L, H, W)
                for i, abr in enumerate(atmos_abr_list)
            }
            y_atmos = {
                abr: atmos_data[i, 2]   # (L, H, W)
                for i, abr in enumerate(atmos_abr_list)
            }

            return {
                'name': self.name,
                "x_srf": x_srf,
                "x_static": x_static,
                "x_atmos": x_atmos,
                "y_srf": y_srf,
                "y_static": y_static,
                "y_atmos": y_atmos,
                "x_time": str(x_ind1),
                "y_time": str(y_ind),
                "lat": self.lat,
                "lon": self.lon,
                "atmos_levels": self.atmos_levels,
                "locations": self.locations,
                "scales": self.scales,
                "grid_resolution": self.grid_resolution,
                "is_global_observation": self.is_global_observation,
                "lead_time_seconds": timedelta(hours=self.lead_time_h).total_seconds(),
            }

        
    def _get_reconst(self, idx, lead_time_x_hist=6):
        ## get the reconstruction data
        ## Returns data of the form (x=(x-6h, x), y=(x), t=x_t)
        x_ind0, x_ind1 = self.d_ind_pairs[f"{lead_time_x_hist}h_reconst"][idx]
        x_srf = {d_srf_full2abr[k]: np.stack((self.surf_vars[d_srf_full2abr[k]].sel(time=x_ind0).values, self.surf_vars[d_srf_full2abr[k]].sel(time=x_ind1).values),axis=-3) for k in self.dict_vars['surf_vars']}
        if self.co2 is not None:
            sample_year = int(x_ind0.astype('datetime64[Y]').astype(int) + 1970)
            co2 = self.year_to_co2(sample_year)
            x_srf['co2'] = np.full_like(next(iter(x_srf.values())), co2)

        x_static = {d_static_full2abr[k]: self.static_vars[d_static_full2abr[k]].values for k in self.dict_vars['static_vars']} ## cache this in a next iteration
        x_atmos = {d_atmos_full2abr[k]: np.stack((self.atmos_vars[d_atmos_full2abr[k]].sel(time=x_ind0).values, self.atmos_vars[d_atmos_full2abr[k]].sel(time=x_ind1).values),axis=-4) for k in self.dict_vars['atmos_vars']}
        y_srf = {d_srf_full2abr[k]: x_srf[d_srf_full2abr[k]][..., -1, :, :] for k in self.dict_vars['surf_vars']}
        y_static = x_static.copy() #[self.static_vars[k].values for k in self.dict_vars['static_vars']] ## cache this
        y_atmos = {d_atmos_full2abr[k]: x_atmos[d_atmos_full2abr[k]][..., -1, :, :, :] for k in self.dict_vars['atmos_vars']}
        # x_ind0 = str(x_ind0)
        x_time = str(x_ind1)
        y_time = x_time
        return {
            'x_srf': x_srf,
            'x_static':x_static,
            'x_atmos': x_atmos,
            'y_srf':y_srf,
            'y_static':y_static,
            'y_atmos':y_atmos,
            'x_time':x_time,
            'y_time':y_time,
            'lat': self.lat,
            'lon': self.lon,
            'atmos_levels':self.atmos_levels,
            'locations': self.locations,
            'scales': self.scales,
            'grid_resolution': self.grid_resolution,
            'is_global_observation': self.is_global_observation,
            'lead_time_seconds': 0.,
        }
    
class WB2Masked(WeatherBench2Raw):
    def __init__(self, str_masking_task=None, **kwargs):
        """
        A subclass of WeatherBench2Raw that will do random Masking based on str_task.
        It inherits all properties and methods from WeatherBench2Raw.
        Args:
            str_task (str): The task to perform, e.g., 'spatial-unmask', 
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.str_masking_task = str_masking_task
        self.prob_mask_var = kwargs.get('prob_mask_var', 0.2)  # Probability of masking a variable
        self.prob_mask_spatial = kwargs.get('prob_mask_spatial', 0.5)  # Probability of masking spatial locations
        self.prob_mask_vertical = kwargs.get('prob_mask_vertical', 0.3)  # Probability of masking vertical levels
        self.patch_size = kwargs.get('tokenization_patch_size', 4)  # Size of the patches to mask
        self.prng = None
        
    def _init_prng(self):
        """
        Initializes a separate pseudo-random number generator (PRNG) for each worker.
        
        This method ensures that each worker in a multi-worker data loading setup has its own
        PRNG instance, initialized with a unique seed. The seed is derived from `torch.initial_seed()`,
        which is specific to each worker. This approach guarantees reproducibility and prevents
        workers from sharing the same random number sequence.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            seed = torch.initial_seed() % 2**32
        else:
            # In worker process
            seed = torch.initial_seed() % 2**32
        self.prng = np.random.RandomState(seed)
        
    def __getitem__(self, idx):
        """
        Returns a dictionary containing masked data based on the specified str_masking_task.
        """
        d_sample = super().__getitem__(idx)  # Call the parent class method to ensure all properties are set
        if self.prng is None:
            self._init_prng()
        if self.str_masking_task == 'spatial-unmask':
            d_sample = self._get_spatial_unmask(d_sample)
        elif self.str_masking_task == 'vertical-unmask':
            d_sample = self._get_vertical_unmask(d_sample)
        elif self.str_masking_task == 'variable-unmask':
            d_sample = self._get_variable_unmask(d_sample)
        else:
            pass # No masking applied, return the sample as is
        return d_sample
        
    def _get_spatial_unmask(self, d_sample ):
        """
        Applies spatial masking to surface & atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable.
        prob_mask_spatial: The amount of masking being applied across lat/lon for a variable that will be masked.
        """
        for k in d_sample['x_srf'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Mask the variable. Mask entire patches, assuming patch size of self.patch_size x self.patch_size.
                patched_res = np.array(np.asarray(d_sample['x_srf'][k].shape[1:])/self.patch_size).astype(int) # shape [lat/self.patch_size, lon/self.patch_size]
                mask = self.prng.rand(*patched_res) < self.prob_mask_spatial
                mask = mask.repeat(self.patch_size, axis=0).repeat(self.patch_size, axis=1) # shape [lat, lon]
                d_sample['x_srf'][k][:,mask] = np.nan #broadcast to time dim as well (dim 0)
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Mask the variable. Mask entire patches, assuming patch size of self.patch_size x self.patch_size.
                patched_res = np.concatenate([[d_sample['x_atmos'][k].shape[1]], np.asarray(d_sample['x_atmos'][k].shape[2:])/self.patch_size]).astype(int) #get [atmos-levels, lat/self.patch_size, lon/self.patch_size] shape
                mask = self.prng.rand(*patched_res) < self.prob_mask_spatial
                # rescale mask to match the spatial dimensions of the variable
                mask = mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2) #extend back to [atmos-levels, lat, lon] 
                d_sample['x_atmos'][k][:,mask] = np.nan # broadcast to time dim as well (dim 0)
        return d_sample
                
    def _get_vertical_unmask(self, d_sample):
        """
        Applies vertical masking to the atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable.
        prob_mask_vertical: chances of masking being applied to a given atmospheric level.
        """
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # pick how many vertical levels to mask
                num_levels = d_sample['x_atmos'][k].shape[1]
                mask = self.prng.rand(num_levels) < self.prob_mask_vertical
                # apply the mask to the variable
                d_sample['x_atmos'][k][:, mask, :, :] = np.nan
        return d_sample
    
    def _get_variable_unmask(self, d_sample):
        """
        Applies variable masking to the surface & atmospheric variables in the sample.
        prob_mask_var: Chances of masking being applied to a variable as a whole.
        """
        for k in d_sample['x_srf'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Mask the variable
                d_sample['x_srf'][k][:] = np.nan
        for k in d_sample['x_atmos'].keys():
            if self.prng.rand() < self.prob_mask_var:
                # Mask the variable
                d_sample['x_atmos'][k][:] = np.nan
        return d_sample
    
    

class CMIP6ClimaXDataset(Dataset):
    """
    A PyTorch IterableDataset for loading and iterating over CMIP6 climate data stored in .npz files.
    This Class is only for forcastign and assumes timesteps are always 6 hours.
    Args:
        path (str): Path to the directory containing the .npz files.
        start_idx (int, optional): Starting index for the files to be loaded. 
        end_idx (int, optional): Ending index for the files to be loaded.
        surf_vars (list[str], optional): List of surface variables to be loaded (based on the names in original dataset).
        static_vars (list[str], optional): List of static variables to be loaded (based on the names in original dataset).
        atmos_vars (list[str], optional): List of atmospheric variables to be loaded (based on the names in original dataset).
        variable_name_mapping (dict, optional): A dictionary to map the variables names from original dataset to a uniform naming convension (dict(zip(source_var_naming, ESFM_var_naming))).
        atmos_levels (list[int], optional): List of atmospheric levels to be loaded.
        lat (int, optional): Number of latitude points.
        lon (int, optional): Number of longitude points.
        shuffle (bool, optional): Whether to shuffle the file list.
    Methods:
        find_first_times_key(keys):
            Finds the first key in the provided keys that ends with '_times'.
        convert_to_strtime(time):
            Converts a time object to a string representation.
        iterate_over_single_chunk(path):
            Iterates over a single .npz file and yields data dictionaries.
        __iter__():
            Iterates over the dataset, yielding data dictionaries.
    """

    def __init__(
        self,
        path,
        name='cmip6',
        start_idx: int = 0,
        end_idx: int = None,
        surf_vars: list[str] = ['psl'],
        static_vars: list[str] = [],
        atmos_vars: list[str] = ['va', 'ta', 'ua', 'zg'],
        variable_name_mapping: dict = None,
        atmos_levels: list[int] = [50, 850, 500, 600, 250, 700, 925],
        shuffle: bool = False,
        sample_per_chunk=None,
        str_task: str='6h-forecast',
        wb2_path: str=None,
        is_global_observation: bool = True,
        grid_resolution: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name
        data_dir = os.path.join(path, 'train/*.npz')
        self.lon = np.load(os.path.join(path, 'lon.npy'))
        self.lat = np.flip(np.sort(np.load(os.path.join(path, 'lat.npy'))))
        self.file_list = natsorted(glob.glob(data_dir))
        assert len(self.file_list), f'There is no .npz files under: {data_dir}'
        if sample_per_chunk is None:
            self.sample_per_chunk = next(iter(np.load(self.file_list[0]).values())).shape[0]
        else:
            self.sample_per_chunk = sample_per_chunk
        self.sample_last_chunk = next(iter(np.load(self.file_list[-1]).values())).shape[0]
        self.str_task = str_task
        if self.str_task == '6h-forecast':
            # ignoring the last two timesteps for forcasting purposes (x_(n-2), x_(n-1))-> x_n
            self.sample_per_chunk -= 2 
            self.sample_last_chunk -= 2
        if variable_name_mapping is None:
            # variable_name_mapping is used to map the name of the variables
            self.variable_name_mapping = {k: v for k, v in zip(
                ['ta', 'ua', 'va', 'zg', 'hus', 'tas', 'uas', 'vas', 'psl'],
                ['t', 'u', 'v', 'z', 'q', '2t', '10u', '10v', 'msl']
            )}
        else:
            self.variable_name_mapping = variable_name_mapping

        self.num_samples = self.sample_per_chunk * (len(self.file_list) - 1) + self.sample_last_chunk
        if end_idx is None:
            end_idx = self.num_samples
        assert start_idx < self.num_samples, f"start_idx {start_idx} is out of range for total samples {self.num_samples}"
        assert end_idx <= self.num_samples, f"end_idx {end_idx} is out of range for total samples {self.num_samples}"
        assert start_idx < end_idx, f"start_idx {start_idx} should be smaller than end_idx {end_idx}"

        self.indices = list(range(start_idx, end_idx))
        self.surf_vars = surf_vars
        self.static_vars = static_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels
        if isinstance(self.atmos_levels, list):
            self.atmos_levels = np.asarray(self.atmos_levels, dtype=np.int32)
        self.shuffle = shuffle
        self.is_global_observation = is_global_observation
        self.grid_resolution = grid_resolution

        # mapping the cmip6 variable names to ERA5 names
        self.locations, self.scales = load_normalization_stats(
            path, variable_name_mapping=variable_name_mapping
        )
        if wb2_path and static_vars:
            wb2 = xr.open_zarr(wb2_path) # TODO: check if some CMIP6 data come from different earth models, in this case it doesn't make sense to use era5 static variables

            # Interpolate static variables to match the latitude and longitude of the CMIP6 dataset
            self.static_vars = {
                var: wb2[d_static_abr2full[var]]
                .interp(latitude=self.lat, longitude=self.lon, method='nearest').values
                for var in static_vars
            }
        else:
            self.static_vars = {}

    def __len__(self):
        return len(self.indices)
    
    def find_first_times_key(self, keys):
        for key in keys:
            if key.endswith('_times'):
                return key
        return None

    def convert_to_strtime(self, time):
        if isinstance(time, (cftime.DatetimeNoLeap, datetime)):
            return time.strftime('%Y-%m-%dT%H:%M:%S.%f')
        return time 

    def _get_unmask(self):
        raise NotImplementedError("The method is not implemented yet.")

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if self.str_task == '6h-forecast':
            return self._get_forecast(actual_idx)
        elif self.str_task == 'unmask':
            return self._get_unmask(actual_idx)
        else:
            raise ValueError(f"Invalid task: {self.str_task}")

    def _get_forecast(self, idx):
        i = idx % self.sample_per_chunk # effective index in chunk
        file_idx = int(idx / self.sample_per_chunk)

        data = mmnpz.load(self.file_list[file_idx])
        times = data['times']
        data_dict = {
            'name': self.name,
            'x_time': times[i+1].astype('datetime64[s]').item().strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'y_time': times[i+2].astype('datetime64[s]').item().strftime('%Y-%m-%dT%H:%M:%S.%f'), 
            'x_srf': {}, 
            'x_atmos': {}, 
            'x_static': {},
            'y_srf': {}, 
            'y_atmos': {},
            'y_static': {}, 
            'lat': self.lat.copy(), 
            'lon': self.lon.copy(),
            'atmos_levels': self.atmos_levels,
            'locations': self.locations,
            'scales': self.scales,
            'grid_resolution': self.grid_resolution,
            'is_global_observation': self.is_global_observation,
        }

        for var in self.atmos_vars:
            # use variable_name_mapping to rename the variable if provided
            var_name = self.variable_name_mapping.get(var, var)

            x_data, y_data = [], []
            for level in self.atmos_levels:
                key = f'{var}_{level}'

                if key in data.keys():
                    x_data.append(data[key][i:i+2])
                    y_data.append(data[key][i+2:i+3])
                else:
                    x_data.append(np.full((2, 1, len(self.lat), len(self.lon)), np.nan))
                    y_data.append(np.full((1, 1, len(self.lat), len(self.lon)), np.nan))

            data_dict['x_atmos'][var_name] = np.concatenate(x_data, axis=1)
            data_dict['y_atmos'][var_name] = np.concatenate(y_data, axis=1)

        for var in self.surf_vars:
            # use variable_name_mapping to rename the variable if provided
            var_name = self.variable_name_mapping.get(var, var)
            data_dict['x_srf'][var_name] = data[var][i:i+2]
            data_dict['y_srf'][var_name] = data[var][i+2:i+3]

        # Assumed CMIP6 data doesn't have any static variables, and all static variables are from WB2 dataset
        data_dict['x_static'] = self.static_vars
        data_dict['y_static'] = self.static_vars

        if not data_dict['x_static']  and not data_dict['x_srf']:
            # Add a dummy tensor of nans if there are no 2d features
            B, C, H, W = next(iter(data_dict['x_atmos'].values())).shape
            data_dict['x_srf']['<N/A>'] = torch.full((B, H, W), float('nan')) 

        return data_dict


class CombinedDataLoader:
    def __init__(self, datasets, batch_sizes, *args, **kwargs):
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.shuffle = kwargs.get('shuffle')

        self.loaders = [
            DataLoader(dataset, batch_size=batch_size, *args, **kwargs)
            for dataset, batch_size in zip(datasets, batch_sizes)
        ]

    def __iter__(self):
        if  not self.shuffle:
            # Use sequential for validation or when shuffle=False
            return self._generate_sequential_batches()
        return self._generate_random_batches()

    def _generate_sequential_batches(self):
        """Sequential iteration for validation or when shuffle=False"""
        for loader in self.loaders:
            for batch in loader:
                yield batch


    def _generate_random_batches(self):
        """
        Yields batches from randomly selected datasets.
        """
        iterators = [iter(loader) for loader in self.loaders]
        while True:
            # Randomly select a dataset
            dataset_idx = random.randint(0, len(self.loaders) - 1)
            try:
                # Yield a batch from the selected dataset
                batch = next(iterators[dataset_idx])
                yield batch
            except StopIteration:
                # If one dataset is exhausted, remove it from the pool
                del iterators[dataset_idx]
                del self.loaders[dataset_idx]
                del self.batch_sizes[dataset_idx]
                if len(iterators) == 0:
                    break  # All datasets are exhausted
                
    # def _generate_batches(self):
    #     for loader in self.loaders:
    #         for batch in loader:
    #             yield batch

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)

    def set_epoch(self, epoch):
        """Required for proper distributed training"""
        if torch.distributed.is_initialized():
            for loader in self.loaders:
                if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)


class DatasetMixer:
    def __init__(
        self,
        datasets: List[Dataset],
        subset_sizes: List[int],
        batch_sizes: List[int],
        *args, **kwargs
    ):
        """
        Args:
            datasets (List[Dataset]): List of datasets to mix.
            subset_sizes (List[int]): List of subset sizes for each dataset.
            batch_sizes (List[int]): List of batch sizes for each dataset.
        """
        msg = "Length of datasets, subset_sizes, and batch_sizes must be the same."
        assert len(datasets) == len(subset_sizes) == len(batch_sizes), msg

        self.datasets = datasets
        self.subset_sizes = subset_sizes
        self.batch_sizes = batch_sizes

        # Create subsets
        self.subsets = []
        for dataset, subset_size, batch_size in zip(datasets, subset_sizes, batch_sizes):
             # Generate random indices
            indices = list(range(len(dataset)))
            np.random.shuffle(indices)

            for i in range(0, len(indices), subset_size):
                subset_indices = indices[i:i + subset_size]
                subset = DataLoader(
                    Subset(dataset, subset_indices),
                    batch_size=batch_size,
                    *args,
                    **kwargs
                )
                self.subsets.append(subset)

        # Shuffle the subsets
        np.random.shuffle(self.subsets)

    def __len__(self):
        return sum(len(loader) for loader in self.subsets)

    def __iter__(self):
        for subset in self.subsets:
            for b, batch in enumerate(subset):
                if b == len(subset) - 1:
                    batch['sync'] = True

                yield batch

    def set_epoch(self, epoch):
        """Required for proper distributed training"""
        if torch.distributed.is_initialized():
            for loader in self.loaders:
                if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)


class CombinedDataset(Dataset):
    def __init__(self, datasets, batch_sizes, shuffle=False):
        super().__init__()
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.lengths = [int(len(datasets[i]) // batch_sizes[i]) for i in range(len(datasets))]
        self.length = sum(self.lengths)
        self.batch_inds = None
        self.linear_inds = None
        self._populate_batch_inds()
        
    def _populate_batch_inds(self):
        self.batch_inds = []
        self.linear_inds = []
        for i in range(len(self.datasets)):
            bs = self.batch_sizes[i]
            inds = np.arange(int(self.lengths[i]*self.batch_sizes[i])) #will drop the last non-complete batch.
            if self.shuffle: # shuffle the order of indices within a dataset
                np.random.shuffle(inds)
            inds = inds.reshape(-1, bs) #  shape: (n_batches, bs)
            self.batch_inds.append(inds) #shape: (n_datasets, n_batches, bs)
            num_batches = len(inds)
            # now create a list of (i, j) pairs for each batch
            for j in range(num_batches):
                self.linear_inds.append((i, j))
        
        if self.shuffle: ## shuffles the order of the batches across datasets.
            np.random.shuffle(self.linear_inds)
            
    def _reshuffle_batch_inds(self):
        for i in range(len(self.batch_inds)):
            inds = self.batch_inds[i]
            n,b = inds.shape
            inds = inds.reshape(-1)
            np.random.shuffle(inds)
            self.batch_inds[i] = inds.reshape(n,b)
            
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        dataset_idx, batch_idx = self.linear_inds[index]
        inds_batch = self.batch_inds[dataset_idx][batch_idx]
        for i, idx in enumerate(inds_batch):
            batch_ = self.datasets[dataset_idx][idx]
            if i == 0:
                batch = {
                    'x_time': [batch_['x_time']],
                    'y_time': [batch_['y_time']],
                    'x_srf': {k: np.expand_dims(v, 0) for k, v in batch_['x_srf'].items()},
                    'x_static': {k: np.expand_dims(v, 0) for k, v in batch_['x_static'].items()},
                    'x_atmos': {k: np.expand_dims(v, 0) for k, v in batch_['x_atmos'].items()},
                    'y_srf': {k: np.expand_dims(v, 0) for k, v in batch_['y_srf'].items()},
                    'y_static': {k: np.expand_dims(v, 0) for k, v in batch_['y_static'].items()},
                    'y_atmos': {k: np.expand_dims(v, 0) for k, v in batch_['y_atmos'].items()},
                    'lat': [batch_['lat']],
                    'lon': [batch_['lon']],
                    'atmos_levels': np.expand_dims(batch_['atmos_levels'], 0),
                    'locations': {k: np.expand_dims(v, 0) for k, v in batch_['locations'].items()},
                    'scales': {k: np.expand_dims(v, 0) for k, v in batch_['scales'].items()},
                    'grid_resolution': np.expand_dims(batch_['grid_resolution'], 0),
                    'is_global_observation': np.expand_dims(batch_['is_global_observation'],0),
                }
            else:
                batch['x_time'].append(batch_['x_time'])
                batch['y_time'].append(batch_['y_time'])
                for k in batch_['x_srf']:
                    batch['x_srf'][k] = np.concatenate((batch['x_srf'][k], np.expand_dims(batch_['x_srf'][k], 0)), axis=0)
                for k in batch_['x_static']:
                    batch['x_static'][k] = np.concatenate((batch['x_static'][k], np.expand_dims(batch_['x_static'][k], 0)), axis=0)
                for k in batch_['x_atmos']:
                    batch['x_atmos'][k] = np.concatenate((batch['x_atmos'][k], np.expand_dims(batch_['x_atmos'][k], 0)), axis=0)
                for k in batch_['y_srf']:
                    batch['y_srf'][k] = np.concatenate((batch['y_srf'][k], np.expand_dims(batch_['y_srf'][k], 0)), axis=0)
                for k in batch_['y_static']:
                    batch['y_static'][k] = np.concatenate((batch['y_static'][k], np.expand_dims(batch_['y_static'][k], 0)), axis=0)
                for k in batch_['y_atmos']:
                    batch['y_atmos'][k] = np.concatenate((batch['y_atmos'][k], np.expand_dims(batch_['y_atmos'][k], 0)), axis=0)
                batch['lat'].append(batch_['lat'])
                batch['lon'].append(batch_['lon'])
                batch['atmos_levels'] = np.concatenate((batch['atmos_levels'], np.expand_dims(batch_['atmos_levels'], 0)), axis=0)
                for k in batch_['locations']:
                    batch['locations'][k] = np.concatenate((batch['locations'][k], np.expand_dims(batch_['locations'][k], 0)), axis=0)
                for k in batch_['scales']:
                    batch['scales'][k] = np.concatenate((batch['scales'][k], np.expand_dims(batch_['scales'][k], 0)), axis=0)
                batch['grid_resolution'] = np.concatenate((batch['grid_resolution'], np.expand_dims(batch_['grid_resolution'], 0)), axis=0)
                batch['is_global_observation'] = np.concatenate((batch['is_global_observation'], np.expand_dims(batch_['is_global_observation'], 0)), axis=0)
                
        return batch


class StatefulMultiDatasetLoader(DataLoader):
    """
    A StatefulDataLoader that cycles through multiple datasets with different batch sizes and samplers.
    Switches between datasets every n steps and supports checkpointing through state_dict and load_state_dict.
    
    Args:
        datasets (List[Dataset]): List of PyTorch datasets to load from
        batch_sizes (List[int]): Batch size for each dataset
        samplers (List[Optional[torch.utils.data.Sampler]]): Sampler for each dataset
        switch_steps (int): Number of steps before switching to the next dataset
        collate_fns (List[Optional[Callable]]): Collate function for each dataset
        num_workers (int): Number of workers for all dataloaders
        pin_memory (bool): Whether to pin memory for all dataloaders
        drop_last (bool): Whether to drop the last incomplete batch for all dataloaders
        timeout (float): Timeout value for all dataloaders
        worker_init_fn (Optional[Callable]): Worker init function for all dataloaders
        multiprocessing_context (Optional[str]): Multiprocessing context for all dataloaders
        prefetch_factor (int): Prefetch factor for all dataloaders
        persistent_workers (bool): Whether to use persistent workers for all dataloaders
        snapshot_every_n_steps (int): How often to snapshot the state for checkpointing
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        samplers: List[Optional[torch.utils.data.Sampler]] = None,
        switch_steps: int = 1,
        collate_fns: List[Optional[Callable]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        snapshot_every_n_steps: Optional[int] = 1,
    ):
        # Validate inputs
        assert len(datasets) > 0, "Must provide at least one dataset"
        assert len(batch_sizes) == len(datasets), "Must provide a batch size for each dataset"
        
        if samplers is None:
            samplers = [None] * len(datasets)
        assert len(samplers) == len(datasets), "Must provide a sampler for each dataset"
        
        if collate_fns is None:
            collate_fns = [None] * len(datasets)
        assert len(collate_fns) == len(datasets), "Must provide a collate_fn for each dataset"
        
        # Initialize the parent DataLoader class
        super().__init__(
            dataset=None,  # No single dataset; this is a multi-dataset loader
            batch_size=1,  # Placeholder; actual batch sizes are handled internally
            shuffle=False,  # Shuffle is handled internally
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
        
        # Store parameters specific to multi-dataset functionality
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.switch_steps = switch_steps
        self.collate_fns = collate_fns
        self.common_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "pin_memory_device": pin_memory_device,
            "in_order": in_order,
            "snapshot_every_n_steps": snapshot_every_n_steps,
        }
        
        # Create individual stateful dataloaders
        self.dataloaders = []
        for i, dataset in enumerate(self.datasets):
            dataloader = StatefulDataLoader(
                dataset=dataset,
                batch_size=self.batch_sizes[i],
                shuffle=False,  # We'll use the provided samplers
                sampler=self.samplers[i],
                collate_fn=self.collate_fns[i],
                **self.common_kwargs
            )
            self.dataloaders.append(dataloader)
        
        # Length is the sum of all dataset lengths (in batches)
        self.lengths = [len(dataloader) for dataloader in self.dataloaders]
        self._length = sum(self.lengths)
        
        # Keep track of current dataset index and step counter
        self.current_dataset_idx = 0
        self.step_counter = 0
        self.current_iterators = None
        
    def __iter__(self):
        # But we're going to override with our custom multi-dataset iteration logic
        # Create iterators for each dataloader if they don't exist
        if self.current_iterators is None:
            self.current_iterators = [iter(dataloader) for dataloader in self.dataloaders]
        
        # Initialize counters if this is a fresh iterator
        dataset_idx = self.current_dataset_idx
        step_counter = self.step_counter
        
        # Loop until all dataloaders are exhausted
        while True:
            try:
                # Switch dataset if necessary
                if step_counter % self.switch_steps == 0:
                    dataset_idx = (dataset_idx + 1) % len(self.datasets)
                    self.current_dataset_idx = dataset_idx
                
                # Try to get the next batch from the current dataset
                batch = next(self.current_iterators[dataset_idx])
                yield batch
                
                # Increment step counter
                step_counter += 1
                self.step_counter = step_counter
                
            except StopIteration:
                # Replace the exhausted iterator
                self.current_iterators[dataset_idx] = iter(self.dataloaders[dataset_idx])
                
                # If all datasets are exhausted, break the loop
                if all(not bool(len(list(itertools.islice(iter(dl), 1)))) for dl in self.dataloaders):
                    break
    
    def __len__(self):
        return self._length
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the state of all dataloaders and the multi-dataset cycling state.
        """
        # Get the state of each individual dataloader
        dataloader_states = [dataloader.state_dict() for dataloader in self.dataloaders]
        
        # Combine with the multi-dataset specific state
        state = {
            "dataloader_states": dataloader_states,
            "current_dataset_idx": self.current_dataset_idx,
            "step_counter": self.step_counter,
        }
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Loads the state from a previously saved state_dict.
        """
        if state_dict == {}:
            return
            
        # Load the state for each individual dataloader
        dataloader_states = state_dict.get("dataloader_states", [])
        for i, dataloader_state in enumerate(dataloader_states):
            if i < len(self.dataloaders):
                self.dataloaders[i].load_state_dict(dataloader_state)
        
        # Set the multi-dataset specific state
        self.current_dataset_idx = state_dict.get("current_dataset_idx", 0)
        self.step_counter = state_dict.get("step_counter", 0)
        
        # Reset the iterators to force rebuilding them on next __iter__ call
        self.current_iterators = None
        
        # Tell the parent StatefulDataLoader to reset its iterator
        self._iterator = None
    
    def get_current_dataset_index(self):
        """Return the index of the currently active dataset"""
        return self.current_dataset_idx
    
    def reset(self):
        """Reset the dataloader to start from the first dataset"""
        self.current_dataset_idx = 0
        self.step_counter = 0
        self.current_iterators = None
        self._iterator = None

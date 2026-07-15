import torch
import numpy as np  


def transform_data(data, short_name, eps=1e-5, direct=True): # transformation follows [Rasp 2020], also used in FourCastNet
    if np.isin(short_name, ["tp", "tp_mswep", "r"]):
        if direct:
            if isinstance(data, torch.Tensor):
                return torch.log(1 + data/eps)
            else:
                return np.log(1 + data/eps)
        else:
            if isinstance(data, torch.Tensor):
                return eps*(torch.exp(data) - 1)
            else:
                return eps*(np.exp(data) - 1)
            
    elif np.isin(short_name, ["pe", "e"]):
        if direct:
            return -5e3*data
        else:
            return data/(-5e3)
        
    elif np.isin(short_name, ["tws_gou", "tws_itsg"]):
        if direct:
            return 1e-2*data
        else:
            return 1e2*data
        
    else:
        return data


def accumulate_data(ds, x_ind, short_name, lead_time_h=6):
    if np.isin(short_name, ["tp_mswep"]):
        return ds[short_name].sel(time=slice(x_ind - np.timedelta64(lead_time_h-1, 'h'), x_ind)).sum(dim='time').values
    else:
        return ds[short_name].sel(time=x_ind).values
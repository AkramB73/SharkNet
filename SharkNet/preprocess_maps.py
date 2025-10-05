import os
import numpy as np
import xarray as xr
from tqdm import tqdm

# Const. for geostrophic velocity calculation
G = 9.81
OMEGA = 7.2921e-5
DEG2RAD = np.pi / 180.0
DEG_LAT_TO_M = 111132.0

def align_datasets(dss, target_res=None, bbox=None):
    def get_coord_name(ds, name_part):
        for coord in ds.coords:
            if name_part in str(coord).lower():
                return str(coord)
        raise ValueError(f"Naming convention for lat/lon not found in dataset")

    lat_names = [get_coord_name(ds, 'lat') for ds in dss]
    lon_names = [get_coord_name(ds, 'lon') for ds in dss]

    if bbox is None:
        lat_min = max(ds[lat_name].min().item() for ds, lat_name in zip(dss, lat_names))
        lat_max = min(ds[lat_name].max().item() for ds, lat_name in zip(dss, lat_names))
        lon_min = max(ds[lon_name].min().item() for ds, lon_name in zip(dss, lon_names))
        lon_max = min(ds[lon_name].max().item() for ds, lon_name in zip(dss, lon_names))
        bbox = (lat_min, lat_max, lon_min, lon_max)

    if target_res is None:
        target_res = min(abs(ds[lat_name].values[1] - ds[lat_name].values[0]) for ds, lat_name in zip(dss, lat_names))

    lat_min, lat_max, lon_min, lon_max = bbox
    new_lat = np.arange(lat_min, lat_max, target_res)
    new_lon = np.arange(lon_min, lon_max, target_res)

    print(f"Aligning to grid: {len(new_lat)}x{len(new_lon)} at {target_res:.3f} deg resolution.")
    
    aligned_das = []
    for ds, lat_name, lon_name in zip(dss, lat_names, lon_names):
        renamed_ds = ds.rename({lat_name: 'lat', lon_name: 'lon'})
        interp_da = renamed_ds.interp(lat=new_lat, lon=new_lon, method='nearest', kwargs={"fill_value": "extrapolate"})
        aligned_das.append(interp_da)
        
    return aligned_das

def compute_geostrophic_velocity_vectorized(eta: xr.DataArray):
    lat_rad = eta.lat * DEG2RAD
    f = 2 * OMEGA * np.sin(lat_rad)
    f = f.where(np.abs(f) > 1e-5)
    # compute gradients
    d_eta_d_lat = eta.differentiate("lat") / DEG_LAT_TO_M
    d_eta_d_lon = eta.differentiate("lon") / (DEG_LAT_TO_M * np.cos(lat_rad))
    # geostrophic balance equation
    u = -(G / f) * d_eta_d_lat
    v = (G / f) * d_eta_d_lon
    
    return u.fillna(0).values, v.fillna(0).values


def main():
    SSHA_FILE = "train_ssha.nc" 
    CHL_FILE = "train_chl.nc"
    SST_FILE = "train_sst.nc"
    OUTPUT_DIR = "processed_maps"
    GRID_RESOLUTION = 0.1
    
    print("--- Starting Map Preprocessing ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading datasets...')
    try:
        ds_ssha, ds_chl, ds_sst = (xr.open_dataset(f) for f in [SSHA_FILE, CHL_FILE, SST_FILE])
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find an input file: {e.filename}")
        return

    # Use one idx as a time-reference
    num_time_steps = len(ds_ssha.time)
    print(f"Found {num_time_steps} time steps to process.")

    for t_idx in tqdm(range(num_time_steps), desc="Processing Time Steps"):
        da_ssha = ds_ssha.isel(time=t_idx).to_array().squeeze()
        da_chl = ds_chl.isel(time=t_idx).to_array().squeeze()
        da_sst = ds_sst.isel(time=t_idx).to_array().squeeze()
        
        da_ssha_aligned, da_chl_aligned, da_sst_aligned = align_datasets(
            [da_ssha, da_chl, da_sst], target_res=GRID_RESOLUTION
        )

        u, v = compute_geostrophic_velocity_vectorized(da_ssha_aligned)

        # Clean arrays for this time slice
        u_clean = np.nan_to_num(u, nan=0.0)
        v_clean = np.nan_to_num(v, nan=0.0)
        chl_clean = np.nan_to_num(da_chl_aligned.values, nan=0.0)
        # For sst, calculate the mean of the valid data points for this time slice.
        mean_sst = da_sst_aligned.mean(skipna=True).item()
        sst_clean = da_sst_aligned.fillna(mean_sst).values

        output_filename = f"map_data_time_{t_idx}.npz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        np.savez_compressed(
            output_path,
            u=u_clean,
            v=v_clean,
            chl=chl_clean,
            sst=sst_clean
        )

    print(f"Successfully created {num_time_steps} map files in '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()
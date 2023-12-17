import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from netCDF4 import Dataset
from pathlib import Path
import os
import PIL.Image


# function
# : extract certain nc_data from dataset
# parameters *******************************************
# dataset : dataset to extract nc_data from
# var : variable to extract
# time : time to extract
# ******************************************************
def extract_data(dataset, var):
    return dataset[var]


# function
# : plot climatological data to jpg file w/ nc & plt and save it
# parameters *******************************************
# data : data to plot
# title : title of the plot
# save_path : path to save the plot
# val_min / val_max : minimum / maximum value of the plot
# lon / lat : longitude / latitude of the plot
# ******************************************************
def nc_plot_climatology(data,
                        title,
                        save_path,
                        val_min,
                        val_max,
                        lon,
                        lat):
    # set variables to plot
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_extent([100, 140, 20, 50])

    plt.contourf(lon, lat, data)
    plt.title(title)
    plt.colorbar(location='right')
    plt.clim(val_min, val_max)
    plt.savefig(save_path)
    # clf : clear figure
    plt.clf()


# function
# : plot climatological data to jpg file w/ cartopy & xarray and save it
# parameters *******************************************
# data : data to plot
# title : title of the plot
# save_path : path to save the plot
# val_min / val_max : minimum / maximum value of the plot
# lon / lat : longitude / latitude of the plot
# lon_offset / lat_offset : offset of longitude / latitude
# day : day of the plot
# ******************************************************
def xr_plot_climatology(data,
                        title,
                        save_path,
                        val_min,
                        val_max,
                        lon,
                        lat,
                        lon_offset,
                        lat_offset):

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    norm = colors.Normalize(vmin=val_min, vmax=val_max)

    plt.tight_layout()
    data.plot(ax=ax,
              transform=ccrs.PlateCarree(),
              norm=norm,
              cmap='RdBu_r',
              vmin=val_min,
              vmax=val_max,
              cbar_kwargs={'shrink': 0.5, 'aspect': 15})

    # plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # clf : clear figure
    plt.clf()


# function
# save combined images to gif file
# parameters *******************************************
# image_frames : image array to combine
# save_path : path to save the gif file
# duration : duration of each frame
# ******************************************************
def save_gif(image_frames, save_path, duration):
    image_frames[0].save(save_path,
                         format='GIF',
                         append_images=image_frames[1:],
                         # save_all : 모든 프레임을 저장할 것인지
                         save_all=True,
                         # duration : 프레임 간의 시간 간격 (ms)
                         duration=duration,
                         loop=0)


# function
# calculate correlation between climatological data and certain variable in each grid
# return : correlation data as xr.Dataset
# parameters *******************************************
# ion_con : ion concentration data as xr.Dataset
# cli_data : climatological dataset as xr.Dataset
# ******************************************************
# @jit(nogil=True)
def calculate_lead_lag_correlation(ion_con, cli_data, ion):
    # Resample climatology data to daily frequency
    cli_data_daily = cli_data.resample(time='D').mean()

    corr_results = xr.Dataset()

    resolution = 0.75
    latitude_range = np.arange(90, -90.1, -resolution)
    longitude_range = np.arange(0, 360, resolution)

    # Create the DataArray filled with a constant value
    constant_value = ion_con.loc[ion]  # Replace with the desired constant value
    data = np.ndarray((241, 480))
    certain_ion = xr.full_like(xr.DataArray(data,
                                            coords=[latitude_range, longitude_range],
                                            dims=['latitude', 'longitude']),
                               constant_value)

    # Calculate lead-lag correlation for each variable
    for variable in cli_data_daily.data_vars:
        # Select the variable data for the specific date
        variable_date = cli_data_daily[variable].sel(time='2006-06-20')

        # Create the DataArray filled with a constant value
        constant_value = ion_con.loc['NO3-']  # Replace with the desired constant value
        certain_ion = xr.DataArray(np.full((len(latitude_range), len(longitude_range)), constant_value),
                                   coords=[latitude_range, longitude_range],
                                   dims=['latitude', 'longitude'])

        # Calculate correlation between ion concentration and the variable
        correlation = xr.apply_ufunc(np.corrcoef, certain_ion, variable_date,
                                     input_core_dims=[['time'], ['time']],
                                     output_dtypes=[float],
                                     vectorize=True)

        # Add the correlation values to the results dataset
        corr_results[variable] = correlation.squeeze()

    return corr_results


# function
# calculate correlation between climatological data and certain variable in each grid
# return : correlation data as xr.Dataset
def pr_cor_corr(x, y):
    print("pearson correlation calculation start")
    # Check NA values
    co = np.count_nonzero(~np.isnan(x))

    # If fewer than length of y observations return np.nan
    if co < len(y):
        print('I am here')
        return np.nan, np.nan

    print(f"x : {x}, {x.shape}")
    print(f"y : {y}, {y.shape}")

    warnings.filterwarnings('ignore', category=ConstantInputWarning)
    corr, _ = pearsonr(x, y)

    return corr


# function
# calculate p_value between climatological data and certain variable in each grid
# return : correlation data as xr.Dataset
def pr_cor_pval(x, y):
    # Check NA values
    co = np.count_nonzero(~np.isnan(x))

    # If fewer than length of y observations return np.nan
    if co < len(y):
        return np.nan

    # Run the pearson correlation test
    _, p_value = pearsonr(x, y)

    return p_value


def pearsonr_corr(x, y, func=pr_cor_corr, dim='time'):
    # x = Pixel value, y = a vector containing the date, dim == dimension
    return xr.apply_ufunc(
        func, x, y,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        output_dtypes=[float]
    )


def cor_np(x, y):
    covariance = np.cov(x, y)[0, 1]

    std_x = np.std(x)
    std_y = np.std(y)

    correlation = covariance / (std_x * std_y)
    return correlation

def cor3(da1, da2):
    time = pd.date_range('2006-06-01','2006-06-01',freq="d").day
    lat = da1.latitude
    lon = da1.longitude
    arr = da1.values

    latlontime = []
    for j in range(lat.size):
        for i in range(lon.size):
            if i == 0:
                lontime_stack = cor_np(arr[:, j, i], da2)
            else:
                lontime_piece = cor_np(arr[:,j, i], da2)
                lontime_stack = np.vstack([lontime_stack, lontime_piece])

        latlontime.append(lontime_stack)

    cor = np.stack(latlontime)
    corr = xr.DataArray(cor, coords=(lat, lon, time), dims=["lat", "lon", "time"])
    return corr.transpose("lat", "lon", "time")


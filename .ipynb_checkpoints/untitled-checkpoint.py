import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean


def convert_pr_units(darray, unit=1):
    """Convert kg m-2 s-1 to mm day-1 by default.
   
    Args:
      darray (xarray.DataArray): Precipitation data
      
    Kwargs:
       unit (int): Tells which to convert units to. 
                   0 = kg m-2 s-1 (does not convert units) 
                   1 = mm/day (converts units to mm/day )
    
    """
    if units == 1:
        darray.data = darray.data * 86400
        darray.attrs['units'] = 'mm/day'
    else:
        pass
    return darray


def create_plot(clim, model, season, gridlines=False, levels=None):
    """Plot the precipitation climatology.
   
    Args:
      clim (xarray.DataArray): Precipitation climatology data
      model (str): Name of the climate model
      season (str): Season
      
    Kwargs:
      gridlines (bool): Select whether to plot gridlines
      levels (list): Tick marks on the colorbar    
    
    """

    if not levels:
        levels = np.arange(0, 13.5, 1.5)
        
    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    clim.sel(season=season).plot.contourf(ax=ax,
                                          levels=levels,
                                          extend='max',
                                          transform=ccrs.PlateCarree(),
                                          cbar_kwargs={'label': clim.units},
                                          cmap=cmocean.cm.haline_r)
    ax.coastlines()
    if gridlines:
        plt.gca().gridlines()
    
    title = f'{model} precipitation climatology ({season})'
    plt.title(title)


def main(inargs):
    """Run the program."""
    dset = xr.open_dataset(inargs.pr_file)
    
    clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)
    clim = convert_pr_units(clim,units=inargs.units)

    create_plot(clim, dset.attrs['source_id'], inargs.season,
                gridlines=inargs.gridlines, levels=inargs.cbar_levels)
    plt.savefig(inargs.output_file, dpi=200)


if __name__ == '__main__':
    description='Plot the precipitation climatology.'
    parser = argparse.ArgumentParser(description=description)
   
    parser.add_argument("pr_file", type=str, help="Precipitation data file")
    parser.add_argument("season", type=str, choices=['DJF','MAM','JJA','SON'], help="Season to plot")
    parser.add_argument("output_file", type=str, help="Output file name")

    
    parser.add_argument("--gridlines", action="store_true", default=False,
                        help="Include gridlines on the plot")
    parser.add_argument("--cbar_levels", type=float, nargs='*', default=None,
                        help='list of levels / tick marks to appear on the colorbar')
    parser.add_argument("--units", default='1',
                        help='Change the unit type "0": kgm-2s-1, "1": mm/day')

    args = parser.parse_args
   
    main(args)
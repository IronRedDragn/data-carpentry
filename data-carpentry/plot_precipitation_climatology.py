import argparse

import pdb
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import cmdline_provenance as cmdprov


def convert_pr_units(darray, units=1):
    """Convert kg m-2 s-1 to mm day-1 by default.

    Args:
      darray (xarray.DataArray): Precipitation data
    Kwargs:
       unit (int): Tells which to convert units to. 
                   0 = kg m-2 s-1 (does not convert units) 
                   1 = mm/day (converts units to mm/day )
    """

    assert darray.units == 'kg m-2 s-1', "Script assumes units in precipitation file are in kg m-2 s-1"
    if units == 1:
        darray.data = darray.data * 86400
        darray.attrs['units'] = 'mm/day'
    elif units == 2:
        darray.data = darray.data * 3600
        darray.attrs['units'] = 'mm/hr'
    return darray


def apply_mask(darray, sftlf_file, realm):
    """Mask ocean or land using a sftlf (land surface fraction) file.

    Args:
     darray (xarray.DataArray): Data to mask
     sftlf_file (str): Land surface fraction file
     realm (str): Realm to mask

    """

    dset = xr.open_dataset(sftlf_file)

    assert realm.lower() in [
        'land', 'ocean'], """Valid realms are 'land' or 'ocean' """
    if realm == 'land':
        masked_darray = darray.where(dset['sftlf'].data < 50)
    else:
        masked_darray = darray.where(dset['sftlf'].data > 50)

    return masked_darray

def get_log_and_key(pr_file, history_attr, plot_type):
   """Get key and command line log for image metadata.
  
   Different image formats allow different metadata keys.
  
   Args:
     pr_file (str): Input precipitation file
     history_attr (str): History attribute from pr_file
     plot_type (str): File format for output image
  
   """
   
   valid_keys = {'png': 'History',
                 'pdf': 'Title',
                 'eps': 'Creator',
                 'ps' : 'Creator'}    

   assert plot_type in valid_keys.keys(), f"Image format not one of: {*[*valid_keys],}"
   log_key = valid_keys[plot_type]
   new_log = cmdprov.new_log(infile_history={pr_file: history_attr})
   new_log = new_log.replace('\n', ' END ')
   
   return log_key, new_log


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
        if clim.units == 'mm/day':
            levels = np.arange(0, 13.5, 1.5)
        elif clim.units == 'mm/hr':
            levels = np.arange(0, 0.6, 0.015)
        else:
            levels = np.arange(0, 0.00015, 0.000015)

    fig = plt.figure(figsize=[12, 5])
    ax = fig.add_subplot(
        111, projection=ccrs.PlateCarree(central_longitude=180))
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
    clim = convert_pr_units(clim, units=inargs.units)

    if inargs.mask:
        sftlf_file, realm = inargs.mask
        clim = apply_mask(clim, sftlf_file, realm)

    create_plot(clim, dset.attrs['source_id'], inargs.season,
                gridlines=inargs.gridlines, levels=inargs.cbar_levels)

    log_key, new_log = get_log_and_key(inargs.pr_file, dset.attrs['history'], inargs.output_file.split('.')[-1])
    plt.savefig(inargs.output_file, metadata={log_key: new_log}, dpi=200)   


if __name__ == '__main__':
    description = 'Plot the precipitation climatology for a given season.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("pr_file", type=str, help="Precipitation data file")
    parser.add_argument("season", type=str, choices=[
                        'DJF', 'MAM', 'JJA', 'SON'], help="Season to plot")
    parser.add_argument("output_file", type=str, help="Output file name")

    parser.add_argument("--gridlines", action="store_true", default=False,
                        help="Include gridlines on the plot")

    parser.add_argument("--cbar_levels", type=float, nargs='*', default=None,
                        help='list of levels / tick marks to appear on the colorbar')

    parser.add_argument("--units", type=int, choices=[0, 1, 2], default=1,
                        help='Change the unit type 0: kgm-2s-1, 1: mm/day, 2: mm/hr ')

    parser.add_argument("--mask", type=str, nargs=2,
                        metavar=('SFTLF_FILE', 'REALM'), default=None,
                        help="""Provide sftlf file and realm to mask ('land' or 'ocean')""")

    args = parser.parse_args()

    main(args)

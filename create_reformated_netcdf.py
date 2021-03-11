import argparse

import numpy as np
import xarray as xr
import cmdline_provenance as cmdprov


def convert_pr_units(darray):
    """Convert kg m-2 s-1 to mm day-1.

    Args:
      darray (xarray.DataArray): Precipitation data
    """
    assert darray.units == 'kg m-2 s-1', "Program assumes input units are kg m-2 s-1"
    darray.data = darray.data * 86400
    darray.attrs['units'] = 'mm/day'

    return darray


def main(inargs):
    """Run the program."""
    in_dset = xr.open_dataset(inargs.pr_file)
    clim = in_dset['pr'].groupby('time.season').mean('time', keep_attrs=True)
    clim = convert_pr_units(clim)
    out_dset = clim.to_dataset()
    out_dset.attrs = in_dset.attrs
    new_log = cmdprov.new_log(
        infile_history={inargs.pr_file: in_dset.attrs['history']})
    out_dset.attrs['history'] = new_log
    out_dset.to_netcdf(inargs.output_file)


if __name__ == '__main__':
    description = 'Calculate the seasonal precipitation climatology.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("pr_file", type=str, help="Precipitation data file")
    parser.add_argument("output_file", type=str, help="Output file name")
    args = parser.parse_args()
    main(args)

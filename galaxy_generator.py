import sys
import os
import logging
import galsim
import copy
import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simps

##TEST

##TEST2



## transmission curve based on sdss r-band total throughput for airmass=1.3 extended source

Filter = 'r'
bandpass_file = "passband_sdss_" + Filter
bandpass = galsim.Bandpass(bandpass_file, wave_type = u.angstrom)

eff_wav = 616.5
del_wav = 137
transmission = bandpass(eff_wav)




##psf stuff to come. psf based on SDSS instrumentation and atmospheric conditions, but
## we want to vary psf in some tests also, so maybe just want a Gaussian/Moffat??



## random coordinates just to define wcs and create image stamp

center_ra = 150.  # pick these to match Tractor sim
enter_dec = 2.3
center_ra_hours = 10.

cen_ra = center_ra_hours * galsim.hours
cen_dec = center_dec * galsim.degrees

cen_coord = galsim.CelestialCoord(cen_ra, cen_dec)

pixel_scale = 0.4 #arcsec/pixel
fov_x = 0.25     #deg - can change to make based on size of simulated galaxy after
foy_y = 0.25

image_size_x = fov_x*3600/pixel_scale
image_size_y = fov_y*3600/pixel_scale
image = galsim.Image(image_size_x, image_size_y)

affine_wcs = galsim.PixelScale(pixel_scale).affine().withOrigin(image.center)
wcs = galsim.TanWCS(affine_wcs, world_origin = cen_coord)
image.wcs = wcs




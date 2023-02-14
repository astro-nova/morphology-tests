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




def mag2uJy(mag):
	return 10**(-1*(mag-23.9)/2.5)

def mag2nmgy(mag):
	return 10**(-1*(mag-22.5)/2.5)


def uJy2galflux(uJy, lam_eff, lam_del, throughput):

    lam_eff *= 1e-9  # convert from nm to m
    lam_del *= 1e-9  # convert from nm to m
    nu_del = 2.998e8 / (lam_eff ** 2) * lam_del  # difference in wavelength to difference in frequency
    lum_den = (uJy * u.uJy).to(u.photon / u.cm**2 / u.s / u.Hz, equivalencies=u.spectral_density(lam_eff * u.m)).value
    return throughput * lum_den * nu_del  # return flux in units of photon/cm^2/s


def gen_image(galaxy, centre_ra, centre_dec, pixel_scale, fov_x, fov_y):
	"""
	Generate image with wcs info
	ra, dec in degrees
	pixel_scale in arcsec/pixel
	fov_x, fov_y in degrees  
	"""
	centre_ra_hours = centre_ra/15.
	cen_ra = centre_ra_hours * galsim.hours
	cen_dec = centre_dec * galsim.degrees

	cen_coord = galsim.CelestialCoord(cen_ra, cen_dec)

	image_size_x = fov_x*3600/pixel_scale
	image_size_y = fov_y*3600/pixel_scale
	image = galsim.Image(image_size_x, image_size_y)

	affine_wcs = galsim.PixelScale(pixel_scale).affine().withOrigin(image.center)
	wcs = galsim.TanWCS(affine_wcs, world_origin = cen_coord)
	image.wcs = wcs

	centre_ra_hours = centre_ra/15.
	coord = galsim.CelestialCoord(centre_ra_hours*galsim.hours, centre_dec*galsim.degrees)
	image_pos = wcs.toImage(coord)
	local_wcs = wcs.local(image_pos)
	ix = int(image_pos.x)
	iy = int(image_pos.y)

	stamp = galaxy.drawImage(wcs=local_wcs)
	stamp.setCenter(ix, iy)
	bounds = stamp.bounds & image.bounds
	image[bounds] += stamp[bounds]

	return image, wcs


def gen_galaxy(mag, re, n, q, beta, psf_sig, telescope_params, transmission_params, bandpass):

	g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
	eff_wav, del_wav = transmission_params['eff_wav'],transmission_params['del_wav']

	transmission = bandpass(transmission_params['eff_wav'])

	uJy = mag2uJy(mag)
	
	flux = uJy2galflux(uJy, eff_wav, del_wav, transmission)/g * t_exp * np.pi * (D*100./2)**2

	gal = galsim.Sersic(n=1, flux=flux, half_light_radius=10)
	gal = gal.shear(q = 1, beta=-1*0*galsim.radians)

	psf = galsim.Gaussian(flux=1., sigma=psf_sig)
	final = galsim.Convolve([gal,psf])

	return final


def sky_noise(image, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass):

	g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
	eff_wav, del_wav = transmission_params['eff_wav'],transmission_params['del_wav']

	transmission = bandpass(transmission_params['eff_wav'])
	
	sky_uJy = mag2uJy(sky_mag)*pixel_scale*pixel_scale  
	sky_electrons = uJy2galflux(sky_uJy, eff_wav, del_wav, transmission)/g * t_exp * np.pi * (D*100./2)**2
	sky_rms_electrons = np.sqrt(sky_electrons)
	sky_counts = sky_electrons/g
	sky_rms_counts = sky_rms_electrons/g

	image.addNoise(galsim.GaussianNoise(sigma=sky_rms_counts))

	return image

centre_ra = 150
centre_dec = 2.3
pixel_scale = 0.4
fov = 0.05

## transmission curve based on sdss r-band total throughput for airmass=1.3 extended source
Filter = 'r'
bandpass_file = "passband_sdss_" + Filter
bandpass = galsim.Bandpass(bandpass_file, wave_type = u.angstrom)

telescope_params = {'g':4.8, 't_exp':53.91, 'D':2.5}
transmission_params = {'eff_wav':616.5, 'del_wav':137}

mag = 13
sky_mag = 22 ##mag/arcsec/arcsec

galaxy = gen_galaxy(mag, re=10, n=1, q=1, beta=0, psf_sig=2, telescope_params=telescope_params, 
	transmission_params=transmission_params, bandpass=bandpass)

image, wcs = gen_image(galaxy, centre_ra, centre_dec, pixel_scale, fov, fov)


image = sky_noise(image, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)

import matplotlib.pyplot as plt
plt.imshow(image.array, origin='lower', cmap='Greys')
plt.show()











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
from astropy.visualization import simple_norm
import petrofit
from astropy.modeling.functional_models import Gaussian2D
import matplotlib.pyplot as plt

def mag2uJy(mag):
	"""
	helper function to go from mag to uJy

	Args:
		mag (float) : AB magnitude value
	Returns:
		flux density in uJy 
	"""
	return 10**(-1*(mag-23.9)/2.5)

def mag2nmgy(mag):
	"""
	helper function to go from mag to nmgy

	Args:
		mag (float) : AB magnitude value
	Returns:
		flux density in nanomaggies
	"""
	return 10**(-1*(mag-22.5)/2.5)


def uJy2galflux(uJy, lam_eff, lam_del, throughput):
	"""
	helper function to go from uJy to flux in electrons/cm^2/s using total throughput

	Args:
		uJy (float) : flux density in uJy
		lam_eff (float) : effective wavelength in nanometers
		lam_del (float) : FWHM of throughput curve in nanometers
		throughput (float) : transmission value at lam_eff
	Returns:
		flux value in electrons/s/cm^2

	"""
	lam_eff *= 1e-9  # convert from nm to m
	lam_del *= 1e-9  # convert from nm to m
	nu_del = 2.998e8 / (lam_eff ** 2) * lam_del  # difference in wavelength to difference in frequency
	lum_den = (uJy * u.uJy).to(u.photon / u.cm**2 / u.s / u.Hz, equivalencies=u.spectral_density(lam_eff * u.m)).value
	return throughput * lum_den * nu_del 


def gen_image(centre_ra, centre_dec, pixel_scale, fov_x, fov_y):
	"""
	Generate image with wcs info
	
	Args:
		centre_ra (float) : right ascension in deg
		centre_dec (float)  : declination in deg
		pixel_scale (float) : in arcsec/pixel
		fov_x (float) : in deg
		fov_y (float) : in deg
	Returns:
		image (galsim object) : galsim image object with wcs info and fov
		wcs (wcs object) : WCS header for image 
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
	ix = int(image.center.x)
	iy = int(image.center.y)
	
	return image, wcs



def gen_galaxy(mag, re, n, q, beta, telescope_params, transmission_params, bandpass):
	"""
	create a sersic profile galaxy with given mag, re, n, q, beta

	Args:
		mag (float) : AB magnitude of galaxy
		re (float) : effective radius in arcsec
		n (float) : sersic index
		q (float) : axis ratio of galaxy
		beta (float) : position angle of galaxy
		telescope_params (dict) : telescope parameters (gain, exptime and mirror diameter)
		transmission_params (dict) : tramission parameters (effective wavelength and width)
		bandpass (galsim obbject) : galsim bandpass object defining the total throughput curve
	Returns:
		gal (galsim object) : galsim galaxy object
	"""
	g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
	eff_wav, del_wav = transmission_params['eff_wav'],transmission_params['del_wav']

	transmission = bandpass(transmission_params['eff_wav'])

	uJy = mag2uJy(mag)
	
	## flux in electrons
	flux = uJy2galflux(uJy, eff_wav, del_wav, transmission) * t_exp * np.pi * (D*100./2)**2

	# re is circular re, can then shear the galaxy with an axis ratio and angle
	gal = galsim.Sersic(n=n, flux=flux, half_light_radius=re)
	gal = gal.shear(q = q, beta=-1*beta*galsim.radians)

	return gal


def sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass, seed=None):
	"""
	take image and sky level, calculate level in electrons and apply noise with sky level and source e counts
	can be seeded

	Args:
		image_psf (galsim obj) : galsim image object with sources added and convolved with psf
		sky_mag (float) : mag/arcsec^2 value of sky background
		pixel_scale (flaot) : arcsec/pixel
		telescope_params (dict) : telescope parameters (gain, exptime and mirror diameter)
		transmission_params (dict) : tramission parameters (effective wavelength and width)
		bandpass (galsim obbject) : galsim bandpass object defining the total throughput curve
		seed (int) : seed value for noise (default None)
	Returns:
		image_noise (galsim object) : image_psf + addded noise (image_psf is preserved due to copying)
	"""
	g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
	eff_wav, del_wav = transmission_params['eff_wav'],transmission_params['del_wav']

	transmission = bandpass(transmission_params['eff_wav'])
	
	sky_uJy = mag2uJy(sky_mag)*pixel_scale*pixel_scale  
	sky_electrons = uJy2galflux(sky_uJy, eff_wav, del_wav, transmission) * t_exp * np.pi * (D*100./2)**2

	rng = galsim.BaseDeviate(seed) # if want to seed noise

	# copy image in case iterating over and changing noise level
	image_noise = image_psf.copy()
	image_noise.addNoise(galsim.PoissonNoise(rng=rng, sky_level=sky_electrons))

	return image_noise


def petrosian_sersic(fov, re, n):
	"""
	calculate r_p based on sersic profile
	Args:
		fov (float) : just a stopping point for the range of r_vals [deg]
		re (float) : effective radius in arcsec
		n (float) : sersic index
	Returns:
		PR_p2 : Petrosian radius in arcsec
	"""
	fov = fov*3600
	R_vals = np.arange(0.001, fov/2.0, 0.5)
	R_p2_array = petrofit.modeling.models.petrosian_profile(R_vals, re, n)
	R_p2 = R_vals[np.argmin(np.abs(R_p2_array-0.2))]
	return R_p2


def create_clumps(image, rp, N, gal_mag, telescope_params, transmission_params, bandpass, positions=None, fluxes=None, sigmas=None):
	"""create gaussian clumps to add to galaxy image to simulate intrinsic asymmetry

	Args::
		image (galsim object)  : galsim image with fov and wcs set (needed for setting center)
		rp (float) : Petrosian radius in pixels
		N (int) : number of clumps to create
		telescope_params (dict) : telescope parameters (gain, exptime and mirror diameter)
		transmission_params (dict) : tramission parameters (effective wavelength and width)
		bandpass (galsim obbject) : galsim bandpass object defining the total throughput curve
		positions (list of tuples) : list of (x,y) positions for clumps (default=None and then will be random within rp)
		fluxes (list of floats) : list of fluxes for clumps given in fraction of total galaxy flux (default=None and then will be random)
		sigmas (list of floats) : list of sigma values for gaussian clumps (default=None and then will be random)
	Returns:
		clumps (list of galsim objects) : list of all clump objects to add to image
		all_xi (list of ints) : list of x positions for clumps
		all_yi (list of ints) : list of y positions for clumps
	"""

	# getting galaxy flux from mag
	g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
	eff_wav, del_wav = transmission_params['eff_wav'],transmission_params['del_wav']
	transmission = bandpass(transmission_params['eff_wav'])
	uJy = mag2uJy(gal_mag)
	flux = uJy2galflux(uJy, eff_wav, del_wav, transmission)/g * t_exp * np.pi * (D*100./2)**2


	# get center of image and generate possible pixel values for clumps within r_p
	xc = image.center.x
	yc = image.center.y
	rp = int(rp)
	xvals = np.arange(xc-rp, xc+rp, 1).astype(int)
	yvals = np.arange(yc-rp, yc+rp, 1).astype(int)

	# random flux fractions and sigmas
	flux_fracs = 10**np.linspace(np.log10(0.05), np.log10(0.1), len(xvals))
	sigs = np.linspace(0.5, 5.0, len(xvals))

	clumps = []
	all_xi = []
	all_yi = []
	count = 0

	# if positions, fluxes and sigmas exist and are the same dimensions, use those first
	if positions:
		if len(positions) == len(fluxes) == len(sigmas):
			while count < len(positions):

				pos = positions[count]
				xi = pos[0]
				yi = pos[1]
				flux_frac = fluxes[count]
				sig = sigmas[count]

				# create clump
				clump = galsim.Gaussian(flux=flux*flux_frac, sigma=sig)
				# clump = clump.shear(q = 0.5, beta=-1*galsim.radians)

				# add to lists
				clumps.append(clump)
				all_xi.append(xi)
				all_yi.append(yi)
				count += 1

			N -= len(positions)

	# then randomly assign positions, fluxes and sigmas for remaining clumps
	for i in range(N):

		randx = np.random.randint(0, len(xvals), 1)
		randy = np.random.randint(0, len(yvals), 1)

		xi = xvals[randx][0]
		yi = yvals[randy][0]

		# create clump
		clump = galsim.Gaussian(flux=flux*flux_fracs[randx], sigma=sigs[randx])
		# clump = clump.shear(q = 0.5, beta=-1*galsim.radians)
		
		# add to lists
		clumps.append(clump)
		all_xi.append(xi)
		all_yi.append(yi)

	return clumps, all_xi, all_yi


def add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_sig):
	"""
	adding source galaxy and clumps to image after convolving with psf
	Args:
		image (galsim object)  : galsim image with fov and wcs set (needed for setting center)
		galaxy (galsim object) : galaxy with defined sersic profile
		clumps (list of galsim objects) : list of all clump objects to add to image
		all_xi (list of ints) : list of x positions for clumps
		all_yi (list of ints) : list of y positions for clumps
		psf_sig (float) : sigma for gaussian psf for image
	Returns:
		image_psf (galsim object) : image with psf-convolved objects added in
	"""
	# make copy of image in case iterating over and changing psf each time
	image_psf = image.copy()

	# define Gaussian psf
	psf = galsim.Gaussian(flux=1., sigma=psf_sig)

	# convolve galaxy with psf
	final_gal = galsim.Convolve([galaxy,psf])

	# stamp galaxy and add to image
	stamp_gal = final_gal.drawImage(wcs=image_psf.wcs.local(image_psf.center)) #galaxy at image center
	stamp_gal.setCenter(image_psf.center.x, image_psf.center.y)
	bounds_gal = stamp_gal.bounds & image_psf.bounds
	image_psf[bounds_gal] += stamp_gal[bounds_gal]


	for i in range(len(clumps)):
		clump = clumps[i]
		xi = all_xi[i]
		yi = all_yi[i]

		final_clump = galsim.Convolve([clump,psf])
		stamp_clump = final_clump.drawImage(wcs=image_psf.wcs.local(galsim.PositionI(xi, yi)))
		stamp_clump.setCenter(xi, yi)
		bounds_clump = stamp_clump.bounds & image_psf.bounds
		image_psf[bounds_clump] += stamp_clump[bounds_clump]

	return image_psf



####The following is to test the code#####	

if __name__ == '__main__':

	## transmission curve based on sdss r-band total throughput for airmass=1.3 extended source
	Filter = 'r'
	bandpass_file = "passband_sdss_" + Filter
	bandpass = galsim.Bandpass(bandpass_file, wave_type = u.angstrom)


	## gain, exptime and diameter of telescope
	telescope_params = {'g':4.8, 't_exp':53.91, 'D':2.5}
	## effective wavelength and width of filter
	transmission_params = {'eff_wav':616.5, 'del_wav':137}


	## galaxy and sky params
	mag = 13 # mag of galaxy
	sky_mag = 22 ##mag/arcsec/arcsec sky level
	re = 10 #effective radius in arcsec
	n = 1 # sersic index
	q = 1 #axis ratio
	beta = 0 # orientation angle


	## define ra, dec, pixel scale and fov
	centre_ra = 150
	centre_dec = 2.3
	pixel_scale = 0.4 #arcsec/pixel
	fov = re*12/3600 #deg. Basing off of re


	# generate blank image with fov and wcs info
	image, wcs = gen_image(centre_ra, centre_dec, pixel_scale, fov, fov)

	# create a galaxy with given params
	galaxy = gen_galaxy(mag=mag, re=re, n=n, q=q, beta=beta, telescope_params=telescope_params, 
		transmission_params=transmission_params, bandpass=bandpass)

	# get petrosian radius of galaxy
	rp = petrosian_sersic(fov, re, 1)/pixel_scale  ##in pixels

	# set up creation of clumps for asymmetry
	N = 20  # total number
	positions_clumps=[(50,50)]  # positions (optional) 
	fluxes_clumps = [0.2] # flux fractions (optional), must be same length as positions
	sigmas_clumps = [3] # sigmas for gaussian clumps (optional), must be same length as positions

	# generate all the clumps and their positions
	clumps, all_xi, all_yi = create_clumps(image, rp, N, mag, 
		telescope_params, transmission_params, bandpass, positions_clumps, fluxes_clumps, sigmas_clumps)


	# convolve sources with psf and add to image
	image_psf = add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_sig=2.0)

	# add Poisson noise to image based on pixel counts with added sky level
	image_noise = sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)
	# FINAL IMAGE IN ELECTRON COUNTS


	##############################################
	## Vary the noise size, keep the psf level the same
	seed = None
	noises = [24,22,20,18]
	fig, axs = plt.subplots(nrows=1, ncols=4)
	for d in range(0,4):
		sky_mag = noises[d]


		image_psf = add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_sig=2.0)
		image_noise = sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)


		axs[d].imshow(image_noise.array, origin='lower', cmap='Greys', norm=simple_norm(image_noise.array, stretch='log', log_a=10000))
		axs[d].set_title('Sky level=' + str(sky_mag) + ' mag/arcsec^2')

	fig.show()

	##############################################
	## Vary the psf size, keep the noise level the same
	seed = None
	fig2, axs2 = plt.subplots(nrows=1, ncols=4)
	for d in range(1,5):

		
		image_psf = add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_sig=d)
		image_noise = sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)


		axs2[d-1].imshow(image_noise.array, origin='lower', cmap='Greys', norm=simple_norm(image_noise.array, stretch='log', log_a=10000))
		axs2[d-1].set_title('PSF sig=' + str(d) + '"')

	fig2.show()



	input()

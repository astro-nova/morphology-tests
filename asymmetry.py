import numpy as np
import photutils as phot
from scipy import optimize as opt
from skimage import transform as T
from skimage import measure


def _sky_properties(img, bg_size, a_type='cas'):
    """Calculates the sky asymmetry and flux.
    IN PROGRESS: right now, just draws a sky box in the bottom-left corner. 
    The "rotated" background is simply reflected, works for a Gaussian case.
    TODO: estimate bg asymmetry properly.

    Args:
        img (np.array): an NxN image array
        bg_size (int): size of the square skybox
        a_type (str): formula to use, 'cas' or 'squared'

    Returns:
        sky_a (int): asymmetry of the background per pixel
        sky_norm (int): average background normalization per pixel (<|sky|> or <sky^2>)
    """

    assert a_type in ['cas', 'squared'], 'a_type should be "cas" or "squared"'

    # Get the skybox and rotate it
    sky = img[:bg_size, :bg_size]
    sky_rotated = sky[::-1]
    sky_size = bg_size**2

    # Calculate asymmetry in the skybox
    if a_type == 'cas':
        sky_a = np.sum(np.abs(sky - sky_rotated))
        sky_norm = np.mean(np.abs(sky))
    elif a_type == 'squared':
        sky_a = np.sum((sky-sky_rotated)**2)
        sky_norm = np.mean(sky**2)

    # Calculate per pixel
    sky_a /= sky_size

    return sky_a, sky_norm


def _asymmetry_func(
        center, img, ap_size, sky_a, sky_norm, 
        a_type='cas', bg_corr='full', e=0, theta=0
    ):
    """Calculate asymmetry of the image rotated 180 degrees about a given
    center. This function is minimized in get_asymmetry to find the A center.
    
    Args:
        center (np.array): [x0, y0] coordinates of the asymmetry center.
        img (np.array): an NxN image array.
        ap_size (float): aperture size in pixels.
        sky_a (float): background A calculated in _sky_properties.
        sky_norm (float): the contribution of the sky to the normalization,
            calculated in _sky_properties.
        a_type (str): formula to use, 'cas' or 'squared'.
        bg_corr (str): 
            The way to correct for background between 'none', 'residual', 'full'.
            If 'none', backgorund A is not subtracted. If 'residual', background 
            A is subtracted from the residual term but not the total flux. 
            If 'full', background contribution to the residual AND the total
            flux is subtracted.
        e (float): ellipticity for an elliptical aperture (Default: 0 , circular).
        theta (float): rotation angle for elliptical apertures (Default: 0).

    Returns:
        a (float): asymmetry value
    """

    assert a_type in ['cas', 'squared'], 'a_type should be "cas" or "squared"'
    assert bg_corr in ['none', 'residual', 'full'], 'bg_corr should be "none", "residual", or "full".'

    # Rotate the image about asymmetry center
    img_rotated = T.rotate(img, 180, center=center)

    # Define the aperture
    ap = phot.EllipticalAperture(
        center, a=ap_size, b=ap_size*(1-e), theta=theta)
    
    # Calculate asymmetry of the image
    if a_type == 'cas':
        total_flux = ap.do_photometry(np.abs(img))[0][0]
        residual = ap.do_photometry(np.abs(img-img_rotated))[0][0]
    elif a_type == 'squared':
        total_flux = ap.do_photometry(img**2)[0][0]
        residual = ap.do_photometry((img-img_rotated)**2)[0][0]
    
    # Correct for the background
    if bg_corr is 'none':
        a = residual / total_flux
    elif bg_corr is 'residual':
        a = (residual - ap.area*sky_a) / total_flux
    elif bg_corr is 'full':
        a = (residual - ap.area*sky_a) / (total_flux - ap.area*sky_norm)

    return a



def get_asymmetry(img, ap_size, bg_size, a_type='cas', bg_corr='residual', e=0, theta=0):
    """Finds asymmetry of an image by optimizing the rotation center
    that minimizes the asymmetry calculated in _asymmetry_func. 
    Uses Nelder-Mead optimization from SciPy, same as statmorph.
    
    Args:
        img (np.array): an NxN image array.
        ap_size (float): aperture size in pixels.
        bg_size (int): size of the square skybox
        a_type (str): formula to use, 'cas' or 'squared'.
        bg_corr (str): 
            The way to correct for background between 'none', 'residual', 'full'.
            If 'none', backgorund A is not subtracted. If 'residual', background 
            A is subtracted from the residual term but not the total flux. 
            If 'full', background contribution to the residual AND the total
            flux is subtracted.
        e (float): ellipticity for an elliptical aperture (Default: 0 , circular).
        theta (float): rotation angle for elliptical apertures (Default: 0).

    Returns:
        a (float): asymmetry value
        center (np.array): [x, y] coordinates of the optimum asymmetry center
    """

    # Calculate the background asymmetry and normalization
    sky_a, sky_norm = _sky_properties(img, bg_size, a_type)

    # Initial guess for the A center: center of flux^2. 
    # Note: usually center of flux is used instead. It is often a local maximum, so the optimizer
    # can move towards a local minimum instead of the correct one. Center of flux^2 puts
    # more weight on the central object and avoids placing the first guess on a local max.
    M = measure.moments(img**2, order=2)
    x0 = (M[0, 1] / M[0, 0], M[1, 0] / M[0, 0])

    # Optimize the asymmetry center
    res = opt.minimize(
        _asymmetry_func, x0=x0,  tol=1e-6, method='Nelder-Mead',
        args=(img, ap_size, sky_a, sky_norm, a_type, bg_corr, e, theta))
    
    a = res.fun
    center = res.x
    
    return a, center

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module contains various cutout post-processing tools."""

import os
import warnings
import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord

from .utils.utils import save_fits
from .exceptions import DataWarning



def _combine_headers(headers):
    """
    Combine any number of fits headers such that keywords that
    have the same values in all input headers are unchanged, while
    keywords whose values vary are saved in new keywords as:
        F<header number>_K<#>: Keyword
        F<header number>_V<#>: Value
        F<header number>_C<#>: Comment

    Parameters
    ----------
    headers : list
        List of `~astropy.io.fits.Header` object to be combined.

    Returns
    -------
    response : `~astropy.io.fits.Header`
        The combined `~astropy.io.fits.Header` object.
    """
    
    uniform_cards = []
    varying_keywords = []
    n_vk = 0
    
    for kwd in headers[0]:
        
        # Skip checksums etc
        if kwd in ('S_REGION', 'CHECKSUM', 'DATASUM'):
            continue
        
        if (np.array([x[kwd] for x in headers[1:]]) == headers[0][kwd]).all():
            uniform_cards.append(headers[0].cards[kwd])
        else:
            n_vk += 1
            for i, hdr in enumerate(headers):
                varying_keywords.append((f"F{i+1:02}_K{n_vk:02}", kwd, "Keyword"))
                varying_keywords.append((f"F{i+1:02}_V{n_vk:02}", hdr[kwd], "Value"))
                varying_keywords.append((f"F{i+1:02}_C{n_vk:02}", hdr.comments[kwd], "Comment"))

    # TODO: Add wcs checking? How?
                
    return fits.Header(uniform_cards+varying_keywords)       
        

def build_default_combine_function(template_hdu_arr, no_data_val=np.nan):
    """
    Given an array of `~astropy.io.fits.ImageHdu` objects, 
    initialize a function to combine an array of the same size/shape 
    images where each pixel the mean of all images with available
    data at that pixel.

    Parameters
    ----------
    template_hdu_arr : list
        A list of `~astropy.io.fits.ImageHdu` objects that will be 
        used to create the image combine function.
    no_data_val : scaler
        Optional. The image value that indicates "no data" at a particular pixel.
        The deavault is `~numpy.nan`.

    Returns
    -------
    response : func
        The combiner function that can be applying to other arrays of images.
    """
    
    img_arrs = np.array([hdu.data for hdu in template_hdu_arr])
   
    if np.isnan(no_data_val):
        templates = (~np.isnan(img_arrs)).astype(float)
    else:
        templates = (img_arrs != no_data_val).astype(float)

    multiplier_arr = 1/np.sum(templates, axis=0)
    for t_arr in templates:
        t_arr *= multiplier_arr

    def combine_function(cutout_hdu_arr):
        """
        Combiner function that takes an array of `~astropy.io.fits.ImageHdu` 
        objects and cobines them into a single image.

        Parameters
        ----------
        cutout_hdu_arr : list
            Array of `~astropy.io.fits.ImageHdu` objects that will be 
            combined into a single image.

        Returns
        -------
        response : array
            The combined image array.
        """
        
        cutout_imgs = np.array([hdu.data for hdu in cutout_hdu_arr])
        nans = np.bitwise_and.reduce(np.isnan(cutout_imgs), axis=0)
        
        cutout_imgs[np.isnan(cutout_imgs)] = 0  # don't want any nans because they mess up multiple/add

        combined_img = np.sum(templates*cutout_imgs, axis=0)
        combined_img[nans] = np.nan  # putting nans back if we need to

        return combined_img

    return combine_function



class CutoutsCombiner():
    """
    Class for combining cutouts.
    """

    def __init__(self, fle_list=None, exts=None, img_combiner=None):

        self.input_hdulists = None
        self.center_coord = None
        if fle_list:
            self.load(fle_list, exts)

        self.combine_headers = _combine_headers

        if img_combiner:
            self.combine_images = img_combiner
        else:  # load up the default combiner
            self.build_img_combiner(build_default_combine_function,
                                    builder_args=[self.input_hdulists[0],np.nan])
        
            
    def load(self, fle_list, exts=None):
        """
        Load the input cutout files and select the desired fits extensions.

        Parameters
        ----------
        fle_list : list
            List of files with cutouts to be combined.
        exts : list or None
            Optional. List of fits extensions to combine.
            Default is None, which means all extensions will be combined.
            If the first extension is a PrimaryHeader with no data it will
            be skipped.      
        """
        cutout_hdulists = [fits.open(fle) for fle in fle_list]
        
        if exts is None:
            # Go ahead and deal with possible presence of a primaryHeader and no data as first ext
            if not cutout_hdulists[0][0].data:
                self.input_hdulists = [hdu[1:] for hdu in cutout_hdulists]
            else:
                self.input_hdulists_hdus = cutout_hdulists
        else:
            self.input_hdulists = [hdu[exts] for hdu in cutout_hdulists]

        self.input_hdulists = np.transpose(self.input_hdulists) # Transposing so hdus to be combings are on the short axis

        # Try to find the center coordinate
        try:
            ra = cutout_hdulists[0][0].header['RA_OBJ']
            dec = cutout_hdulists[0][0].header['DEC_OBJ']
            self.center_coord = SkyCoord(f"{ra} {dec}", unit='deg')
        except KeyError:
            warnings.warn(f"Could not find RA/Dec header kewords, center coord will be wrong.",
                          DataWarning)
            self.center_coord = SkyCoord(f"0 0", unit='deg')
            
        except ValueError:
            warnings.warn(f"Invalid RA/Dec values, center coord will be wrong.",
                          DataWarning)
            self.center_coord = SkyCoord(f"0 0", unit='deg')

            
    def build_img_combiner(self, function_builder, builder_args):
        """
        Build the function that will combine cutout extensions.

        Parameters
        ----------
        function_builder : func
            The function that will create the combine function.
        builder_args : list
            Array of arguments for the function builder
        """
        
        self.combine_images = function_builder(*builder_args)
   
        
    def combine(self, output_file="./cutout.fits"):
        """
        Combine cutouts and save the output to a fits file.

        Parameters
        ----------
        output_file : str
            Optional. The filename for the combined cutout file.

        Returns
        -------
        response : str
            The combined cutout filename.
        """

        hdu_list = []

        for ext_hdus in self.input_hdulists:
            
            new_header = self.combine_headers([hdu.header for hdu in ext_hdus])
        
            new_img = self.combine_images([hdu.data for hdu in ext_hdus])
            hdu_list.append(fits.ImageHDU(data=new_img, header=new_header))
 
        save_fits(hdu_list, output_path=output_file, center_coord=self.center_coord)

        return output_file

        

        


# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module contains various cutout post-processing tools."""

import numpy as np
import os

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.time import Time

from scipy.interpolate import splprep, splev


def _combine_headers(headers, constant_only=False):
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

    # Allowing the function to gracefully handle being given a single header
    if len(headers) == 1:
        return headers[0]
    
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
            if constant_only:  # Removing non-constant kewords in this case
                continue
                
            n_vk += 1
            for i, hdr in enumerate(headers):
                varying_keywords.append((f"F{i+1:02}_K{n_vk:02}", kwd, "Keyword"))
                varying_keywords.append((f"F{i+1:02}_V{n_vk:02}", hdr[kwd], "Value"))
                varying_keywords.append((f"F{i+1:02}_C{n_vk:02}", hdr.comments[kwd], "Comment"))

                
    return fits.Header(uniform_cards+varying_keywords)


def _get_bounds(x, y, size):
    """
    Given an x,y coordinates (single or lists) and size, return the bounds of the
    described area(s) as [[[x_min, x_max],[y_min, y_max]],...].
    """
    x = np.array(np.atleast_1d(x))
    y = np.array(np.atleast_1d(y))

    lower_x = np.rint(x - size[0]/2)
    lower_y = np.rint(y - size[1]/2)

    return np.stack((np.stack((lower_x, lower_x + size[0]), axis=1),
                     np.stack((lower_y, lower_y + size[1]), axis=1)), axis=1).astype(int)


def _combine_bounds(bounds1, bounds2):
    """
    Given two bounds of the form [[x_min, x_max],[y_min, y_max]],
    combine them into a new [[x_min, x_max],[y_min, y_max]], that
    encompasses both initial bounds.
    """
    
    bounds_comb = np.zeros((2, 2), dtype=int)
    bounds_comb[0, 0] = bounds1[0, 0] if (bounds1[0, 0] < bounds2[0, 0]) else bounds2[0, 0]
    bounds_comb[1, 0] = bounds1[1, 0] if (bounds1[1, 0] < bounds2[1, 0]) else bounds2[1, 0]
    bounds_comb[0, 1] = bounds1[0, 1] if (bounds1[0, 1] > bounds2[0, 1]) else bounds2[0, 1]
    bounds_comb[1, 1] = bounds1[1, 1] if (bounds1[1, 1] > bounds2[1, 1]) else bounds2[1, 1]
    
    return bounds_comb


def _area(bounds):
    """
    Given bounds of the form [[x_min, x_max],[y_min, y_max]] return
    the area of the described rectangle.
    """
    return (bounds[0, 1] - bounds[0, 0]) * (bounds[1, 1] - bounds[1, 0])


def _get_args(bounds, img_wcs):
    """
    Given bounds of the form [[x_min, x_max],[y_min, y_max]] and a
    `~astropy.wcs.WCS` object return a center coordinate and size
    ([ny, nx] pixels) suitable for creating a rectangular cutout.
    """
    nx = bounds[0, 1]-bounds[0, 0]
    ny = bounds[1, 1]-bounds[1, 0]
    x = nx/2 + bounds[0, 0]
    y = ny/2 + bounds[1, 0]
    return {"coordinates": img_wcs.pixel_to_world(x, y),
            "size": (ny, nx)}


# TODO: Put this in utils 
def path_to_footprints(path, size, img_wcs, max_pixels=10000):
    """
    Given a path that intersects with a wcs footprint, return
    one or more rectangles that fully contain that intersection 
    (plus padding given by 'size') with each rectangle no more than 
    max_pixels in size.
    
    Parameters
    ----------
    path : `~astropy.coordinate.SkyCoord`
        SkyCoord object list of coordinates that represent a continuous path.
    size : array
        Size of the rectangle centered on the path locations that must 
        be included in the returned footprint(s). Formatted as [ny,nx]
    img_wcs : `~astropy.wcs.WCS`
        WCS object the path intersects with. Must include naxis information.
    max_pixels : int
        Optional, default 10000. The maximum area in pixels for individual
        footprints.
        
    Returns
    -------
    response : list
       List of footprints, each a dictionary of the form:
       {'center_coord': `~astropy.coordinate.SkyCoord`, 'size': [ny,nx]}
    """
    
    x, y = img_wcs.world_to_pixel(path)
    
    # Removing any coordinates outside of the img wcs
    valid_locs = ((x >= 0) & (x < img_wcs.array_shape[0])) & ((y >= 0) & (y < img_wcs.array_shape[1]))
    x = x[valid_locs]
    y = y[valid_locs]
    
    bounds_list = _get_bounds(x, y, size)

    combined_bounds = list()
    cur_bounds = bounds_list[0]
    for bounds in bounds_list[1:]:
        new_bounds = _combine_bounds(cur_bounds, bounds)
        
        if _area(new_bounds) > max_pixels:
            combined_bounds.append(cur_bounds)
            cur_bounds = bounds
        else:
            cur_bounds = new_bounds
            
    combined_bounds.append(cur_bounds)
    
    footprints = []
    for bounds in combined_bounds:
        footprints.append(_get_args(bounds, img_wcs))
        
    return footprints 


def _moving_target_focus(path, size, cutout_fles, verbose=False):
    """
    Given a moving target path (list of RA/Decs) and size, that intersects with 
    the given cutout(s) make a cutout of requested size centered on the 
    moving target given by the path.
    
    Note: No resampling is done so there will be some jitter in the moving target
    placement.
    
    Parameters
    ----------
    path : `~astropy.table.Table`
        Table (or similar) object with columns "time," containing `~astropy.time.Time`
        objects and "position," containing `~astropy.coordinate.Skycoord` objects.
    size : array
        Size in pixels of the cutout rectangle centered on the path locations ther will be 
        returned. Formatted as [ny,nx]
    cutout_fles : list
        List of strings that are Target Pixel File paths.
    verbose : bool
        Optional. If true intermediate information is printed.
        
    Returns
    -------
    response : `~astropy.table.Table`
        New cutout table.
    """
    
    cutout_table_list = list()
    
    for fle in cutout_fles:
        if verbose:
            print(fle)
        
        # Get the stuff we need from the cutout file
        hdu = fits.open(fle)
        cutout_table = Table(hdu[1].data)
        cutout_wcs = WCS(hdu[2].header)
        hdu.close()
        
        
        path["x"], path["y"] = cutout_wcs.world_to_pixel(path["position"])
        # This line might need to be refined
        rel_pts = ((path["x"] - size[0]/2 >= 0) & (path["x"] + size[0]/2 < cutout_wcs.array_shape[1]) & 
                   (path["y"] - size[1]/2 >= 0) & (path["y"] + size[1]/2 < cutout_wcs.array_shape[0]))
        
        tck_tuple, u = splprep([path["x"][rel_pts], path["y"][rel_pts]], u=path["time"][rel_pts].jd, s=0)
        
        
        cutout_table["time_jd"] = cutout_table["TIME"] + 2457000  # TESS specific code
        cutout_table = cutout_table[(cutout_table["time_jd"] >= np.min(path["time"][rel_pts].jd)) & 
                                    (cutout_table["time_jd"] <= np.max(path["time"][rel_pts].jd))]
        
        
        cutout_table["x"], cutout_table["y"] = splev(cutout_table["time_jd"], tck_tuple)
        cutout_table["bounds"] = _get_bounds(cutout_table["x"], cutout_table["y"], size)
        
        
        cutout_table["TGT_X"] = cutout_table["x"] - cutout_table["bounds"][:, 0, 0]
        cutout_table["TGT_Y"] = cutout_table["y"] - cutout_table["bounds"][:, 1, 0]
        
        positions = cutout_wcs.pixel_to_world(cutout_table["x"], cutout_table["y"])
        cutout_table["TGT_RA"] = positions.ra.value
        cutout_table["TGT_DEC"] = positions.dec.value

        # This is y vs x beacuse of the way the pixels are stored by fits
        cutout_table["bounds"] = [(slice(*y), slice(*x)) for x, y in cutout_table["bounds"]]
        
        cutout_table["RAW_CNTS"] = [x["RAW_CNTS"][tuple(x["bounds"])] for x in cutout_table]
        cutout_table["FLUX"] = [x["FLUX"][tuple(x["bounds"])] for x in cutout_table]
        cutout_table["FLUX_ERR"] = [x["FLUX_ERR"][tuple(x["bounds"])] for x in cutout_table]
        cutout_table["FLUX_BKG"] = [x["FLUX_BKG"][tuple(x["bounds"])] for x in cutout_table]
        cutout_table["FLUX_BKG_ERR"] = [x["FLUX_BKG_ERR"][tuple(x["bounds"])] for x in cutout_table]
        
        cutout_table.remove_columns(['time_jd', 'bounds', 'x', 'y'])
        cutout_table_list.append(cutout_table)
        
    cutout_table = vstack(cutout_table_list)
    cutout_table.sort("TIME")
    
    return cutout_table


def _configure_bintable_header(new_header, table_headers):
    """
    Given a newly created bintable header (as from `~astropy.io.fits.table_to_hdu`) and
    a list of headers from the tables that went into the new header, add additional common header
    keywords and more desctiption to the new header.
    """

    # Using a single header to get the column descriptions
    column_info = {}
    for kwd in table_headers[0]:
        if "TTYPE" not in kwd:
            continue
        
        colname = table_headers[0][kwd]
        num = kwd.replace("TTYPE", "")
    
        cards = []
        for att in ['TTYPE', 'TFORM', 'TUNIT', 'TDISP', 'TDIM']:
            try:
                cards.append(table_headers[0].cards[att+num])
            except KeyError:
                pass  # if we don't have info for this keyword, just skip it
        
        column_info[colname] = (num, cards)

    # Adding column descriptions and additional info
    for kwd in new_header:
        if "TTYPE" not in kwd:
            continue
        
        colname = new_header[kwd]
        num = kwd.replace("TTYPE", "")
    
        info_row = column_info.get(colname)
        if not info_row:
            new_header.comments[kwd] = 'column name'
            new_header.comments[kwd.replace("TTYPE", "TFORM")] = 'column format'
            continue
    
        info_num = info_row[0]
        cards = info_row[1]
    
        for key, val, desc in cards:
            key_new = key.replace(info_num, num)
            try:
                ext_card = new_header.cards[key_new]
            
                if ext_card[1]:
                    val = ext_card[1]
                if ext_card[2]:
                    desc = ext_card[2]
                
                new_header[key_new] = (val, desc)
            except KeyError:  # card does not already exist, just add new one
                new_header.set(key_new, val, desc, after=kwd)

    # Adding any additional keywords from the original cutout headers
    shared_keywords = _combine_headers(table_headers, constant_only=True)
    for kwd in shared_keywords:
        if kwd in new_header:  # Don't overwrite anything already there
            continue

        if any(x in kwd for x in ["WCA", "WCS", "CTY", "CRP", "CRV", "CUN",
                                  "CDL", "11PC", "12PC", "21PC", "22PC"]):  # Skipping column WCS keywords
            continue

        new_header.append(shared_keywords.cards[kwd])



def center_on_path(path, size, cutout_fles, target=None, img_wcs=None,
                   target_pixel_file=None, output_path=".", verbose=True):
    """
    Given a moving target path that crosses through one or more cutout files 
    (as produced by `cube_cut`/tesscut) and size, create a target pixel file 
    containint a cutout of the requested size centered on the moving target 
    given in the providedpath.
    
    Note: No resampling is done so there will be some jitter in the moving target
    placement.
    
    Parameters
    ----------
    path : `~astropy.table.Table`
        Table (or similar) object with columns "time," containing `~astropy.time.Time`
        objects and "position," containing `~astropy.coordinate.Skycoord` objects.
    size : array
        Size in pixels of the cutout rectangle centered on the path locations ther will be 
        returned. Formatted as [ny,nx]
    cutout_fles : list
        List of strings, Target Pixel File paths that the path crosses.
    target : str
        Optional. The name or ID of the moving target represented by the path.
    img_wcs : `~astropy.wcs.WCS`
        Optional WCS object that is the WCS from the original image (TESS FFI usually)
        all the cutouts came from.
    target_pixel_file : str
        Optional. The name for the output target pixel file. 
        If no name is supplied, the file will be named: 
        ``<target/path>_<cutout_size>_<time range>_astrocut.fits``
    output_path : str
        Optional. The path where the output file is saved. 
        The current directory is default.
    verbose : bool
            Optional. If true intermediate information is printed. 
        
    Returns
    -------
    response : str
        The file path for the output target pixel file.
    """

    # TODO: add ability to take sizes like in rest of cutout functionality
    
    # Performing the path transformation
    cutout_table = _moving_target_focus(path, size, cutout_fles, verbose)

    # Collecting header info we need
    primary_header_list = list()
    table_headers = list()
    for fle in cutout_fles:
        hdu = fits.open(fle, mode='denywrite', memmap=True)
        primary_header_list.append(hdu[0].header)
        table_headers.append(hdu[1].header)
        hdu.close()

    # Building the new primary header
    primary_header = _combine_headers(primary_header_list, constant_only=True)
    primary_header['DATE'] = Time.now().to_value('iso', subfmt='date')
    if target:
        primary_header["OBJECT"] = (target, "Moving target object name/identifier")
    primary_header["TSTART"] = cutout_table["TIME"].min()
    primary_header["TSTOP"] = cutout_table["TIME"].max()

    primary_hdu = fits.PrimaryHDU(header=primary_header)
    
    # Building the cutout table extension
    mt_cutout_fits_table = fits.table_to_hdu(cutout_table)
    _configure_bintable_header(mt_cutout_fits_table.header, table_headers)

    # Building the aperture extension if possible
    if img_wcs:
        aperture = np.zeros(img_wcs.array_shape, dtype=np.int32)
        x_arr, y_arr = img_wcs.world_to_pixel(SkyCoord(cutout_table["TGT_RA"],
                                                       cutout_table["TGT_DEC"], unit='deg'))
        x_2 = size[0]/2
        y_2 = size[1]/2
        
        for x, y in zip(x_arr, y_arr):
            aperture[int(x-x_2): int(x+x_2), int(y-y_2): int(y+y_2)] = 1

        aperture_hdu = fits.ImageHDU(data=aperture)
        aperture_hdu.header['EXTNAME'] = 'APERTURE'
        aperture_hdu.header.extend(img_wcs.to_header(relax=True).cards)
        aperture_hdu.header['INHERIT'] = True

        mt_hdu_list = fits.HDUList(hdus=[primary_hdu, mt_cutout_fits_table, aperture_hdu])

    else:
        mt_hdu_list = fits.HDUList(hdus=[primary_hdu, mt_cutout_fits_table])
        
    if not target_pixel_file:
        target = "path" if not target else target
        target_pixel_file = (f"{target}_{primary_header['TSTART']}-{primary_header['TSTop']}_"
                             f"{size[0]}-x-{size[1]}_astrocut.fits")

    filename = os.path.join(output_path, target_pixel_file)
    mt_hdu_list.writeto(filename, overwrite=True, checksum=True)

    return filename

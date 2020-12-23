import numpy as np
from os import path

from astropy.io import fits
from astropy.coordinates import SkyCoord

from .utils_for_test import create_test_imgs
from .. import cutout_processing, cutouts


def test_combine_headers():

    header_1 = fits.Header(cards=[('KWD_SHR', 20, 'Shared keyword'),
                                  ('KWD_DIF', 'one', 'Different keyword'),
                                  ('CHECKSUM', 1283726182378, "Keyword to drop")])
    header_2 = fits.Header(cards=[('KWD_SHR', 20, 'Shared keyword'),
                                  ('KWD_DIF', 'two', 'Different keyword'),
                                  ('CHECKSUM', 1248721378218, "Keyword to drop")])

    combined_header = cutout_processing._combine_headers([header_1, header_2])

    assert len(combined_header) == 7 
    assert "KWD_SHR" in combined_header
    assert 'KWD_DIF' not in combined_header
    assert 'CHECKSUM' not in combined_header
    assert combined_header['F01_K01'] == combined_header['F02_K01']
    assert combined_header['F01_V01'] != combined_header['F02_V01']
    assert combined_header['F01_V01'] == header_1[combined_header['F01_K01']]
    assert 'F01_K02' not in combined_header


def test_default_combine():
    hdu_1 = fits.ImageHDU(np.array([[1, 1], [0, 0]]))
    hdu_2 = fits.ImageHDU(np.array([[0, 0], [1, 1]]))

    # Two input arrays no overlapping pixels
    combine_func = cutout_processing.build_default_combine_function([hdu_1, hdu_2], 0)
    assert (combine_func([hdu_1, hdu_2]) == 1).all()
    assert (combine_func([hdu_2, hdu_1]) == 0).all()

    # Three input arrays overlapping pixels
    hdu_3 = fits.ImageHDU(np.array([[0, 1], [1, 0]]))
    combine_func = cutout_processing.build_default_combine_function([hdu_1, hdu_2, hdu_3], 0)

    im4 = fits.ImageHDU(np.array([[4, 5], [0, 0]]))
    im5 = fits.ImageHDU(np.array([[0, 0], [4, 5]]))
    im6 = fits.ImageHDU(np.array([[0, 3], [8, 0]]))
    comb_img = combine_func([im4, im5, im6])
    assert (comb_img == [[4, 4], [6, 5]]).all()

    im4 = fits.ImageHDU(np.array([[4, 5], [-3, 8]]))
    im5 = fits.ImageHDU(np.array([[5, 2], [4, 5]]))
    im6 = fits.ImageHDU(np.array([[4, 3], [8, 9]]))
    assert (combine_func([im4, im5, im6]) == comb_img).all()

    # Two input arrays, with nans and a missing pixel
    hdu_1 = fits.ImageHDU(np.array([[1, np.nan], [np.nan, np.nan]]))
    hdu_2 = fits.ImageHDU(np.array([[np.nan, np.nan], [1, 1]]))

    combine_func = cutout_processing.build_default_combine_function([hdu_1, hdu_2])
    assert np.allclose(combine_func([hdu_1, hdu_2]), [[1, np.nan], [1, 1]], equal_nan=True)


def test_combiner(tmpdir):

    test_images = create_test_imgs(50, 6, dir_name=tmpdir)
    center_coord = SkyCoord("150.1163213 2.200973097", unit='deg')
    cutout_size = 2

    cutout_file_1 = cutouts.fits_cut(test_images[:3], center_coord, cutout_size, 
                                     cutout_prefix="cutout_1", output_dir=tmpdir)
    cutout_file_2 = cutouts.fits_cut(test_images[3:], center_coord, cutout_size, 
                                     cutout_prefix="cutout_2", output_dir=tmpdir)

    combiner = cutout_processing.CutoutsCombiner([cutout_file_1, cutout_file_2])

    # Checking the load function
    assert center_coord.separation(combiner.center_coord) == 0
    assert len(combiner.input_hdulists) == 3
    assert len(combiner.input_hdulists[0]) == 2

    # Checking the combiner function was set properly
    comb_1 = combiner.combine_images(combiner.input_hdulists[0])
    combine_func = cutout_processing.build_default_combine_function(combiner.input_hdulists[0])
    assert (comb_1 == combine_func(combiner.input_hdulists[0])).all()

    # Running the combine function and checking the results
    out_fle = combiner.combine(path.join(tmpdir, "combination.fits"))
    comb_hdu = fits.open(out_fle)
    assert len(comb_hdu) == 4
    assert (comb_hdu[1].data == comb_1).all()
    assert np.isclose(comb_hdu[0].header['RA_OBJ'], center_coord.ra.deg)
    assert np.isclose(comb_hdu[0].header['DEC_OBJ'], center_coord.dec.deg)
    comb_hdu.close()
    
    

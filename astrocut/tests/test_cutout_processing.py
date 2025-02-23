import numpy as np
import os

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.table import Table
from astropy.time import Time

from .utils_for_test import create_test_ffis
from .. import cutout_processing, CubeFactory, CutoutFactory

# Example FFI WCS for testing
with open(get_pkg_data_filename('data/ex_ffi_wcs.txt'), "r") as FLE:
    WCS_STR = FLE.read()


def test_combine_headers():

    header_1 = fits.Header(cards=[('KWD_SHR', 20, 'Shared keyword'),
                                  ('KWD_DIF', 'one', 'Different keyword'),
                                  ('CHECKSUM', 1283726182378, "Keyword to drop")])
    header_2 = fits.Header(cards=[('KWD_SHR', 20, 'Shared keyword'),
                                  ('KWD_DIF', 'two', 'Different keyword'),
                                  ('CHECKSUM', 1248721378218, "Keyword to drop")])

    combined_header = cutout_processing._combine_headers([header_1, header_2])

    assert len(combined_header) == 7 
    assert 'KWD_SHR' in combined_header
    assert 'KWD_DIF' not in combined_header
    assert 'CHECKSUM' not in combined_header
    assert combined_header['F01_K01'] == combined_header['F02_K01']
    assert combined_header['F01_V01'] != combined_header['F02_V01']
    assert combined_header['F01_V01'] == header_1[combined_header['F01_K01']]
    assert 'F01_K02' not in combined_header

    combined_header = cutout_processing._combine_headers([header_1, header_2], constant_only=True)
    assert len(combined_header) == 1
    assert 'KWD_SHR' in combined_header
    assert 'KWD_DIF' not in combined_header
    assert 'F01_K01' not in combined_header


def test_get_bounds():

    x = [5, 10]
    y = [2, 20]
    size = [3, 5]
    bounds = cutout_processing._get_bounds(x, y, size)
    assert (bounds == np.array([[[4, 7], [0, 5]], [[8, 11], [18, 23]]])).all()
    
    for nx, ny in bounds:
        assert nx[1]-nx[0] == size[0]
        assert ny[1]-ny[0] == size[1]

    # test that if we move the center a small amount, we still get the same integer bounds
    x = [5.9, 9.8]
    y = [2.2, 20.2]
    assert (cutout_processing._get_bounds(x, y, size) == bounds).all()


def test_combine_bounds():

    x = [5, 10]
    y = [2, 20]
    size = [3, 5]
    bounds = cutout_processing._get_bounds(x, y, size)

    big_bounds = cutout_processing._combine_bounds(bounds[0], bounds[1])
    
    assert big_bounds.dtype == int
    for bx, by in bounds:
        assert big_bounds[0, 0] <= bx[0]
        assert big_bounds[0, 1] >= bx[1]
        assert big_bounds[1, 0] <= by[0]
        assert big_bounds[1, 1] >= by[1]


def test_area():

    x = [5, 10]
    y = [2, 20]
    size = [3, 5]
    area = np.multiply(*size)
    
    bounds = cutout_processing._get_bounds(x, y, size)
    area_0 = cutout_processing._area(bounds[0])
    area_1 = cutout_processing._area(bounds[1])

    assert area_0 == area
    assert area_0 == area_1


def test_get_args():

    wcs_obj = WCS(WCS_STR, relax=True)
    bounds = np.array([[0, 4], [0, 6]])

    args = cutout_processing._get_args(bounds, wcs_obj)
    assert args["coordinates"] == wcs_obj.pixel_to_world(2, 3)
    assert args["size"] == (6, 4)

    
def test_path_to_footprints():

    img_wcs = WCS(WCS_STR, relax=True)
    size = [4, 5]

    xs = [10, 20, 30, 40, 50]
    ys = [1000, 950, 900, 810, 800]
    path = img_wcs.pixel_to_world(xs, ys)

    footprints = cutout_processing.path_to_footprints(path, size, img_wcs)
    assert len(footprints) == 1

    assert (np.max(xs) - np.min(xs) + size[0]) == footprints[0]["size"][1]
    assert (np.max(ys) - np.min(ys) + size[1]) == footprints[0]["size"][0]

    cent_x = (np.max(xs) - np.min(xs) + size[0])//2 + np.min(xs) - size[0]/2 
    cent_y = (np.max(ys) - np.min(ys) + size[1])//2 + np.min(ys) - size[1]/2 
    assert (img_wcs.pixel_to_world([cent_x], [cent_y]) == footprints[0]["coordinates"]).all()

    # Lowering the max pixels so we force >1 footprint
    max_pixels = 100
    footprints = cutout_processing.path_to_footprints(path, size, img_wcs, max_pixels)

    assert len(footprints) == 5
    for fp in footprints:
        assert np.multiply(*fp["size"]) <= max_pixels


def test_moving_target_focus(tmpdir):

    # Making the test cube/cutout
    cube_maker = CubeFactory()
    
    img_sz = 1000
    num_im = 10
    
    ffi_files = create_test_ffis(img_sz, num_im, dir_name=tmpdir)
    cube_file = cube_maker.make_cube(ffi_files, os.path.join(tmpdir, "test_cube.fits"), verbose=False)

    cutout_file = CutoutFactory().cube_cut(cube_file, "250.3497414839765  2.280925599609063", 100, 
                                           target_pixel_file="cutout_file.fits", output_path=tmpdir,
                                           verbose=False)

    cutout_wcs = WCS(fits.getheader(cutout_file, 2))
    cutout_data = Table(fits.getdata(cutout_file, 1))

    # Focusing on a path where the time points line up with cutout times
    coords = cutout_wcs.pixel_to_world([4, 5, 10, 20], [10, 10, 11, 12])
    times = Time(Table(fits.getdata(cutout_file, 1))["TIME"].data[:len(coords)] + 2457000, format="jd")
    path = Table({"time": times, "position": coords})
    size = [4, 4]

    mt_cutout_table = cutout_processing._moving_target_focus(path, size, [cutout_file])
    assert np.allclose(coords.ra.deg, mt_cutout_table["TGT_RA"])
    assert np.allclose(coords.dec.deg, mt_cutout_table["TGT_DEC"])
    assert (mt_cutout_table["TIME"] == cutout_data["TIME"][:len(coords)]).all()
    assert (mt_cutout_table["FFI_FILE"] == cutout_data["FFI_FILE"][:len(coords)]).all()

    # Focusing on a path where interpolation will actually have to be used
    times = Time(Table(fits.getdata(cutout_file, 1))["TIME"].data[:len(coords)*2:2] + 2457000, format="jd")
    path = Table({"time": times, "position": coords})

    mt_cutout_table = cutout_processing._moving_target_focus(path, size, [cutout_file])
    assert mt_cutout_table["TIME"].max() == (times.jd[-1] - 2457000)
    assert len(mt_cutout_table) > len(path)

    
def test_configure_bintable_header(tmpdir):
    

    # Making the test cube/cutout/table we need
    cube_maker = CubeFactory()
    
    img_sz = 1000
    num_im = 10
    
    ffi_files = create_test_ffis(img_sz, num_im, dir_name=tmpdir)
    cube_file = cube_maker.make_cube(ffi_files, os.path.join(tmpdir, "test_cube.fits"), verbose=False)

    cutout_file = CutoutFactory().cube_cut(cube_file, "250.3497414839765  2.280925599609063", 100, 
                                           target_pixel_file="cutout_file.fits", output_path=tmpdir,
                                           verbose=False)

    cutout_wcs = WCS(fits.getheader(cutout_file, 2))
    coords = cutout_wcs.pixel_to_world([4, 5, 10, 20], [10, 10, 11, 12])
    times = Time(Table(fits.getdata(cutout_file, 1))["TIME"].data[:len(coords)] + 2457000, format="jd")
    path = Table({"time": times, "position": coords})
    size = [4, 4]

    mt_cutout_table = cutout_processing._moving_target_focus(path, size, [cutout_file])
    mt_cutout_fits_table = fits.table_to_hdu(mt_cutout_table)
    
    new_header = mt_cutout_fits_table.header
    orig_header = new_header.copy()
    cutout_header = fits.getheader(cutout_file, 1)

    cutout_processing._configure_bintable_header(new_header, [cutout_header])
    for kwd in new_header:
        if kwd in orig_header:
            assert orig_header[kwd] == new_header[kwd]
        else:
            assert cutout_header[kwd] == new_header[kwd]

    # TODO: add test where there are more than one cutout headers


def test_center_on_path(tmpdir):
    
    # Making the test cube/cutout
    cube_maker = CubeFactory()
    
    img_sz = 1000
    num_im = 10
    
    ffi_files = create_test_ffis(img_sz, num_im, dir_name=tmpdir)
    cube_file = cube_maker.make_cube(ffi_files, os.path.join(tmpdir, "test_cube.fits"), verbose=False)

    cutout_maker = CutoutFactory()
    cutout_file = cutout_maker.cube_cut(cube_file, "250.3497414839765  2.280925599609063", 100, 
                                        target_pixel_file="cutout_file.fits", output_path=tmpdir,
                                        verbose=False)

    cutout_wcs = WCS(fits.getheader(cutout_file, 2))

    coords = cutout_wcs.pixel_to_world([4, 5, 10, 20], [10, 10, 11, 12])
    times = Time(Table(fits.getdata(cutout_file, 1))["TIME"].data[:len(coords)] + 2457000, format="jd")
    path = Table({"time": times, "position": coords})
    size = [4, 4]

    # Giving both a target name and a specific output filename
    img_wcs = cutout_maker.cube_wcs
    out_file = cutout_processing.center_on_path(path, size, [cutout_file], "Test Target", img_wcs, 
                                                "mt_cutout.fits", tmpdir, False)
    assert "mt_cutout.fits" in out_file
    
    mt_wcs = WCS(fits.getheader(out_file, 2))
    assert img_wcs.to_header(relax=True) == mt_wcs.to_header(relax=True)

    primary_header = fits.getheader(out_file)
    assert primary_header["DATE"] == Time.now().to_value('iso', subfmt='date')
    assert primary_header["OBJECT"] == "Test Target"

    # Using the default output filename and not giving an image wcs
    out_file = cutout_processing.center_on_path(path, size, [cutout_file],
                                                output_path=tmpdir, verbose=False)
    assert "path" in out_file

    hdu = fits.open(out_file)
    assert len(hdu) == 2
    assert hdu[0].header["DATE"] == Time.now().to_value('iso', subfmt='date')
    assert hdu[0].header["OBJECT"] == ""
    hdu.close()
    

    

    

    
    

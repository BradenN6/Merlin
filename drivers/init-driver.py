import os
import sys
import copy

import numpy as np
import yt
from yt.frontends.ramses.field_handlers import RTFieldFileHandler
import matplotlib.pyplot as plt

# Import PyPi Package
import merlin_spectra as merlin

# Alternatively, use locally stored files
#from merlin_spectra.emission import EmissionLineInterpolator
#from merlin_spectra.galaxy_visualization import VisualizationManager

from importlib.resources import files, as_file

p = files("merlin_spectra") / "linelists"
print(list(p.iterdir()))

'''
init-driver.py

COMBINED INITIALIZATION AND DRIVER SCRIPT FOR ZARATAN

Author: Braden J. Marazzo-Nowicki

Braden J. Marazzo-Nowicki with Dr. Massimo Ricotti
University of Maryland, College Park Astronomy Department

Script to visualize RAMSES-RT Simulations of high-redshift galaxies in a 
variety of metal lines.
Ionization Parameter, Number Density, and Temperature for each pixel are input
into an interpolator for each line; the interpolator is created via the
module 'emission.py'. An EmissionLineInterpolator object is instantiated
given a filepath for a Cloudy-generated line flux list/data table.

Initialisation script--import in driver.py to initialise objects needed
for analysis.
'''

#----------------
# Initialisation
#----------------

#filename = "/Users/bnowicki/Research/Ricotti/output_00273/info_00273.txt"
#filename = "/Users/bnowicki/Research/Ricotti/output_00389/info_00389.txt"
#logSFC_path = "/Users/bnowicki/Research/Ricotti/output_00273/logSFC"
#output_dir = "/Users/bnowicki/Research/Github/Merlin/drivers"

filename = sys.argv[1]
ramses_dir = "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"
logSFC_path = "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial/logSFC"
output_dir = "/scratch/zt1/project/ricotti-prj/user/bnowicki"
print(f'RAMSES-RT Data Filepath = {filename}')

# List of lines in Cloudy Table
lines=["H1_6562.80A","H1_4861.35A","O1_1304.86A","O1_6300.30A","O2_3728.80A",
       "O2_3726.10A","O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A",
       "O3_5006.84A","He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A",
       "C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A",
       "N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

# Associated wavelengths
wavelengths=[6562.80, 4861.35, 1304.86, 6300.30, 3728.80, 3726.10, 1660.81, 
             1666.15, 4363.21, 4958.91, 5006.84, 1640.41, 1335.66, 1906.68, 
             1908.73, 1549.00, 2795.53, 2802.71, 3868.76, 3967.47, 1238.82, 
             1242.80, 1486.50, 1749.67, 6716.44, 6730.82]

# Fields present in RAMSES-RT Output
# NOTE: Requires reading in the updated hydro_file_descriptor.txt
cell_fields = [
    "Density",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "Pressure",
    "Metallicity",
    "xHI",
    "xHII",
    "xHeII",
    "xHeIII",
]

# Extra Particle Fields
epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]

#---------------------------
# Derived Field Definitions
#---------------------------

def _ion_param(field, data):
    # Ionization Parameter Field
    # Based on photon densities in bins 2-4
    # Don't include bin 1 -> Lyman Werner, non-ionizing
    p = RTFieldFileHandler.get_rt_parameters(ds).copy()
    p.update(ds.parameters)

    cgs_c = 2.99792458e10  # light velocity

    # Convert to physical photon number density in cm^-3
    pd_2 = data['ramses-rt','Photon_density_2']*p["unit_pf"]/cgs_c
    pd_3 = data['ramses-rt','Photon_density_3']*p["unit_pf"]/cgs_c
    pd_4 = data['ramses-rt','Photon_density_4']*p["unit_pf"]/cgs_c

    photon = pd_2 + pd_3 + pd_4

    return photon/data['gas', 'number_density']


def _my_temperature(field, data):
    # A more accurate temperature field
    # y(i): abundance per hydrogen atom
    XH_RAMSES=0.76 # defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 # defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") # defined by RAMSES in cooling_module.f90
    kB_RAMSES=yt.YTArray(1.3806200e-16,"erg/K") # defined by RAMSES in cooling_module.f90

    dn=data["ramses","Density"].in_cgs()
    pr=data["ramses","Pressure"].in_cgs()
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yHe = YHE_RAMSES*0.25/XH_RAMSES
    yHeII=data["ramses","xHeII"]*yHe
    yHeIII=data["ramses","xHeIII"]*yHe
    yH2=1.-yHI-yHII
    yel=yHII+yHeII+2*yHeIII
    mu=(yHI+yHII+2.*yH2 + 4.*yHe) / (yHI+yHII+yH2 + yHe + yel)

    return pr/dn * mu * mH_RAMSES / kB_RAMSES


def _my_H_nuclei_density(field, data):
    # number density of hydrogen atoms

    dn=data["ramses","Density"].in_cgs()
    XH_RAMSES=0.76 # defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 # defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") # defined by RAMSES in cooling_module.f90

    return dn*XH_RAMSES/mH_RAMSES


def _my_He_number_density(field, data):
    dn=data["ramses","Density"].in_cgs()
    XH_RAMSES=0.76 # defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 # defined by RAMSES in cooling_module.f90
    mHe_RAMSES=yt.YTArray(6.64600000e-24,"g") # TODO

    return dn*YHE_RAMSES/mHe_RAMSES


def _OII_ratio(field, data):
    # Local OII Ratio -- Diagnostic of electron number density
    # TODO lum or flux?
    #return data['gas', 'flux_O2_3728.80A']/data['gas', 'flux_O2_3726.10A']
    flux1 = data['gas', 'flux_O2_3728.80A']
    flux2 = data['gas', 'flux_O2_3726.10A']

    flux2 = np.where(flux2 < 1e-30, 1e-30, flux2)

    ratio = flux1 / flux2

    return ratio

def _SII_ratio(field, data):
    # Local SII Ratio -- Diagnostic of electron number density
    flux1 = data['gas', 'flux_S2_6716.44A']
    flux2 = data['gas', 'flux_S2_6730.82A']

    flux2 = np.where(flux2 < 1e-30, 1e-30, flux2)

    ratio = flux1 / flux2

    return ratio

def _OIII_ratio(field, data):
    # Local OIII Ratio -- Diagnostic of temperature
    flux1 = data['gas', 'flux_O3_5006.84A']
    flux2 = data['gas', 'flux_O3_4958.91A']
    flux3 = data['gas', 'flux_O3_4363.21A']

    flux3 = np.where(flux3 < 1e-30, 1e-30, flux3)

    ratio = (flux1 + flux2) / flux2

    return ratio


def _pressure(field, data):
    if 'hydro_thermal_pressure' in dir(ds.fields.ramses): # and 
        #'Pressure' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_thermal_pressure']


def _xHI(field, data):
    # Hydrogen, Helium Ionisation Fractions
    if 'hydro_xHI' in dir(ds.fields.ramses): # and \
        #'xHI' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHI']


def _xHII(field, data):
    if 'hydro_xHII' in dir(ds.fields.ramses): # and \
        #'xHII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHII']


def _xHeII(field, data):
    if 'hydro_xHeII' in dir(ds.fields.ramses): # and \
        #'xHeII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHeII']


def _xHeIII(field, data):
    if 'hydro_xHeIII' in dir(ds.fields.ramses): # and \
        #'xHeIII' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_xHeIII']


def _electron_number_density(field, data):
    # units cm^-3
    # Total Hydrogen number density (approximate, assuming H is most of the mass)
    # Using 'number_density'
    # Or, calculate from rho and H mass fraction: nH = rho * X / m_p
    
    # Get H and He ionization fractions (HII=H+/H, HeII=He+/He, HeIII=He++/He)
    #h_p1 = data["ramses", "xHII"]
    #he_p1 = data["ramses", "xHeII"]
    #he_p2 = data["ramses", "xHeIII"]
    
    # Total gas density in number density (n_H + n_He + n_e)
    # 'gas', 'number_density'. 
    # Alternatively, use 'density' / (1.4 * m_p) for default mix.
    # 'number_density' as total atom+ion number density
    #nH_plus_nHe = data["gas", "number_density"]
    
    # Typical primordial abundances: X=0.76, Y=0.24. 
    # Number fraction of He is nHe/nH = (Y/4) / (X/1) = Y / (4X)
    # Roughly nHe/nH ~ 1/12.4
    
    # electron_density = nH * xHII + nHe * xHeII + 2 * nHe * xHeIII
    return data["gas", "my_H_nuclei_density"] * data["ramses", "xHII"] + \
        data["gas", "my_He_number_density"] * (data["ramses", "xHeII"] + \
                                            2 * data["ramses", "xHeIII"])


'''
-------------------------------------------------------------------------------
Load Simulation Data
Add Derived Fields
-------------------------------------------------------------------------------
'''

ds = yt.load(filename, extra_particle_fields=epf)

# NOTE: The derived field units automatically determined by yt
# may be incorrect.

ds.add_field(
    ("gas","number_density"),
    function=_my_H_nuclei_density,
    sampling_type="cell",
    units="1/cm**3",
    force_override=True
)

ds.add_field(
    ("ramses","Pressure"),
    function=_pressure,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHI"),
    function=_xHI,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHII"),
    function=_xHII,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHeII"),
    function=_xHeII,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("ramses","xHeIII"),
    function=_xHeIII,
    sampling_type="cell",
    units="1",
    #force_override=True
)

ds.add_field(
    ("gas","my_temperature"),
    function=_my_temperature,
    sampling_type="cell",
    units='K*cm*dyn/erg',
    force_override=True
)

# Ionisation parameter
ds.add_field(
    ('gas', 'ion_param'),
    function=_ion_param,
    sampling_type="cell",
    units="cm**3",
    force_override=True
)

ds.add_field(
    ("gas","my_H_nuclei_density"),
    function=_my_H_nuclei_density,
    sampling_type="cell",
    units="1/cm**3",
    force_override=True
)

ds.add_field(
    ("gas","my_He_number_density"),
    function=_my_He_number_density,
    sampling_type="cell",
    units="1/cm**3",
    force_override=True
)

# electron number density
ds.add_field(
    ("gas", "electron_number_density"),
    function=_electron_number_density,
    sampling_type="cell",
    units="cm**-3"
)

# Normalise by Density Squared Flag
# If True, returned emissivities are normalised by the density squared.
# Nebular emission lines typically scale as N^2
# NOTE: nonphysical units due to yt's automatic determination
# of derived field units
dens_normalized = False
if dens_normalized: 
    flux_units = '1/cm**6'
    lum_units = '1/cm**3'
else:
    flux_units = '1'
    lum_units = 'cm**3'

#---------------------------------------------------
# Instance of EmissionLineInterpolator for line list
#---------------------------------------------------

# To use a locally stored line list-----
#line_list = os.path.join(os.getcwd(), '../src/merlin_spectra/linelists/linelist-all.dat')
#emission_interpolator = EmissionLineInterpolator(lines, filename=line_list, use_import=False)

# To use a line list in the merlin package source----
# NOTE: linelist-all.dat updated to include HBeta
emission_interpolator = merlin.EmissionLineInterpolator(lines, filename=None,
                 use_import=True, linelist_name="linelist-all.dat")

# Add flux and luminosity fields for all lines in the list
for i, line in enumerate(lines):
    ds.add_field(
        ('gas', 'flux_' + line),
        function=emission_interpolator.get_line_emission(
            i, dens_normalized=dens_normalized
        ),
        sampling_type='cell',
        units=flux_units,
        force_override=True
    )

    ds.add_field(
        ('gas', 'luminosity_' + line),
        function=emission_interpolator.get_luminosity(lines[i]),
        sampling_type='cell',
        units=lum_units,
        force_override=True
    )


ds.add_field(
    ("gas","OII_ratio"),
    function=_OII_ratio,
    sampling_type="cell",
    units="1",
    force_override=True
)

ds.add_field(
    ("gas","SII_ratio"),
    function=_SII_ratio,
    sampling_type="cell",
    units="1",
    force_override=True
)

ds.add_field(
    ("gas","OIII_ratio"),
    function=_OIII_ratio,
    sampling_type="cell",
    units="1",
    force_override=True
)


ad = ds.all_data()
print(ds.field_list)
print(ds.derived_field_list)


#------------------------------------------------
# Limits Dicts for movie-making/scale consistency
#------------------------------------------------

lims_fiducial_00319 = {
    ('gas', 'temperature'):                 (5e2, 1e5),
    ('gas', 'density'):                     (1e-27, 1e-22),
    ('gas', 'my_H_nuclei_density'):         (1e-3, 1e2),
    ('gas', 'my_temperature'):              (5e2, 1e5),
    ('gas', 'ion_param'):                   (4e-12, 1e-4),
    ('gas', 'metallicity'):                 (1e-5, 1e-3),
    ('gas', 'OII_ratio'):                   (1e-5, 1.5),
    ('gas', 'SII_ratio'):                   (1e-5, 1.5), # TODO check these
    ('gas', 'OIII_ratio'):                   (1e-5, 1.5),
    ('ramses', 'xHI'):                      (6e-1, 1e0),
    ('ramses', 'xHII'):                     (0.5e-2, 1e0),
    ('ramses', 'xHeII'):                    (1e-4, 1e-1),
    ('ramses', 'xHeIII'):                   (1e-5, 1e-1),
    ('gas', 'my_He_number_density'):        (0.5e-3, 1.5e3),
    ('gas', 'electron_number_density'):     (0.5e-2, 1.5e4),
    ('gas', 'flux_H1_6562.80A'):            (1e-9, 1e-2),
    ('gas', 'flux_H1_4861.35A'):            (1e-9, 1e-2),
    ('gas', 'flux_O1_1304.86A'):            (6e-17, 5e-10),
    ('gas', 'flux_O1_6300.30A'):            (6e-13, 2e-8),
    ('gas', 'flux_O2_3728.80A'):            (5e-12, 5e-7),
    ('gas', 'flux_O2_3726.10A'):            (5e-12, 5e-7),
    ('gas', 'flux_O3_1660.81A'):            (5e-14, 5e-10),
    ('gas', 'flux_O3_1666.15A'):            (5e-14, 1e-9),
    ('gas', 'flux_O3_4363.21A'):            (1e-16, 1e-9),
    ('gas', 'flux_O3_4958.91A'):            (5e-14, 1e-8),
    ('gas', 'flux_O3_5006.84A'):            (1e-15, 1e-7), 
    ('gas', 'flux_He2_1640.41A'):           (1e-13, 1.5e-8),
    ('gas', 'flux_C2_1335.66A'):            (1e-15, 1e-8),
    ('gas', 'flux_C3_1906.68A'):            (1e-15, 5e-8),
    ('gas', 'flux_C3_1908.73A'):            (1e-15, 5e-8),
    ('gas', 'flux_C4_1549.00A'):            (1e-11, 1e-10),
    ('gas', 'flux_Mg2_2795.53A'):           (1e-12, 5e-7),
    ('gas', 'flux_Mg2_2802.71A'):           (1e-14, 1e-7),
    ('gas', 'flux_Ne3_3868.76A'):           (1e-16, 5e-10),
    ('gas', 'flux_Ne3_3967.47A'):           (5e-17, 1e-10),
    ('gas', 'flux_N5_1238.82A'):            (9e-11, 5e-10),
    ('gas', 'flux_N5_1242.80A'):            (9e-11, 5e-10),
    ('gas', 'flux_N4_1486.50A'):            (1e-18, 1e-12),
    ('gas', 'flux_N3_1749.67A'):            (1e-14, 5e-10),
    ('gas', 'flux_S2_6716.44A'):            (1e-12, 5e-7),
    ('gas', 'flux_S2_6730.82A'):            (1e-14, 1e-7),
}


#----------------
# Line Ratio Diagnostics
#----------------

# Electron Number Density Diagnostics #
# -- [O II] 3729/3726 --
# -- [S II] 6717/6731 --

# Temperature Diagnostics #
# -- [O III] (5007 + 4959)/4363 --
# -- [O I] (6300 + 6364)/5577 missing lines
# -- [N II] (6548 + 6583)/5755 missing lines
# -- [Ne III] (3869 + 3968)/3343 missing lines
# -- [S III] (9531 + 9069)/6312 missing lines



#import yt_initialization
#import merlin_spectra as merlin

#import numpy as np

'''
driver.py

Author: Braden J. Marazzo-Nowicki

Braden J. Marazzo-Nowicki with Dr. Massimo Ricotti
University of Maryland, College Park Astronomy Department

Script to visualize RAMSES-RT Simulations of high-redshift galaxies in a 
variety of metal lines.
Ionization Parameter, Number Density, and Temperature for each pixel are input
into an interpolator for each line; the interpolator is created via the
module 'emission.py'. An EmissionLineInterpolator object is instantiated
given a filepath for a Cloudy-generated line flux list/data table.

Driver script
'''

#------------------------------------------------------------------------------
# Run routines on data
#------------------------------------------------------------------------------

#filename = yt_initialization.filename
#ramses_dir = yt_initialization.ramses_dir
#logSFC_path = yt_initialization.logSFC_path
#lines = yt_initialization.lines
#wavelengths = yt_initialization.wavelengths
#ds = yt_initialization.ds
#ad = yt_initialization.ad
#lims_fiducial_00319 = yt_initialization.lims_fiducial_00319
#emission_interpolator = yt_initialization.emission_interpolator

# Create a visualization object
viz = merlin.VisualizationManager(filename, ramses_dir, logSFC_path, lines, wavelengths, ds, ad, output_dir=output_dir, minimal_output=True, lims_dict=lims_fiducial_00319)

# Star centre of mass; sphere objects; window width
star_ctr = viz.star_center(ad)
sp = ds.sphere(star_ctr, (3000, "pc"))
sp_lum = ds.sphere(star_ctr, (10, 'kpc'))
width = (1500, 'pc')

print(star_ctr)

# Save Simulation Information
viz.save_sim_info()
viz.calc_luminosities(sp)
viz.save_sim_field_info(sp)

#-----------------------------
# Projection/Slice Plots
#-----------------------------

field_list = [
    ('gas', 'temperature'),
    ('gas', 'density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_temperature'),
    ('gas', 'ion_param'),
    ('gas', 'metallicity'),
    ('gas', 'OII_ratio'),
    ('gas', 'SII_ratio'),
    ('gas', 'OIII_ratio'),
    ('ramses', 'xHI'),
    ('ramses', 'xHII'),
    ('ramses', 'xHeII'),
    ('ramses', 'xHeIII'),
    ('gas', 'my_He_number_density'),
    ('gas', 'electron_number_density'),
]

weight_field_list = [
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
]

title_list = [
    'Default Temperature [K]',
    r'Density [g cm$^{-3}$]',
    r'H Nuclei Number Density [cm$^{-3}$]',
    'Temperature [K]',
    'Ionization Parameter',
    'Metallicity',
    r'[O II] Ratio $\lambda$ 3728.80A/$\lambda$ 3726.10A',
    r'[S II] Ratio $\lambda$ 6716.44A/$\lambda$ 6730.82A',
    r'[O III] Ratio',
    r'X$_{\text{HI}}$',
    r'X$_{\text{HII}}$',
    r'X$_{\text{HeII}}$',
    r'X$_{\text{HeIII}}$',
    r'He Number Density [cm$^{-3}$]',
    r'Electron Number Density [cm$^{-3}$]',
]

#r'[O III] Ratio ($\lambda$ 5006.84A + $\lambda$ 4958.91A)/$\lambda$ 4363.21A',

for line in lines:
    if line == 'H1_6562.80A':
        line_title = r'H$\alpha$_6562.80A'
    elif line == 'H1_4861.35A':
        line_title = r'H$\beta$_4861.35A'
    else:
        line_title = line

    field_list.append(('gas', 'flux_'  + line))
    title_list.append(line_title.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]')
    weight_field_list.append(None)


viz.plot_wrapper(sp, width, star_ctr, field_list,
                     weight_field_list, title_list, proj=True, slc=False,
                     lims_dict=lims_fiducial_00319)

viz.plot_wrapper(sp, width, star_ctr, field_list,
                    weight_field_list, title_list, proj=True, slc=False,
                    lims_dict=None)

#-----------------------------
# Phase Plots
#-----------------------------

sp_mass = ds.sphere(star_ctr, (300, "pc"))

phase_config_list = [
    {'x_field': ('gas', 'my_temperature'),
     'y_field': ('gas', 'my_H_nuclei_density'),
     'z_field': ('gas', 'flux_H1_6562.80A'),
     'extrema': {('gas', 'my_temperature'): (1e3, 1e8),
                 ('gas', 'my_H_nuclei_density'): (1e-4, 1e6),
                 ('gas', 'flux_H1_6562.80A'): (1e-20, 1e-14)},
     'x_label': 'Temperature [K]', 
     'y_label': r'H Nuclei Number Density [cm$^{-3}$]', 
     'z_label': r'H$\alpha$_6562.80A'.replace('_', ' ') + 
                    r' Flux [erg s$^{-1}$ cm$^{-2}$]',
     'linear': False
    },
    {'x_field': ('gas', 'my_temperature'),
     'y_field': ('gas', 'electron_number_density'),
     'z_field': ('gas', 'flux_H1_6562.80A'),
     'extrema': {('gas', 'my_temperature'): (1e3, 1e8),
                 ('gas', 'electron_number_density'): (0.5e-2, 1e5),
                 ('gas', 'flux_H1_6562.80A'): (1e-20, 1e-14)},
     'x_label': 'Temperature [K]', 
     'y_label': r'Electron Number Density [cm$^{-3}$]', 
     'z_label': r'H$\alpha$_6562.80A'.replace('_', ' ') + 
                    r' Flux [erg s$^{-1}$ cm$^{-2}$]',
     'linear': False
    },
    {'x_field': ('gas', 'my_temperature'),
     'y_field': ('gas', 'electron_number_density'),
     'z_field': ('gas', 'mass'),
     'extrema': {('gas', 'my_temperature'): (1e3, 1e8),
                 ('gas', 'electron_number_density'): (0.5e-2, 1e5),
                 ('gas', 'mass'): (0.5e30, 1.5e40)},
     'x_label': 'Temperature [K]', 
     'y_label': r'Electron Number Density [cm$^{-3}$]', 
     'z_label': 'Mass [g]',
     'linear': False
    },
    {'x_field': ('gas', 'my_temperature'),
     'y_field': ('gas', 'my_H_nuclei_density'),
     'z_field': ('gas', 'mass'),
     'extrema': {('gas', 'my_temperature'): (1e3, 1e8),
                 ('gas', 'my_H_nuclei_density'): (1e-4, 1e6),
                 ('gas', 'mass'): (0.5e30, 1.5e40)},
     'x_label': 'Temperature [K]', 
     'y_label': r'H Nuclei Number Density [cm$^{-3}$]', 
     'z_label': 'Mass [g]',
     'linear': False
    },
    {'x_field': ('gas', 'my_temperature'),
     'y_field': ('gas', 'electron_number_density'),
     'z_field': ('gas', 'flux_C3_1906.68A'),
     'extrema': {('gas', 'my_temperature'): (1e3, 1e8),
                 ('gas', 'electron_number_density'): (1e-3, 1e5),
                 ('gas', 'flux_C3_1906.68A'): (1e-15, 5e-8)},
     'x_label': 'Temperature [K]', 
     'y_label': r'Electron Number Density [cm$^{-3}$]', 
     'z_label': 'C3_1906.68A'.replace('_', ' ') + 
                    r' Flux [erg s$^{-1}$ cm$^{-2}$]',
     'linear': False
    },
    {'x_field': ('gas', 'my_temperature'),
     'y_field': ('gas', 'electron_number_density'),
     'z_field': ('gas', 'flux_O2_3728.80A'),
     'extrema': {('gas', 'my_temperature'): (1e3, 1e8),
                 ('gas', 'electron_number_density'): (1e-3, 1e5),
                 ('gas', 'flux_O2_3728.80A'): (5e-12, 5e-7)},
     'x_label': 'Temperature [K]', 
     'y_label': r'Electron Number Density [cm$^{-3}$]', 
     'z_label': 'O2_3728.80A'.replace('_', ' ') + 
                    r' Flux [erg s$^{-1}$ cm$^{-2}$]',
     'linear': False
    },
]

viz.phase_plot_wrapper(sp, phase_config_list)

#-----------------------------
# Spectra Generation
#-----------------------------

# TODO self.current_redshift
viz.spectra_driver(1000, 1e-25)
# TODO lum_lims

#line_title = r'H$\alpha$_6562.80A'

#-----------------------------
# Additional Plots
#-----------------------------

# Cumulative Flux Plot
viz.plot_cumulative_field(sp, [('gas', 'flux_H1_6562.80A')],
                          r'H$\alpha$_6562.80A'.replace('_', ' ') + 
                            r' Flux [erg s$^{-1}$ cm$^{-2}$]',
                            'flux_H1_6562.80A_cumulative',
                            (0,1000))


# TODO all 'Projected'
panel_config_envi = [
    {'field': ('gas', 'density'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'Density [g cm$^{-3}$]'},
     {'field': ('gas', 'my_H_nuclei_density'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'H Nuclei Number Density [cm$^{-3}$]'},
     {'field': ('gas', 'electron_number_density'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'Electron Number Density [cm$^{-3}$]'},
    {'field': ('gas', 'my_temperature'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'Temperature [K]'},
     {'field': ('gas', 'ion_param'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'Ionization Parameter'},
     {'field': ('gas', 'metallicity'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'Metallicity $Z/Z_0$'}
]

panel_config_ion_fracs = [
    {'field': ('ramses', 'xHI'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'X$_{\text{HI}}$'},
     {'field': ('ramses', 'xHII'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'X$_{\text{HII}}$'},
     {'field': ('ramses', 'xHeII'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'X$_{\text{HeII}}$'},
    {'field': ('ramses', 'xHeIII'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'X$_{\text{HeIII}}$'}
]

panel_config_eden_line_ratios = [
    {'field': ('gas', 'OII_ratio'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'[O II] Ratio $\lambda$ 3728.80A/$\lambda$ 3726.10A'},
     {'field': ('gas', 'SII_ratio'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'[S II] Ratio $\lambda$ 6716.44A/$\lambda$ 6730.82A'},
     {'field': ('gas', 'electron_number_density'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'Electron Number Density [cm$^{-3}$]'},
    {'field': ('gas', 'my_temperature'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'Temperature [K]'},
]

panel_config_temp_line_ratios = [
    {'field': ('gas', 'OIII_ratio'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'[O III] Ratio'},
     {'field': ('gas', 'electron_number_density'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'Electron Number Density [cm$^{-3}$]'},
    {'field': ('gas', 'my_temperature'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'Temperature [K]'},
]

panel_config_line_emission_4 = [
    {'field': ('gas', 'flux_H1_6562.80A'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'H$\alpha$_6562.80A'.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'},
     {'field': ('gas', 'flux_H1_4861.35A'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'H$\beta$_4861.35A'.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'},
    {'field': ('gas', 'flux_O2_3726.10A'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'O2_3726.10A'.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'},
    {'field': ('gas', 'flux_O3_5006.84A'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'O3_5006.84A'.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'},
]

lines = ["H1_6562.80A","H1_4861.35A","O1_1304.86A","O1_6300.30A","O2_3728.80A",
       "O2_3726.10A","O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A",
       "O3_5006.84A","He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A",
       "C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A",
       "N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

panel_config_line_emission_all = [
    {'field': ('gas', 'flux_H1_6562.80A'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'H$\alpha$_6562.80A'.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'},
     {'field': ('gas', 'flux_H1_4861.35A'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'H$\beta$_4861.35A'.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'},
]

# not quite all, skipping O1 1304
for line in lines[3::]:
    config_element = {'field': ('gas', 'flux_' + line),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': line.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]'}
    panel_config_line_emission_all.append(config_element)



# TODO panels of line emission
# TODO plots over time, connect with SFR bursts, star particle mass

# TODO lims
#viz.panel_plot(sp, panel_config_envi, width, star_ctr, nrows=2, ncols=3, filename='panel_envi')
#viz.panel_plot(sp, panel_config_ion_fracs, width, star_ctr, nrows=2, ncols=2, filename='panel_ion_fracs')
#viz.panel_plot(sp, panel_config_eden_line_ratios, width, star_ctr, nrows=2, ncols=2, filename='panel_eden_line_ratios')
#viz.panel_plot(sp, panel_config_temp_line_ratios, width, star_ctr, nrows=1, ncols=3, filename='panel_temp_line_ratios')
#viz.panel_plot(sp, panel_config_line_emission_4, width, star_ctr, nrows=2, ncols=2, filename='panel_line_emission_4')
#viz.panel_plot(sp, panel_config_line_emission_all, width, star_ctr, nrows=5, ncols=5, filename='panel_line_emission_all')

viz.panel_plot(sp, panel_config_envi, width, star_ctr, nrows=2, ncols=3, filename='panel_envi_lims', lims_dict=lims_fiducial_00319)
#viz.panel_plot(sp, panel_config_ion_fracs, width, star_ctr, nrows=2, ncols=2, filename='panel_ion_fracs_lims', lims_dict=lims_fiducial_00319)
#viz.panel_plot(sp, panel_config_eden_line_ratios, width, star_ctr, nrows=2, ncols=2, filename='panel_eden_line_ratios_lims', lims_dict=lims_fiducial_00319)
#viz.panel_plot(sp, panel_config_temp_line_ratios, width, star_ctr, nrows=1, ncols=3, filename='panel_temp_line_ratios_lims', lims_dict=lims_fiducial_00319)
#viz.panel_plot(sp, panel_config_line_emission_4, width, star_ctr, nrows=2, ncols=2, filename='panel_line_emission_4_lims', lims_dict=lims_fiducial_00319)
#viz.panel_plot(sp, panel_config_line_emission_all, width, star_ctr, nrows=5, ncols=5, filename='panel_line_emission_all_lims', lims_dict=lims_fiducial_00319)


# Stellar Density
viz.star_gas_overlay(sp, star_ctr, width, ('gas', 'flux_H1_6562.80A'),
                    line_title.replace('_', ' ') + 
                            r' Flux [erg s$^{-1}$ cm$^{-2}$]', gas_flag=True,
                            lims_dict=lims_fiducial_00319)






# TODO OII ratio
# TODO lims - fix dicts
# TODO phase plot lims, annotation, more phases
# TODO save as fits files
# TODO change title and axis font sizes
# TODO total on phase profile
# TODO array of data for proj plots - plot in matplotlib
# units -> observed surface brightness
# TODO additional line ratios - also phase plots
# electron density phase plots
# TODO electorn density phase plots, test star-gas overlay, test panel
# TODO ricotti plots
# TODO update panel
# TODO CC-Fiducial 2nd paper with F. Garcia
# star formation efficiencies with C.C. He, Garcia
# nuclear star cluster
# Cloudy
# spectrum from JWST observation -> motivate
# faint, low-mass galaxy; scale luminosity with mass for comparison
# high resolution
# same lines? line ratios compatible?
# total luminosity -> observed
# Merlin
# Science Cases
# - line time dependence, fade after burst compared to tburst
# - line diagnostics vs. ISM phase; inferred density vs distribution
#   - phase plots
#   - Ratio vs ne inflection point - good indicator
#   - known distribution - high dens tail, gaussian; mass fraction vs. ne
#   - inferred e- density, average <ne>
# high den tail contribution high, ne^2
# plot mass vs. ne
# phase n vs. ne
# TODO phase label
# log scale histogram
# high den tail power law 
# broken power law 
# slopes and cutoff at high den
# break degeneracy with multiple diagnostics
# linear profile plots

# log-log plot
# high den power law tail
# wrapper phase profiles
# save fits
# fix labels on proj plots
# double-peak gas contribution to CII - hotter low den, cooler high den; line ratio CII
# map ratios of line intensities spatially resolved - lims
# time dependence
# varying gas mass distribution - Gaussian, power law tail
# clear units - CGS
# change to observed flux - microJy, arcsec; surface brightness/(1+z)^4


# Linear Profile Plots
# Any other phase plots
# More line ratios
# panel with lims
# lims on spectra
# lims and non-lims versions in the same loops

#formation_times = ad[("star", "particle_birth_epoch")]
#birth_times = formation_times * ds.unit_system['time']

#print(birth_times)

#birth_time = ds.arr(ad[('star', 'particle_birth_epoch')], 'code_time')
#age = ds.current_time - birth_time
#age_myr = age.to('Myr')

#print(birth_time.to('Myr'))

# TODO additional line ratios
# TODO angstrom label
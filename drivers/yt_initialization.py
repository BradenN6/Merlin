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
yt-initialization.py

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
filename = "/Users/bnowicki/Research/Ricotti/output_00389/info_00389.txt"
ramses_dir = "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"
logSFC_path = "/Users/bnowicki/Research/Ricotti/output_00273/logSFC"
#filename = sys.argv[1]
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
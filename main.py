import os
import sys
import copy

import numpy as np
import yt
from yt.frontends.ramses.field_handlers import RTFieldFileHandler
import matplotlib.pyplot as plt

from merlin.emission import EmissionLineInterpolator
from merlin import galaxy_visualization

#filename = "/Users/bnowicki/Documents/Research/Ricotti/output_00273/info_00273.txt"
filename = sys.argv[1]
#print(f'Line List Filepath = {filename}')

lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A",
       "O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", 
       "He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A","C4_1549.00A",
       "Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A",
       "N5_1238.82A",
       "N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

wavelengths=[6562.80, 1304.86, 6300.30, 3728.80, 3726.10, 1660.81, 1666.15,
             4363.21, 4958.91, 5006.84, 1640.41, 1335.66,
             1906.68, 1908.73, 1549.00, 2795.53, 2802.71, 3868.76,
             3967.47, 1238.82, 1242.80, 1486.50, 1749.67, 6716.44, 6730.82]

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

epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]

# Ionization Parameter Field
# Based on photon densities in bins 2-4
# Don't include bin 1 -> Lyman Werner non-ionizing
def _ion_param(field, data):
    p = RTFieldFileHandler.get_rt_parameters(ds).copy()
    p.update(ds.parameters)

    cgs_c = 2.99792458e10     #light velocity

    # Convert to physical photon number density in cm^-3
    pd_2 = data['ramses-rt','Photon_density_2']*p["unit_pf"]/cgs_c
    pd_3 = data['ramses-rt','Photon_density_3']*p["unit_pf"]/cgs_c
    pd_4 = data['ramses-rt','Photon_density_4']*p["unit_pf"]/cgs_c

    photon = pd_2 + pd_3 + pd_4

    return photon/data['gas', 'number_density']


def _my_temperature(field, data):
    #y(i): abundance per hydrogen atom
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90
    kB_RAMSES=yt.YTArray(1.3806200e-16,"erg/K") #defined by RAMSES in cooling_module.f90

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


#number density of hydrogen atoms
def _my_H_nuclei_density(field, data):
    dn=data["ramses","Density"].in_cgs()
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90

    return dn*XH_RAMSES/mH_RAMSES


def _OII_ratio(field, data):
    # TODO lum or flux?
    return data['gas', 'flux_O2_3728.80A']/data['gas', 'flux_O2_3726.10A']


def _pressure(field, data):
    if 'hydro_thermal_pressure' in dir(ds.fields.ramses): # and 
        #'Pressure' not in dir(ds.fields.ramses):
        return data['ramses', 'hydro_thermal_pressure']


def _xHI(field, data):
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


'''
-------------------------------------------------------------------------------
Load Simulation Data
Add Derived Fields
-------------------------------------------------------------------------------
'''

ds = yt.load(filename, extra_particle_fields=epf)

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
    # TODO units
    #units="K",
    #units="K*cm**3/erg",
    units='K*cm*dyn/erg',
    force_override=True
)

# Ionization parameter
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


# Normalize by Density Squared Flag
dens_normalized = False
if dens_normalized: 
    flux_units = '1/cm**6'
    lum_units = '1/cm**3'
else:
    flux_units = '1'
    lum_units = 'cm**3'

# Instance of EmissionLineInterpolator for line list at filename
#line_list = os.path.join(os.getcwd(), 'data/linelist.dat')
# TODO altar for zaratan
line_list = os.path.join(os.getcwd(), 'data/linelist2.dat')
emission_interpolator = EmissionLineInterpolator(line_list, lines)

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
# TODO


ad = ds.all_data()
print(ds.field_list)
print(ds.derived_field_list)


'''
-------------------------------------------------------------------------------
Run routines on data
-------------------------------------------------------------------------------
'''

viz = galaxy_visualization.VisualizationManager(filename, lines, wavelengths)
star_ctr = viz.star_center(ad)
sp = ds.sphere(star_ctr, (3000, "pc"))
sp_lum = ds.sphere(star_ctr, (10, 'kpc'))
width = (1500, 'pc')

# Save Simulation Information
viz.save_sim_info(ds)
#viz.save_sim_field_info(ds, ad, sp)
#viz.calc_luminosities(sp)

# Projection and Slice Plots

'''
lims_fiducial_00319 = {
    'Ionization Parameter': [10e-7, 10e0],
    'Number Density': [10e-3, 10e2],
    'Mass Density': [10e-27, 10e-20],
    'Temperature': [10e2, 10e10],
    'Metallicity': [10e-3, 10e0],
    "H1_6562.80A": [10e-9, 10e2],
    "O1_1304.86A": [10e-13, 10e-4],
    "O1_6300.30A": [10e-12, 10e-3],
    "O2_3728.80A": [10e-11, 10e-2],
    "O2_3726.10A": [10e-11, 10e-2],
    "O3_1660.81A": [10e-12, 10e-3],
    "O3_1666.15A": [10e-11, 10e-3],
    "O3_4363.21A": [10e-12, 10e-4],
    "O3_4958.91A": [10e-11, 10e-3],
    "O3_5006.84A": [10e-11, 10e-3], 
    "He2_1640.41A": [10e-9, 10e-4],
    "C2_1335.66A": [10e-12, 10e-3],
    "C3_1906.68A": [10e-11, 10e-2],
    "C3_1908.73A": [10e-10, 10e-3],
    "C4_1549.00A": [10e-13, 10e-7],
    "Mg2_2795.53A": [10e-11, 10e-2],
    "Mg2_2802.71A": [10e-11, 10e-2],
    "Ne3_3868.76A": [10e-12, 10e-4],
    "Ne3_3967.47A": [10e-12, 10e-4],
    "N5_1238.82A": [10e-8, 10e-3],
    "N5_1242.80A": [10e-9, 10e-4],
    "N4_1486.50A": [10e-12, 10e-3],
    "N3_1749.67A": [10e-11, 10e-4],
    "S2_6716.44A": [10e-11, 10e-3],
    "S2_6730.82A": [10e-11, 10e-3]
}
'''
# TODO lims

field_list = [
    ('gas', 'temperature'),
    ('gas', 'density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_temperature'),
    ('gas', 'ion_param'),
    ('gas', 'metallicity'),
    #('gas', 'OII_ratio'),
    ('ramses', 'xHI'),
    ('ramses', 'xHII'),
    ('ramses', 'xHeII'),
    ('ramses', 'xHeIII'),
]

weight_field_list = [
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    ('gas', 'my_H_nuclei_density'),
    #('gas', 'my_H_nuclei_density'),
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
    #r'OII Ratio 3728.80$\AA$/3726.10$\AA$',
    r'X_{HI}',
    r'X_{HII}',
    r'X_{HeII}',
    r'X_{HeIII}',
]

field_list.append(('gas', 'flux_'  + 'H1_6562.80A'))
title_list.append(r'H$\alpha$_6562.80A'.replace('_', ' ') + 
                  r' Flux [erg s$^{-1}$ cm$^{-2}$]')
weight_field_list.append(None)


for line in lines:
    if line == 'H1_6562.80A':
        line_title = r'H$\alpha$_6562.80A'
    else:
        line_title = line

    field_list.append(('gas', 'flux_'  + line))
    title_list.append(line_title.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]')
    weight_field_list.append(None)

    #field_list.append(('gas', 'luminosity_'  + line))
    #title_list.append(line_title.replace('_', ' ') + 
    #                  r' Luminosity [erg s$^{-1}$]')
    #weight_field_list.append(None)


viz.plot_wrapper(ds, sp, width, star_ctr, field_list,
                     weight_field_list, title_list, proj=True, slc=False)



# Phase Plots

extrema = {('gas', 'my_temperature'): (1e3, 1e8),
           ('gas', 'my_H_nuclei_density'): (1e-4, 1e6)}

# ('gas', 'flux_H1_6562.80A'): {} Set z - field TODO

line_title = r'H$\alpha$_6562.80A'

phase_profile, x_vals, y_vals, z_vals = viz.phase_plot(ds, sp, x_field=('gas', 'my_temperature'),
               y_field=('gas', 'my_H_nuclei_density'), z_field=('gas', 'flux_H1_6562.80A'),
               extrema=extrema, x_label='Temperature [K]', 
               y_label=r'H Nuclei Number Density [cm$^{-3}$]', 
               z_label=line_title.replace('_', ' ') + 
                      r' Flux [erg s$^{-1}$ cm$^{-2}$]')

viz.phase_with_profiles(ds, sp, phase_profile, x_field=('gas', 'my_temperature'),
                        y_field=('gas', 'my_H_nuclei_density'),
                        z_field=('gas', 'flux_H1_6562.80A'),
                        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
                        x_label='Temperature [K]',
                        y_label=r'H Nuclei Number Density [cm$^{-3}$]',
                        z_label=line_title.replace('_', ' ') + 
                            r' Flux [erg s$^{-1}$ cm$^{-2}$]')

# Spectra Generation

viz.spectra_driver(ds, 1000, 1e-25)
# TODO lum_lims


line_title = r'H$\alpha$_6562.80A'

# Cumulative Flux Plot
viz.plot_cumulative_field(ds, sp, ('gas', 'flux_H1_6562.80A'),
                          line_title.replace('_', ' ') + 
                            r' Flux [erg s$^{-1}$ cm$^{-2}$]',
                            'flux_H1_6562.80A_cumulative',
                            (0,1000))

# Stellar Density
#viz.star_gas_overlay(ds, ad, sp, star_ctr, width, ('gas', 'flux_H1_6562.80A'),
#                    line_title.replace('_', ' ') + 
#                            r' Flux [erg s$^{-1}$ cm$^{-2}$]', gas_flag=False)

# TODO OII ratio
# TODO lims
# TODO stellar density
# TODO phase plot lims, annotation, more phases
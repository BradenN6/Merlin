import yt_initialization
import merlin_spectra as merlin

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

filename = yt_initialization.filename
lines = yt_initialization.lines
wavelengths = yt_initialization.wavelengths
ds = yt_initialization.ds
ad = yt_initialization.ad
lims_fiducial_00319 = yt_initialization.lims_fiducial_00319
emission_interpolator = yt_initialization.emission_interpolator

# Create a visualization object
viz = merlin.VisualizationManager(filename, lines, wavelengths, ds, ad, lims_dict=lims_fiducial_00319)

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
    r'[O II] Ratio $\lambda$ 3728.80\AA/$\lambda$ 3726.10\AA',
    r'[S II] Ratio $\lambda$ 6716.44\AA/$\lambda$ 6730.82\AA',
    r'[O III] Ratio ($\lambda$ 5006.84\AA + $\lambda$ 4958.91\AA)/$\lambda$ 4363.21\AA',
    r'X$_{\text{HI}}$',
    r'X$_{\text{HII}}$',
    r'X$_{\text{HeII}}$',
    r'X$_{\text{HeIII}}$',
    r'He Number Density [cm$^{-3}$]',
    r'Electron Number Density [cm$^{-3}$]',
]

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


#viz.plot_wrapper(sp, width, star_ctr, field_list,
#                     weight_field_list, title_list, proj=True, slc=False,
#                     lims_dict=lims_fiducial_00319)

#viz.plot_wrapper(sp, width, star_ctr, field_list,
#                    weight_field_list, title_list, proj=True, slc=False,
#                    lims_dict=None)

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

#viz.phase_plot_wrapper(sp, phase_config_list)

#-----------------------------
# Spectra Generation
#-----------------------------

# TODO self.current_redshift
viz.spectra_driver(ds, 1000, 1e-25)
# TODO lum_lims

#line_title = r'H$\alpha$_6562.80A'

#-----------------------------
# Additional Plots
#-----------------------------

# Cumulative Flux Plot
#viz.plot_cumulative_field(sp, [('gas', 'flux_H1_6562.80A')],
#                          r'H$\alpha$_6562.80A'.replace('_', ' ') + 
#                            r' Flux [erg s$^{-1}$ cm$^{-2}$]',
#                            'flux_H1_6562.80A_cumulative',
#                            (0,1000))


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
     'title': r'[O II] Ratio $\lambda$ 3728.80\AA/$\lambda$ 3726.10\AA'},
     {'field': ('gas', 'SII_ratio'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'[S II] Ratio $\lambda$ 6716.44\AA/$\lambda$ 6730.82\AA'},
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
     'title': r'[O III] Ratio ($\lambda$ 5006.84\AA + $\lambda$ 4958.91\AA)/$\lambda$ 4363.21\AA'},
     {'field': ('gas', 'electron_number_density'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': r'Electron Number Density [cm$^{-3}$]'},
    {'field': ('gas', 'my_temperature'),
     'plot_type': 'projection',
     'weight_field': ('gas', 'my_H_nuclei_density'),
     'title': 'Temperature [K]'},
]

# TODO panels of line emission
# TODO plots over time, connect with SFR bursts, star particle mass

# TODO lims
viz.panel_plot(sp, panel_config_envi, width, star_ctr, nrows=2, ncols=3, filename='panel_envi')
viz.panel_plot(sp, panel_config_ion_fracs, width, star_ctr, nrows=2, ncols=2, filename='panel_ion_fracs')
viz.panel_plot(sp, panel_config_eden_line_ratios, width, star_ctr, nrows=2, ncols=2, filename='panel_eden_line_ratios')
viz.panel_plot(sp, panel_config_temp_line_ratios, width, star_ctr, nrows=1, ncols=3, filename='panel_temp_line_ratios')


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
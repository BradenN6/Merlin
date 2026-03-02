import yt_initialization
import merlin_spectra as merlin

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

# Create a visualization object
viz = merlin.VisualizationManager(filename, lines, wavelengths, ds, ad, lims_dict=lims_fiducial_00319)

# Star centre of mass; sphere objects; window width
star_ctr = viz.star_center(ad)
sp = ds.sphere(star_ctr, (3000, "pc"))
sp_lum = ds.sphere(star_ctr, (10, 'kpc'))
width = (1500, 'pc')

# Save Simulation Information
viz.save_sim_info()
viz.calc_luminosities(sp)
#viz.save_sim_field_info(ds, ad, sp)

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
]

title_list = [
    'Default Temperature [K]',
    r'Density [g cm$^{-3}$]',
    r'H Nuclei Number Density [cm$^{-3}$]',
    'Temperature [K]',
    'Ionization Parameter',
    'Metallicity',
    r'OII Ratio 3728.80$\AA$/3726.10$\AA$',
    r'X$_{\text{HI}}$',
    r'X$_{\text{HII}}$',
    r'X$_{\text{HeII}}$',
    r'X$_{\text{HeIII}}$',
    r'He Number Density [cm$^{-3}$]',
    r'Electron Number Density [cm$^{-3}$]',
    r'Electron Number Density (approximate) [cm$^{-3}$]',
]

#field_list.append(('gas', 'flux_'  + 'H1_6562.80A'))
#title_list.append(r'H$\alpha$_6562.80A'.replace('_', ' ') + 
#                  r' Flux [erg s$^{-1}$ cm$^{-2}$]')
#weight_field_list.append(None)

#field_list.append(('gas', 'flux_'  + 'H1_4861.35A'))
#title_list.append(r'H$\beta$_4861.35A'.replace('_', ' ') + 
#                  r' Flux [erg s$^{-1}$ cm$^{-2}$]')
#weight_field_list.append(None)


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


#viz.plot_wrapper(ds, sp, width, star_ctr, field_list,
#                     weight_field_list, title_list, proj=True, slc=False,
#                     lims_dict=lims_fiducial_00319)

viz.plot_wrapper(sp, width, star_ctr, field_list,
                    weight_field_list, title_list, proj=True, slc=False,
                    lims_dict=None)


#-----------------------------
# Phase Plots
#-----------------------------

extrema = {('gas', 'my_temperature'): (1e3, 1e8),
           ('gas', 'my_H_nuclei_density'): (1e-4, 1e6),
           ('gas', 'flux_H1_6562.80A'): (1e-20, 1e-14)}

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
                            r' Flux [erg s$^{-1}$ cm$^{-2}$]', linear=True)


#-----------------------------
# Spectra Generation
#-----------------------------

viz.spectra_driver(ds, 1000, 1e-25)
# TODO lum_lims

line_title = r'H$\alpha$_6562.80A'

#-----------------------------
# Additional Plots
#-----------------------------

# Cumulative Flux Plot
viz.plot_cumulative_field(ds, sp, ('gas', 'flux_H1_6562.80A'),
                          line_title.replace('_', ' ') + 
                            r' Flux [erg s$^{-1}$ cm$^{-2}$]',
                            'flux_H1_6562.80A_cumulative',
                            (0,1000))

# Stellar Density
viz.star_gas_overlay(ds, ad, sp, star_ctr, width, ('gas', 'flux_H1_6562.80A'),
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
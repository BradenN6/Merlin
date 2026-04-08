# importing packages
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import merlin_spectra.emission
import astropy
import yt
from yt.units import dimensions
import copy
from scipy.special import voigt_profile
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import LogNorm, Normalize
import sys
from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from typing import Optional, Tuple, List, Dict

'''
galaxy_visualization.py

Author: Braden J. Marazzo-Nowicki

Visualization and analysis routines for RAMSES-RT Simulations.

Last Updated (DD-MM-YYYY): 03-03-2026
'''


class VisualizationManager:
    '''
    High-level visualization and analysis manager for RAMSES-RT simulations.

    This class provides methods to:
        - Generate projection and slice plots
        - Produce phase diagrams
        - Compute emission line luminosities
        - Create panel figures
        - Export image data to FITS files
        - Save simulation metadata

    Designed for publication-quality figure production and
    reproducible scientific workflows.
    '''

    #------------------------------------------------------------------
    # Initialization
    #------------------------------------------------------------------

    def __init__(self, 
                 filename: str,
                 ramses_dir,
                 logSFC_path,
                 lines,
                 wavelengths,
                 ds,
                 ad,
                 output_dir,
                 minimal_output=True,
                 buff_size: int=2000,
                 lims_dict=None):
        '''
        Initialise a VisualizationManager object.

        Parameters
        ----------
        filename : str
            Filepath to the RAMSES-RT output_*/info_*.txt file
        ramses_dir : str
            Filepath to the RAMSES-RT output directory
            e.g. "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"
        logSFC_path : str
            Filepath to the logSFC file
            e.g. 
        lines : List, str 
            List of nebular emission lines
        wavelengths : List, float
            List of corresponding wavelengths
        ds : yt.Dataset
            Loaded dataset. Specify a default ds object for the instance of
            class.
        ad : alldata object
            Specify a default ad object for this instance of the
            class.
        buff_size : int
            Resolution of fixed resolution buffers (FRBs).
        lims_dict : None or Dict
            dictionary of Tuple(float, float):(vmin, vmax) fixed limits on
            colorbar values for image if desired; otherwise None

        file_dir : str 
            Filepath to output directory
        output_file : str
            Output folder, e.g. output_00273
        sim_run : str
            Time slice number for simulation eg. 00273
        info_file : str
            Filename with info file appended '/info_00273.txt'
        directory : str
            Analysis output directory
        redshift : float
            Current redshift.

        Returns
        -------
        None
        '''

        self.filename = filename
        self.ramses_dir = ramses_dir
        self.logSFC_path = logSFC_path
        self.file_dir = os.path.dirname(self.filename)
        self.lines = lines
        self.wavelengths = wavelengths
        self.output_file = self.file_dir.split('/')[-1]
        self.sim_run = self.output_file.split('_')[1]
        self.ds = ds
        self.ad = ad
        self.output_dir = output_dir
        self.minimal_output = minimal_output
        self.buff_size = buff_size
        self.lims_dict = lims_dict
        self.redshift = ds.current_redshift
        self.hubble_constant = ds.hubble_constant

        # Analysis directory for saving
        self.directory = f'{output_dir}/analysis/{self.output_file}_analysis'

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        print(f'Filename = {self.filename}')
        print(f'File Directory = {self.file_dir}')
        print(f'Output File = {self.output_file}')
        print(f'Simulation Run = {self.sim_run}')
        print(f'Analysis Directory = {self.directory}')


    #------------------------------------------------------------------
    # Helpers
    #------------------------------------------------------------------

    def star_center(self,
                    ad=None):
        '''
        Locate the center of mass of star particles in code units.

        Parameters
        ----------
        ad : alldata object, optional
            data object from RAMSES-RT output loaded into yt-project
            (optionally specify; otherwise use the object-stored ad).

        Returns
        -------
        ctr_at_code : List, float 
            Coordinates (code units) of center of mass
        '''

        # if ad is not specified in the call, use object-stored ad
        if ad is None:
            ad = self.ad

        # Compute star center of mass
        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)
        x_pos = x_pos - x_center
        y_pos = y_pos - y_center
        z_pos = z_pos - z_center
        ctr_at_code = np.array([x_center, y_center, z_center])

        return ctr_at_code


    def write_fits_image(self,
                     data: np.ndarray,
                     filename: str,
                     field: Optional[str]=None,
                     width: Optional[Tuple[float, str]]=None,
                     center: Optional[np.ndarray]=None,
                     redshift: Optional[float]=None,
                     extra_header: Optional[Dict]=None) -> None:
        """
        Write 2D image array to a standardized FITS file.

        Parameters
        ----------
        data : ndarray
            2D image array.
        filename : str
            Output filename (must include .fits).
        field : str, optional
            Field name stored in header.
        width : tuple, optional
            Image width (value, unit).
        center : ndarray, optional
            3-element array specifying image center.
        redshift : float, optional
            Snapshot redshift.
        extra_header : dict, optional
            Additional FITS header entries.

        Returns
        -------
        None
        """
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdr = hdu.header

        if field:
            hdr["FIELD"] = str(field)
        if redshift is not None:
            hdr["REDSHIFT"] = redshift
        if width:
            hdr["WIDTH"] = width[0]
            hdr["WUNIT"] = width[1]
        if center is not None:
            hdr["CENX"] = float(center[0])
            hdr["CENY"] = float(center[1])
            hdr["CENZ"] = float(center[2])

        if extra_header:
            for key, val in extra_header.items():
                hdr[key] = val

        hdu.writeto(filename, overwrite=True)


    def get_norm(self,
                 image: np.ndarray,
                 lims: Optional[Tuple[float, float]]=None,
                 log: bool=True):
        """
        Construct matplotlib normalisation object.

        Parameters
        ----------
        image : ndarray
            Image array.
        lims : tuple, optional
            (vmin, vmax) limits.
        log : bool
            Use logarithmic scaling.

        Returns
        -------
        norm : matplotlib.colors.Normalize
        """
        if lims is not None:
            vmin, vmax = lims
        else:
            vmin = np.nanmin(image[image > 0]) if log else np.nanmin(image)
            vmax = np.nanmax(image)

            if vmin == vmax:
                vmax = vmin * 10

        if log:
            return LogNorm(vmin=vmin, vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)
    
    
    def convert_to_plt(self, yt_plot, plot_type, field, width, 
                       title, lims=None, figsize=(8,6), dpi=300):
        '''
        Convert a yt projection or slice plot to matplotlib.

        Parameters
        ----------
        yt_plot : yt.ProjectionPlot or yt.SlicePlot Object
        plot_type : str
            Type of plot (for filename) - 'proj' or 'slc'
        field : Tuple[str, str]
            field to plot, e.g. ('gas', 'temperature')
        width : tuple, int and str
            width in code units or formatted with units, e.g. (1500, 'pc')
        title : str
            Plot title
        lims : None or List
            [vmin, vmax] fixed limits on colorbar values
            for image if desired; otherwise None.
        figsize : Tuple[Int, Int]
            matplotlib figure size
        dpi : int

        Returns
        -------
        p_img : np.ndarray, float
            2D numpy array containing the image data

        Saves desired figures with usable file naming scheme.
        '''

        lbox = width[0]
        length_unit = width[1]
        field_comma = field[1].replace('.', ',')

        plot_title = f'{self.output_file}_{lbox}{length_unit}_' + \
            f'{field_comma}_{plot_type}'
        
        plt_path = os.path.join(self.directory, 'proj_slc_plots')
        fits_path = os.path.join(plt_path, 'fits')
        if not os.path.exists(plt_path):
            os.makedirs(plt_path)
        if not os.path.exists(fits_path):
            os.makedirs(fits_path)

        fname = os.path.join(plt_path, plot_title)
        if lims is not None:
            fname = fname + '_lims'

        plot_frb = yt_plot.frb
        p_img = np.array(plot_frb[field[0], field[1]])

        # Clip non-positive values to avoid log of zero or negative numbers
        if np.min(p_img) <= 0:
            print('Warning: Data contains non-positive values. Adjusting ' +
                  'for LogNorm.')
            
            # Clip values below 1e-10
            p_img = np.clip(p_img, a_min=1e-10, a_max=None)

        # Replace NaN with 0 and Inf with finite numbers
        if np.any(np.isnan(p_img)) or np.any(np.isinf(p_img)):
            print('Warning: Data contains NaN or Inf values. ' +
                  'Replacing with 0.')
            p_img = np.nan_to_num(p_img)

        # TODO
        #p_img = gaussian_filter(p_img, sigma=1)

        # Set the extent of the plot
        extent_dens = [-lbox/2, lbox/2, -lbox/2, lbox/2]

        # Define the color normalization based on the range of the data
        # Viridis, Inferno, Magma maps work - perceptually uniform
        dens_norm = self.get_norm(p_img, lims)

        fig = plt.figure(figsize=figsize)

        im = plt.imshow(p_img, norm=dens_norm, extent=extent_dens, 
                        origin='lower', aspect='equal', 
                        interpolation='nearest', cmap='viridis')

        plt.xlabel(f'X [{length_unit}]', fontsize=16)
        plt.ylabel(f'Y [{length_unit}]', fontsize=16)
        #plt.title(title, fontsize=14)

        plt.xlim(-lbox/2, lbox/2)
        plt.ylim(-lbox/2, lbox/2)

        cbar = plt.colorbar(im)
        cbar.set_label(title, size=14)  # labelpad=10, y=1.05)

        # Add redshift
        plt.text(0.05, 0.05, f'z = {self.redshift:.5f}', color='white',
                 fontsize=9, ha='left', va='bottom',
                 transform=plt.gca().transAxes)

        if self.minimal_output is False:
            fits_fname = os.path.join(fits_path, plot_title)

            # Save FITS file
            self.write_fits_image(
                p_img,
                f"{fits_fname}.fits",
                field=str(field),
                width=width,
                center=self.star_center(),
                redshift=self.redshift
            )

        # Save the figure
        plt.savefig(f"{fname}.png", dpi=300)

        if self.minimal_output is False:
            plt.savefig(f"{fname}.pdf", dpi=300)
        
        plt.close()

        return p_img


    #------------------------------------------------------------------
    # Simulation Information
    #------------------------------------------------------------------

    def calc_luminosities(self, sp):
        '''
        Agreggate luminosities for each emission line in sphere sp.

        Parameters
        ----------
        sp: Data sphere object.

        Returns
        -------
        luminosities : List, float
            array of luminosities for corresponding emission lines.
        '''

        lum_file_path = os.path.join(self.directory, 
                                     f'{self.output_file}_line_luminosity.txt')

        luminosities = []

        for line in self.lines:
            luminosity=sp.quantities.total_quantity(
                ('gas', 'luminosity_' + line)
            )
            luminosities.append(luminosity.value)
            print(f'{line} Luminosity = {luminosity} erg/s')

        self.luminosities = luminosities

        with open(lum_file_path, 'w') as file:
            for i, line in enumerate(self.lines):
                file.write(f'{line} Luminosity: {self.luminosities[i]}\n')

        return luminosities
    

    def code_age_to_myr(self, all_star_ages, hubble_const, unique_age=True, true_age=False):
        r"""
        Returns an array with unique birth epochs in Myr given
        raw_birth_epochs = ad['star', 'particle_birth_epoch']
        AND
        hubble = ds.hubble_constant
        Youngest is 0 Myr, all others are relative to the youngest.

        Relative ages option is currently yielding inconsistent results

        Adapted from F. A. Garcia
        """
        cgs_yr = 3.1556926e7  # 1yr (in s)
        cgs_pc = 3.08567758e18  # pc (in cm)
        h_0 = hubble_const * 100  # hubble parameter (km/s/Mpc)
        h_0_invsec = h_0 * 1e5 / (1e6 * cgs_pc)  # hubble constant h [km/s Mpc-1]->[1/sec]
        h_0inv_yr = 1 / h_0_invsec / cgs_yr  # 1/h_0 [yr]

        if unique_age is True:
            # process to unique birth epochs only as well as sort them
            be_star_processed = np.array(sorted(list(set(all_star_ages.to_ndarray()))))
            star_age_myr = (be_star_processed * h_0inv_yr) / 1e6  # t=0 is the present
            relative_ages = star_age_myr - star_age_myr.min()
        else:
            all_stars = all_star_ages
            star_age_myr = all_stars * h_0inv_yr / 1e6  # t=0 is the present
            relative_ages = star_age_myr - star_age_myr.min()
        if true_age is True:
            return star_age_myr  # + 13.787 * 1e3
        else:
            return relative_ages  # t = 0 is the age of


    def get_star_ages(self, ram_ds=None, ram_ad=None, logsfc=None):
        """
        star's ages in [Myr]

        Adapted from F. A. Garcia
        """

        if ram_ds is None:
            ram_ds = self.ds
        if ram_ad is None:
            ram_ad = self.ad
        if logsfc is None:
            logsfc = self.logSFC_path

        first_form = np.loadtxt(logsfc, usecols=2).max()  # redshift z
        current_hubble = ram_ds.hubble_constant
        current_time = float(ram_ds.current_time.in_units("Myr"))

        birth_start = np.round(
            float(ram_ds.cosmology.t_from_z(first_form).in_units("Myr")), 0
        )
        converted_unfiltered = self.code_age_to_myr(
            ram_ad["star", "particle_birth_epoch"],
            current_hubble,
            unique_age=False,
        )
        birthtime = np.round(converted_unfiltered + birth_start, 3)  #!
        current_ages = np.array(np.round(current_time, 3) - np.round(birthtime, 3))
        return current_ages


    def calc_sfr(self, dt, ds=None, ad=None):
        '''
        Calculate the star formation rate in a time slice.

        Parameters
        ----------
        ds : yt.Dataset
            loaded RAMSES-RT data set (optionally specify; otherwise use
            object-stored ds)
        dt : float
            Time window (e.g., last 10 Myr - dt = 10.0)

        Returns
        -------
        sfr : float
        '''

        if ds is None:
            ds = self.ds
        if ad is None:
            ad = self.ad

        # Get star particle masses and formation times
        masses = self.ad[("star", "particle_mass")].to("Msun")

        current_ages = self.get_star_ages()

        # Select recently formed stars
        young = current_ages < dt

        # Compute SFR
        sfr = masses[young].sum() / (dt * 1e6)  # Msun/yr, all stars 10.0029Msun

        return sfr


    def save_sim_info(self, ds=None):
        '''
        Save simulation parameters/information.

        Parameters
        ----------
        ds : yt.Dataset
            loaded RAMSES-RT data set (optionally specify; otherwise use
            object-stored ds)

        Returns
        -------
        None
        '''

        if ds is None:
            ds = self.ds

        self.current_time = ds.current_time
        self.domain_dimensions = ds.domain_dimensions
        self.domain_left_edge = ds.domain_left_edge
        self.domain_right_edge = ds.domain_right_edge
        self.cosmological_simulation = ds.cosmological_simulation
        #self.current_redshift = ds.current_redshift
        self.omega_lambda = ds.omega_lambda
        self.omega_matter = ds.omega_matter
        self.omega_radiation = ds.omega_radiation
        self.hubble_constant = ds.hubble_constant
        self.sfr_10 = self.calc_sfr(10.0)
        self.sfr_5 = self.calc_sfr(5.0)
        self.sfr_1 = self.calc_sfr(1.0)

        file_path = os.path.join(self.directory, 
                                f'{self.output_file}_sim_info.txt')
        
        with open(file_path, 'w') as file:
            file.write(f'current_time: {self.current_time}\n')
            file.write(f'domain_dimensions: {self.domain_dimensions}\n')
            file.write(f'domain_left_edge: {self.domain_left_edge}\n')
            file.write(f'domain_right_edge: {self.domain_right_edge}\n')
            file.write(f'cosmological_simulation: ' +
                       f'{self.cosmological_simulation}\n')
            file.write(f'current_redshift: {self.redshift}\n')
            file.write(f'omega_lambda: {self.omega_lambda}\n')
            file.write(f'omega_matter: {self.omega_matter}\n')
            file.write(f'omega_radiation: {self.omega_radiation}\n')
            file.write(f'hubble_constant: {self.hubble_constant}\n')
            file.write(f'sfr_10: {self.sfr_10}\n')
            file.write(f'sfr_5: {self.sfr_5}\n')
            file.write(f'sfr_1: {self.sfr_1}\n')

        # Copy information files from data folder to analysis
        # TODO logSFC
        sim_info_files = [
            os.path.join(self.file_dir, f'header_{self.sim_run}.txt'),
            os.path.join(self.file_dir, 'hydro_file_descriptor.txt'),
            os.path.join(self.file_dir, f'info_{self.sim_run}.txt'),
            os.path.join(self.file_dir, f'info_rt_{self.sim_run}.txt'),
            os.path.join(self.file_dir, 'namelist.txt')
        ]

        for sim_info_file in sim_info_files:
            shutil.copy2(sim_info_file, self.directory)


    def save_sim_field_info(self, sp, ad=None):
        '''
        Save min, max, mean, and aggregate of each field in fields array
        within a sphere sp.

        Parameters
        ----------
        sp : data sphere object
        ad : alldata object
            Optionally specify; otherwise, use object-stored ad.
        '''

        if ad is None:
            ad = self.ad

        fields = [
            ('gas', 'temperature'),
            ('gas', 'density'),
            ('ramses', 'Pressure'),
            ('gas', 'my_H_nuclei_density'),
            ('gas', 'my_temperature'),
            ('gas', 'ion_param'),
            ('gas', 'metallicity'),
            ('gas', 'OII_ratio'),
            ('ramses', 'xHI'),
            ('ramses', 'xHII'),
            ('ramses', 'xHeII'),
            ('ramses', 'xHeIII'),
            ('star', 'particle_mass'),
            ('gas','my_He_number_density'),
            ('gas', 'electron_number_density')
        ]

        for line in self.lines:
            fields.append(('gas', 'flux_'  + line))
            fields.append(('gas', 'luminosity_'  + line))

        # Calculate desired quantities for each field
        field_info = []

        for field in fields:
            min = sp.min(field).value
            print(f'{field}_min: {min}')
            max = sp.max(field).value
            print(f'{field}_max: {max}')
            mean = sp.mean(field).value
            print(f'{field}_mean: {mean}')
            agg = sp.quantities.total_quantity(field).value
            print(f'{field}_agg: {agg}')

            field_info.append((min, max, mean, agg))

        # Save data to a file
        file_path = os.path.join(self.directory, 
                                f'{self.output_file}_field_info.txt')
        
        stellar_mass = \
            ad.quantities.total_quantity(('star', 'particle_mass')).value
        
        with open(file_path, 'w') as file:
            for i, field in enumerate(fields):
                file.write(f'{field}_min: {field_info[i][0]}\n')
                file.write(f'{field}_max: {field_info[i][1]}\n')
                file.write(f'{field}_mean: {field_info[i][2]}\n')
                file.write(f'{field}_agg: {field_info[i][3]}\n')

            file.write(f'Stellar Mass: {stellar_mass}' )

        '''
        Reading the data file example:

        Regex Pattern for float: r'[-+]?\d*\.\d+([eE][-+]?\d+)?'
        Scientific Notation Possible

        import re

        with open('data.txt', 'r') as file:
            file_content = file.read()

        temp_min_pattern = fr'{field}_min: [-+]?\d*\.\d+([eE][-+]?\d+)?'

        temp_min = float(re.search(temp_min_pattern, file_content).group(1)) 
        '''


    #------------------------------------------------------------------
    # Projection/Slice Plots, Plot Wrapper, Phase Plots, Cumulative Sum,
    # Star/Gas Overlay, Panel Plot
    #------------------------------------------------------------------

    def proj_plot(self, sp, width, center, field, weight_field, ds=None):
        '''
        Projection Plot Driver.

        Parameters
        ----------
        sp: sphere data object to project within
        width : Tuple[int, str]
            width in code units or formatted with units, e.g. (1500, 'pc')
        center : List(float)
            center (array of 3 values) in code units
        field : Tuple[str, str]
            field to project, e.g. ('gas', 'temperature')
        weight_field : Tuple[str, str]
            field to weight project (or None if unweighted)
        ds : yt.Dataset
            loaded RAMSES-RT data set (optionally specify; otherwise use
            object-stored ds)

        Returns
        -------
        yt.ProjectionPlot Object
        '''

        if ds is None:
            ds = self.ds

        if weight_field == None:
            p = yt.ProjectionPlot(ds, "z", field,
                          width=width,
                          data_source=sp,
                          buff_size=(self.buff_size, self.buff_size),
                          center=center)
        else:
            p = yt.ProjectionPlot(ds, "z", field,
                          width=width,
                          weight_field=weight_field,
                          data_source=sp,
                          buff_size=(self.buff_size, self.buff_size),
                          center=center)
        return p


    def slc_plot(self, width, center, field, ds=None):
        '''
        Slice Plot Driver.

        Parameters
        ----------
        width (tuple, int and str): width in code units or formatted with 
            units, e.g. (1500, 'pc')
        center (List, float): center (array of 3 values) in code units
        field (tuple, str): field to project, e.g. ('gas', 'temperature')
        ds : yt.Dataset
            loaded RAMSES-RT data set (optionally specify; otherwise use
            object-stored ds)
        
        Returns
        -------
        yt.SlicePlot Object
        '''

        if ds is None:
            ds = self.ds

        slc = yt.SlicePlot(
                        ds, "z", field,
                        center=center,
                        width=width,
                        buff_size=(self.buff_size, self.buff_size))

        return slc
    

    def plot_wrapper(self, sp, width, center, field_list,
                     weight_field_list, title_list, ds=None, proj=True, slc=True,
                     lims_dict=None):
        '''
        Wrapper for plotting a variety of fields simultaneously.

        Parameters
        -----------
        sp : sphere data object to project within
        center : List, float
            center (array of 3 values) in code units
        width : Tuple[int, str]
            width in code units or formatted with units, e.g. (1500, 'pc')
        field_list : List of Tuple[str, str]: 
            list of fields to plot, e.g. ('gas', 'temperature')
        weight_field_list : List of Tuple[str, str]
            list of fields to weight projections (or None if unweighted)
        title_list : List of str
            list of titles associated with plots
        ds : yt.Dataset
            loaded RAMSES-RT data set (optionally specify; otherwise use
            object-stored ds)
        proj : bool
            flag ProjectionPlot
        slc : bool
            flag for SlicePlot
        lims_dict : None or Dict
            dictionary of [vmin, vmax] fixed limits on
            colorbar values for image if desired; otherwise None

        Returns
        --------
        p_img_arr : list of np.ndarray(float)
            list of 2D image arrays
        '''

        if ds is None:
            ds = self.ds

        redshift = self.redshift

        p_img_arr = []

        for i, field in enumerate(field_list):
            if proj:
                p = self.proj_plot(sp, width, center, field, 
                                   weight_field_list[i])
                
                if lims_dict is None:
                    p_img = self.convert_to_plt(p, 'proj', field, width,
                                                'Projected ' + title_list[i])
                else:
                    p_img = self.convert_to_plt(p, 'proj', field, width,
                                                'Projected ' + title_list[i],
                                                lims_dict[field])

            if slc:
                p = self.slc_plot(width, center, field)
                
                if lims_dict == None:
                    p_img = self.convert_to_plt(p, 'slc', field, width,
                                                redshift,
                                                title_list[i])
                else:
                    p_img = self.convert_to_plt(p, 'slc', field, width,
                                                redshift, title_list[i],
                                                lims_dict[field])
                    
            p_img_arr.append(p_img)
        
        return p_img_arr
                    

    def phase_plot(self, sp, x_field, y_field, z_field, extrema,
                   x_label, y_label, z_label):
        '''
        Generate a phase plot.
        
        Parameters
        -----------
        sp : sphere data object to project within
        x_field : Tuple[str, str]
            field to plot on the x-axis, i.e. ('gas', 'my_H_nuclei_density')
        y_field : Tuple[str, str]
            field to plot on the y-axis, i.e. ('gas', 'my_temperature')
        z_field Tuple[str, str]
            field to plot with colormap, i.e. ('gas', 'flux_H1_6562.80A')
        extrema : Dict[Tuple[str, str], Tuple[float, float]]
            Dictionary specifying the extrema of the plot, i.e.
            extrema = {('gas', 'my_H_nuclei_density'): (1e-4, 1e4), 
                            ('gas', 'my_temperature'): (1e3, 1e8)}
        x_label : str
            label for x-axis
        y_label : str
            label for y-axis
        z_label : str
            label for colorbar

        Returns
        --------
        phase_profile: profile associated with PhasePlot object
            Can extract attributes
        x_vals : np.ndarray(float)
            x values associated with phase plot
        y_vals : np.ndarray(float)
            y values associated with phase plot
        z_vals : np.ndarray(float)
            2D z values associated with phase plot
        '''

        plot_title = f'{self.output_file}_' + \
            f'{x_field[1]}_{y_field[1]}_{z_field[1]}_phase.png'
        
        phase_path = os.path.join(self.directory, 'phase_plots')
        #fits_path = os.path.join(plt_path, '/fits')
        if not os.path.exists(phase_path):
            os.makedirs(phase_path)
        #if not os.path.exists(fits_path):
        #    os.makedirs(fits_path)

        fname = os.path.join(phase_path, plot_title)

        profile = yt.create_profile(
            sp,
            #ds.all_data(), # TODO
            [x_field, y_field],
            #n_bins=[128, 128],
            fields=[z_field],
            weight_field=None,
            #units=units,
            extrema=extrema,
        )

        plot = yt.PhasePlot.from_profile(profile)

        phase_profile = plot.profile

        plot.set_colorbar_label(z_field, z_label)
        plot.render()

        x_vals = phase_profile.x
        y_vals = phase_profile.y
        z_vals = phase_profile[z_field]  # alternatively field_data attr
        #print(x_vals.shape)
        #print(y_vals.shape)
        #print(z_vals.shape)

        #p_img = np.reshape(z_vals, (len(x_vals)-1, len(y_vals)-1))
        # TODO may need to average in each bin

        # Get a reference to the matplotlib axes object for the plot
        ax = plot.plots[z_field[0], z_field[1]].axes
        fig = plot.plots[z_field[0], z_field[1]].figure
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plot.save(fname)

        # TODO save fits/data values
        return (phase_profile, x_vals, y_vals, z_vals)


    def phase_with_profiles(self, x_field, y_field, z_field,
                        x_vals, y_vals, z_vals, x_label, y_label,
                        z_label, linear=False):
        '''
        Generate a phase plot with additional profile plots.

        Parameters
        -----------
        x_field : Tuple[str, str]
            field to plot on the x-axis, i.e. ('gas', 'my_H_nuclei_density')
        y_field : Tuple[str, str]
            field to plot on the y-axis, i.e. ('gas', 'my_temperature')
        z_field : Tuple[str, str]
            field to plot with colormap, i.e. ('gas', 'flux_H1_6562.80A')
        x_vals : np.ndarray(float)
            x values associated with phase plot
        y_vals : np.ndarray(float)
            y values associated with phase plot
        z_vals : np.ndarray(float)
            z values associated with phase plot
        x_label : str
            label for x-axis
        y_label : str
            label for y-axis
        z_label : str
            label for colorbar
        linear : bool
            flag to plot profiles linear (True) or logarithmically (False)

        Returns
        --------
        TODO

        TODO lims on profiles, z
        TODO save fits
        '''

        plot_title = f'{self.output_file}_' + \
                    f'{x_field[1]}_{y_field[1]}_{z_field[1]}_phase_profile.png'
        
        phase_path = os.path.join(self.directory, 'phase_plots')
        if not os.path.exists(phase_path):
            os.makedirs(phase_path)

        fname = os.path.join(phase_path, plot_title)

        # Logarithmic scaling of the data
        x_vals = np.log10(x_vals)
        y_vals = np.log10(y_vals)
        z_vals = np.log10(z_vals).transpose()

        # Find the location of the peak z value
        peak_z_idx = np.unravel_index(np.argmax(z_vals), z_vals.shape)[::-1]
        peak_x = x_vals[peak_z_idx[0]]
        peak_y = y_vals[peak_z_idx[1]]
        peak_z = z_vals[peak_z_idx[1]][peak_z_idx[0]]

        # Create the figure and gridspec layout
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(4, 4, figure=fig)

        # Central phase plot (imshow), takes larger area
        ax0 = fig.add_subplot(gs[1:4, 0:3])
        cax = ax0.imshow(z_vals, origin="lower", aspect="auto",
                         extent=(min(x_vals), max(x_vals), min(y_vals),
                                 max(y_vals)))
        ax0.set_xlabel(x_label, fontsize=16)
        ax0.set_ylabel(y_label, fontsize=16)
        ax0.scatter(peak_x, peak_y, color="red",
                    label=f"Peak ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f})")
        ax0.legend(loc="upper right")

        # Profile plot at the top (z vs x), touching top border of phase plot
        ax1 = fig.add_subplot(gs[0, 0:3])
        avg_z_vals_x = np.mean(10 ** z_vals, axis=0)#[::-1]
        avg_z_vals_x[avg_z_vals_x < 1e-30] = 1e-30
        if linear:
            ax1.plot(x_vals, avg_z_vals_x, color="blue")

            # Add a red dot at the peak location on the top profile
            ax1.scatter(
                peak_x, avg_z_vals_x[np.argmax(x_vals == peak_x)],
                color='red', s=50)
        else:
            ax1.plot(x_vals, np.log10(avg_z_vals_x), color="blue")

            # Add a red dot at the peak location on the top profile
            ax1.scatter(
                peak_x, np.log10(avg_z_vals_x[np.argmax(x_vals == peak_x)]),
                color='red', s=50)
        
        # Ensure x-axis of top profile matches phase plot, remove x-ticks
        ax1.set_xlim(ax0.get_xlim()) 
        ax1.tick_params(axis='x', which='both', bottom=False, top=False)
        ax1.set_xticklabels([])

        # Profile plot on the right (z vs y), touching right border of phase
        ax2 = fig.add_subplot(gs[1:4, 3])
        avg_z_vals_y = np.mean(10 ** z_vals, axis=1)#[::-1]
        avg_z_vals_y[avg_z_vals_y < 1e-30] = 1e-30

        if linear:
            ax2.plot(avg_z_vals_y, y_vals, color="blue")
    
            # Add a red dot at the peak location on the right profile
            ax2.scatter(
                avg_z_vals_y[np.argmax(y_vals == peak_y)], peak_y,
                color='red', s=50)

        else:
            ax2.plot(np.log10(avg_z_vals_y), y_vals, color="blue")

            # Add a red dot at the peak location on the right profile
            ax2.scatter(
                np.log10(avg_z_vals_y[np.argmax(y_vals == peak_y)]), peak_y,
                color='red', s=50)
        
        # Ensure y-axis of right profile matches phase plot,
        # Remove y-axis ticks
        ax2.set_ylim(ax0.get_ylim())
        ax2.tick_params(axis='y', which='both', left=False, right=False)
        ax2.set_yticklabels([])

        # Adjust layout and position the colorbar
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        # Add colorbar on the right side of the profile plot
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
        cbar.set_label(z_label, size=16)

        #z_total = sp.quantities.total_quantity(z_field).value
        #annotation_text = f'Total: {z_total:.4f}'
        #fig.text(0.95, 0.95, annotation_text, ha='right', va='top',
        #         fontsize=12, color='black')

        # Save the figure
        plt.savefig(fname, dpi=300)
        plt.close()

        # TODO z extrema


    def phase_plot_wrapper(self, sp, config_list):
        '''
        Generate a series of phase plots and phase plots with profiles.

        Parameters
        -----------
        sp : sphere data object to project within
        config_list : List[Dict]
            A list of config Dicts with the following information:

            x_field : Tuple[str, str]
                field to plot on the x-axis, i.e. ('gas', 'my_H_nuclei_density')
            y_field : Tuple[str, str]
                field to plot on the y-axis, i.e. ('gas', 'my_temperature')
            z_field : Tuple[str, str]
                field to plot with colormap, i.e. ('gas', 'flux_H1_6562.80A')
            x_label : str
                label for x-axis
            y_label : str
                label for y-axis
            z_label : str
                label for colorbar
            linear : bool
                flag to plot profiles linear (True) or logarithmically (False)

            Intermediate Values:
            x_vals : np.ndarray(float)
                x values associated with phase plot
            y_vals : np.ndarray(float)
                y values associated with phase plot
            z_vals : np.ndarray(float)
                z values associated with phase plot

        Returns
        --------
        TODO
        Add sp to config list
        '''

        for config in config_list:
            x_field = config['x_field']
            y_field = config['y_field']
            z_field = config['z_field']
            extrema = config['extrema']
            x_label = config['x_label']
            y_label = config['y_label']
            z_label = config['z_label']
            linear = config['linear']

            phase_profile, x_vals, y_vals, z_vals = \
                self.phase_plot(sp, 
                    x_field=x_field,
                    y_field=y_field,
                    z_field=z_field,
                    extrema=extrema,
                    x_label=x_label, 
                    y_label=y_label, 
                    z_label=z_label)
            
            self.phase_with_profiles( 
                    x_field=x_field,
                    y_field=y_field,
                    z_field=z_field,
                    x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
                    x_label=x_label, 
                    y_label=y_label, 
                    z_label=z_label,
                    linear=linear
            )

        return phase_profile, x_vals, y_vals, z_vals


    def plot_cumulative_field(self, sp, fields, titles, fname,
                              idx_lims=None):
        '''
        Flatten and order the values in an image of a field 

        Parameters
        -----------
        sp: sphere data object to project within
        fields : List of Tuple[str, str]
            fields to plot, e.g. ('gas', 'temperature')
        titles : List of str 
            titles
        fname : str
            figure name
        idx_lims : Tuple[int, int]
            range of indices/cells to plot
        '''

        fig = plt.figure(figsize=(8, 6))
        plt.xlabel('Index')
        plt.ylabel('Cumulative Value')
        plt.title(f'Cumulative Sum')

        for i, field in enumerate(fields):
            pix = sp[field].value
            pix_sort = np.sort(pix, axis=None)[::-1]
            idxs = np.arange(0, len(pix_sort), 1)
            cum_val = np.cumsum(pix_sort) / np.sum(pix_sort)

            if idx_lims is not None:
                idxs = idxs[idx_lims[0]: idx_lims[1]]
                cum_val = cum_val[idx_lims[0]: idx_lims[1]]
        
            plt.plot(idxs, cum_val, label=titles[i])
        
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.directory,
                         f'output_{self.sim_run}_{fname}.png'), dpi=300
        )
        plt.close()


    def star_gas_overlay(self, sp, center, width, field, gas_title,
                         gas_flag=False, lims_dict=None, ds=None, ad=None):
        '''
        Plot stellar density and, optionally, a field of the gas overlayed.

        Star + Gas Plot
        Adapted from work by Sarunyapat Phoompuang  
        
        Parameters
        -----------
        sp : sphere data object to project within
        center : List(float)
            center (array of 3 values) in code units
        width : Tuple[int, str]
            width in code units or formatted with units, e.g. (1500, 'pc')
        field : TODO
        gas_title : str
            title for overlay plot, 
            i.e. r'H$\alpha$ Flux [$erg\: s^{-1}\: cm^{-2}$]'
        gas_flag : bool
            choose whether to plot gas overlay in addition to stellar
            mass density
        lims_dict : None or Dict
            dictionary of [vmin, vmax] fixed limits on
            colorbar values for image if desired; otherwise None
        ds: yt.Dataset
            loaded RAMSES-RT data set. Optionally specify, otherwise


        Returns
        --------
        None
        '''

        if ds is None:
            ds = self.ds
        if ad is None:
            ad = self.ad

        redshift = self.redshift
        lbox = width[0]

        #lims = lims_dict[field[1]]

        fname = os.path.join(self.directory, self.output_file + '_' +
                             str(width[0]) + width[1] + '_stellar_dist')
        
        # Finding center of the data
        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)
        x_pos = x_pos - x_center
        y_pos = y_pos - y_center
        z_pos = z_pos - z_center

        # Create a ProjectionPlot
        p = yt.ProjectionPlot(ds, "z", field,
                          width=width,
                          data_source=sp,
                          buff_size=(self.buff_size, self.buff_size),
                          center=center)
        print(field)

        # Fixed Resolution Buffer
        p_frb = p.frb
        p_img = np.array(p_frb[field[0], field[1]])
        star_bins = 2000
        star_mass = np.ones_like(x_pos) * 10
        #pop2_xyz = np.array(
        #    ds.arr(np.vstack([x_pos, y_pos, z_pos]),
        #           "code_length").to("pc")).T
        #pop2_xyz = np.array(ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")).T
        pop2_xyz = np.vstack([x_pos, y_pos, z_pos]) * ds.length_unit.in_units("pc").value
        pop2_xyz = pop2_xyz.T
        extent_dens = [-lbox/2, lbox/2, -lbox/2, lbox/2]
    
        stellar_mass_dens, _, _ = \
            np.histogram2d(pop2_xyz[:, 0], pop2_xyz[:, 1],
                           bins = star_bins,weights = star_mass,
                           range = [[-lbox / 2, lbox / 2],
                                    [-lbox / 2, lbox / 2],],
        )
        stellar_mass_dens = stellar_mass_dens.T
        stellar_mass_dens = np.where(stellar_mass_dens <= 1, 0,
                                     stellar_mass_dens)
        stellar_range = [1, 1200]
        norm2 = LogNorm(vmin = stellar_range[0], vmax = stellar_range[1])
        plt.figure(figsize = (8, 6))
        lumcmap = "cmr.amethyst"
        plt.imshow(stellar_mass_dens, norm = norm2, extent = extent_dens,
                   origin = 'lower', aspect = 'auto', cmap = 'winter_r')
        cbar = plt.colorbar(pad=0.04)
        cbar.set_label('Stellar Mass Density', size=16)
        plt.xlabel("X (pc)", fontsize=16)
        plt.ylabel("Y (pc)", fontsize=16)
        #plt.title("Stellar Mass Density Distribution")
        plt.text(0.05, 0.05, f'z = {redshift:.5f}', color='black', fontsize=9,
                 ha='left', va='bottom', transform=plt.gca().transAxes)
        plt.savefig(fname=fname)
        plt.close()
    	
        if gas_flag:
            # Check for min/max values of p_img
            #print(np.min(p_img), np.max(p_img))
            # Check for min/max values of stellar_mass_dens
            #print(np.min(stellar_mass_dens), np.max(stellar_mass_dens))  

            #gas_range = (20, 2e4)
            lims = lims_dict[field]
            norm1 = LogNorm(vmin=lims[0], vmax=lims[1])

            overlay_fname = fname + field[1] + '.png'
            fig, ax = plt.subplots(figsize = (12, 8))
            alpha_star = stellar_mass_dens
            alpha_star = np.where(stellar_mass_dens <= 1, 0.0, 1)

            #print(alpha_star.shape)
            #print(p_img.shape)

            img1 = ax.imshow(p_img, norm = norm1, extent = extent_dens,
                             origin = 'lower', aspect = 'auto',
                             cmap = 'inferno',
                             alpha = 1, interpolation='bilinear')
            cbar1 = fig.colorbar(img1, ax = ax, orientation = 'vertical',
                                 pad = 0.04)
            cbar1.set_label('Projected ' + gas_title, size=16)
            img2 = ax.imshow(stellar_mass_dens, norm = norm2,
                             extent = extent_dens, origin = 'lower',
                             aspect = 'auto', cmap = 'winter_r',
                             interpolation='bilinear')

            # Make sure alpha_star matches the image shape (it must be the same
            # size as the image)
            if img2.get_array().shape != alpha_star.shape:
                print(f'Shape mismatch: Image shape {img2.get_array().shape}' +
                      f' vs. alpha_star shape {alpha_star.shape}')

            # Apply alpha mask after plotting
            img2.set_alpha(alpha_star)

            cbar2 = fig.colorbar(img2, ax = ax, orientation = 'vertical',
                                 pad = 0.04)
            cbar2.set_label("Stellar Mass Density", size=16)
            # ax.scatter(pop2_xyz[:, 0], pop2_xyz[:, 1], s=5, marker='.', color='black')
            ax.set_xlabel("X (pc)", fontsize=16)
            ax.set_ylabel("Y (pc)", fontsize=16)
            #ax.set_title(gas_title + ' and Stellar Mass Density Distribution')
            ax.set_xlim(-lbox / 2, lbox / 2)
            ax.set_ylim(-lbox / 2, lbox / 2)

            plt.text(0.05, 0.05, f'z = {redshift:.5f}', color='white',
                     fontsize=9,
                     ha='left', va='bottom', transform=plt.gca().transAxes)

            plt.savefig(fname=overlay_fname)
            plt.close()


    def panel_plot(self,
               sp,
               panel_config,
               width,
               center,
               nrows=2,
               ncols=2,
               ds=None,
               lims_dict=None,
               log=True,
               filename="panel"):
        '''
        Generate a mixed multi-panel figure of projection and slice plots.

        Each panel can independently specify:
            - field
            - plot_type ('projection' or 'slice')
            - weight_field (optional)

        Parameters
        ----------
        sp : yt data container
            Region of interest (used for projections).
        panel_config : list of dict
            Each dict must contain:
                {
                  "field": ('gas','density'),
                  "plot_type": "projection" or "slice",
                  "weight_field": optional tuple,
                  "title": str title
                }
        width : tuple
            Plot width (value, unit).
        center : ndarray
            Plot center.
        nrows : int
            Number of panel rows.
        ncols : int
            Number of panel columns.
        ds : yt.Dataset
            Loaded dataset.
        lims_dict : dict, optional
            Dictionary mapping field -> (vmin, vmax).
        log : bool
            Logarithmic scaling.
        filename : str
            Base output filename.

        Returns
        -------
        images : dict
            Dictionary mapping field name -> image array.
        '''

        # TODO add redshift to bottom left

        if ds is None:
            ds = self.ds

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6*ncols, 5*nrows))

        axes = axes.flatten()
        images = {}

        for ax, config in zip(axes, panel_config):

            field = config["field"]
            plot_type = config.get("plot_type", "projection")
            weight_field = config.get("weight_field", None)

            #------------------------------------------
            # Generate image depending on plot type
            #------------------------------------------

            if plot_type.lower() == "projection":

                proj = yt.ProjectionPlot(ds, "z", field,
                                         center=center,
                                         width=width,
                                         weight_field=weight_field,
                                         data_source=sp,
                                         buff_size=(self.buff_size, self.buff_size))

                # TODO check this 
                frb = proj.data_source.to_frb(width, self.buff_size)
                image = np.array(frb[field]).astype(np.float32)

            elif plot_type.lower() == "slice":

                slc = yt.SlicePlot(ds, "z", field,
                                   center=center,
                                   width=width,
                                   buff_size=(self.buff_size, self.buff_size))

                frb = slc.data_source.to_frb(width, self.buff_size)
                image = np.array(frb[field]).astype(np.float32)

            else:
                raise ValueError("plot_type must be 'projection' or 'slice'")

            images[field[1]] = image

            #------------------------------------------
            # Normalisation
            #------------------------------------------

            lims = None
            if lims_dict is not None:
                lims = lims_dict.get(field)

            norm = self.get_norm(image, lims=lims, log=log)

            im = ax.imshow(image,
                           origin="lower",
                           norm=norm)

            cbar = fig.colorbar(im, ax=ax)
            #cbar.set_label(field[1])
            cbar.set_label(config["title"], size=14)

            # TODO change titles
            #ax.set_title(f"{field[1]} ({plot_type})")
            ax.set_xlabel(f"X [{width[1]}]", fontsize=16)
            ax.set_ylabel(f"Y [{width[1]}]", fontsize=16)

            #------------------------------------------
            # Save FITS per panel
            #------------------------------------------
            panel_path = os.path.join(self.directory, 'panel_plots')
            fits_path = os.path.join(panel_path, 'panel_fits')

            if not os.path.exists(panel_path):
                os.makedirs(panel_path)
            if not os.path.exists(fits_path):
                os.makedirs(fits_path)

            if self.minimal_output is False:
                self.write_fits_image(
                    image,
                    f"{fits_path}/{filename}_{field[1]}.fits",
                    field=str(field),
                    width=width,
                    center=center,
                    redshift=ds.current_redshift
                )

        # Turn off unused axes
        for ax in axes[len(panel_config):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(f"{panel_path}/{filename}.pdf", dpi=300)
        if self.minimal_output is False:
            fig.savefig(f"{panel_path}/{filename}.png", dpi=300)
        plt.close(fig)

        return images
    

    #------------------------------------------------------------------
    # Spectra Generation TODO
    #------------------------------------------------------------------

    def spectra_driver(self, resolving_power, noise_lvl,
                       lum_lims=None, flux_lims=None, linear=False):
        '''
        Generate spectra.

        Parameters
        -----------
        resolving_power (float): resolving power R = lambda/delta_lambda
            for the observational system. i.e. R = 1000
        noise_lvl (float): noise level/lower floor on signal, i.e. 10e-25
        lum_lims (List, float): manual limits on the luminosity values,
            i.e. lum_lims=[32, 44]
        flux_lims (List, float): manual limits of the flux values
            i.e. flux_lims=[-24, -19]
        '''

        spectra_path = os.path.join(self.directory, 'spectra')
        
        if not os.path.exists(spectra_path):
            os.makedirs(spectra_path)

        cosmo = FlatLambdaCDM(H0=70, Om0=self.omega_matter)  # around 0.3
        
        # Mpc to cm
        d_1 = cosmo.luminosity_distance(self.redshift)*3.086e24
        #self.flux_arr = (self.luminosities / (4 * np.pi * d_1 ** 2)).value
        self.flux_arr = np.array(self.luminosities) / (4 * np.pi * d_1.value ** 2)

        fname = os.path.join(spectra_path, self.output_file)

        # Raw spectra values
        self.plot_spectra(noise_lvl, resolving_power, 1000,
                          fname + '_raw_spectra', sim_spectra=False,
                          redshift_wavelengths=False)

        # Sim spectra, not redshifted
        self.plot_spectra(noise_lvl, resolving_power, 1000,
                          fname + '_sim_spectra', sim_spectra=True,
                          redshift_wavelengths=False)

        # Sim spectra, redshifted
        self.plot_spectra(noise_lvl, resolving_power, 1000,
                          fname + '_sim_spectra_redshifted', sim_spectra=True,
                          redshift_wavelengths=True)


        # With limits for animation
        # Sim spectra, not redshifted
        self.plot_spectra(noise_lvl, resolving_power, 1000,
                          fname + '_sim_spectra', sim_spectra=True,
                          redshift_wavelengths=False,
                          lum_lims=lum_lims, flux_lims=flux_lims)

        # Sim spectra, redshifted
        self.plot_spectra(noise_lvl, resolving_power, 1000,
                          fname + '_sim_spectra_redshifted', sim_spectra=True,
                          redshift_wavelengths=True,
                          lum_lims=lum_lims, flux_lims=flux_lims)


    def plot_spectra(self, noise_lvl, resolving_power, pad, figname,
                     sim_spectra=False, redshift_wavelengths=False,
                     lum_lims=None, flux_lims=None, linear=False):
        '''
        Plot a spectrum with certain options.

        Parameters:
        -----------
        noise_lvl (float): noise level/lower floor on signal
        resolving_power (float): resolving power
        pad (float): pad on wavelengths around each voigt profile, i.e. 1000A
        figname (str): filename of figure
        sim_spectra (bool): option to simulate spectra with voigt profiles.
            If False values are plotted in a scatter plot.
        redshift_wavelengths (bool): option to account for redshift in
            wavelengths on the x-axis.
        lum_lims (List, float): manual limits on the luminosity values
        flux_lims (List, float): manual limits of the flux values
        linear (bool): option to plot on linear rather than log y-axis
        '''

        wavelengths = self.wavelengths

        # Display spectra at redshifted wavelengths
        # lambda_obs = (1+z)*lambda_rest
        if redshift_wavelengths:
            wavelengths = (1 + self.redshift) * np.array(wavelengths)
            pad *= 5

        line_widths = np.array(wavelengths) / resolving_power  # Angstroms

        if sim_spectra:
            x_range, y_vals_f = self.plot_voigts(wavelengths, self.flux_arr,
                                                 line_widths,
                                                 [0.0]*len(wavelengths),
                                                 noise_lvl, pad)
            
            fig, ax1 = plt.subplots(1)
            
            if not linear:
                ax1.plot(x_range, np.log10(y_vals_f), color='black')
            else:
                ax1.plot(x_range, y_vals_f, color='black')

            if flux_lims != None:
                if not linear:
                    ax1.set_ylim(flux_lims)
                else:
                    ax1.set_ylim([10**flux_lims[0], 10**flux_lims[1]])

            ax1.set_xlabel(r'Wavelength [$\AA$]', fontsize=12)
            if not linear:
                ax1.set_ylabel(
                    r'Log(Flux) [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]', fontsize=12
                )
            else:
                ax1.set_ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]', fontsize=12)

            # Add redshift
            plt.text(0.95, 0.95, f'z = {self.redshift:.5f}', color='black',
                     fontsize=9, ha='right', va='top',
                     transform=plt.gca().transAxes)

            flux_fname = figname + '_flux'
            if self.minimal_output is False:
                plt.savefig(f"{flux_fname}.pdf", dpi=300)
            plt.savefig(f"{flux_fname}.png", dpi=300)
            #plt.savefig(flux_fname)
            plt.close()

            fig, ax1 = plt.subplots(1)
            x_range, y_vals_l = self.plot_voigts(wavelengths, self.luminosities,
                                                 line_widths,
                                                 [0.0]*len(wavelengths),
                                                 noise_lvl, pad)
            
            if not linear:
                ax1.plot(x_range, np.log10(y_vals_l), color='black')
            else:
                ax1.plot(x_range, y_vals_l, color='black')

            if lum_lims != None:
                if linear == False:
                    ax1.set_ylim(lum_lims)
                else:
                    ax1.set_ylim([10**lum_lims[0], 10**lum_lims[1]])

            ax1.set_xlabel(r'Wavelength [$\AA$]', fontsize=12)
            if not linear:
                ax1.set_ylabel(
                    r'Log(Luminosity) [erg s$^{-1}$ $\AA^{-1}$]', fontsize=12
                )
            else:
                ax1.set_ylabel(r'Luminosity [erg s$^{-1}$ $\AA^{-1}$]', fontsize=12)

            lum_fname = figname + '_lum'
            if self.minimal_output is False:
                plt.savefig(f"{lum_fname}.pdf", dpi=300)
            plt.savefig(f"{lum_fname}.png", dpi=300)
            #plt.savefig(lum_fname)
            plt.close()

        else:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            ax1.plot(wavelengths, np.log10(self.flux_arr), 'o')
            ax2.plot(wavelengths, np.log10(self.luminosities), 'o')
            ax2.set_xlabel(r'Wavelength [$\AA$]', fontsize=12)
            ax1.set_ylabel(r'Log(Flux) [erg s$^{-1}$ cm$^{-2}$]', fontsize=12)
            ax2.set_ylabel(r'Log(Luminosity) [erg s$^{-1}$]', fontsize=12)
            if self.minimal_output is False:
                plt.savefig(f"{figname}.pdf", dpi=300)
            plt.savefig(f"{figname}.png", dpi=300)
            #plt.savefig(figname)
            plt.close()


    def plot_voigts(self, centers, amplitudes, sigmas, gammas,
                    noise_lvl, pad):
        '''
        Plot voigt profiles for spectral lines over a specified noise level.

        Parameters:
        -----------
        All lists must be of same length.

        centers (list, float): centers of voigt profiles
        amplitudes (list, float): corresponding amplitudes (i.e.,
            luminosities) for each profile
        sigmas (list, float): list of associated standard deviations of
            a normal distribution
        gammas (list, float): list of associated FWHM of Cauchy distribution
        noise_lvl (float): noise level/lower floor on signal
        pad (float): pad on wavelengths around each voigt profile, i.e. 1000A

        Returns:
        ----------
        x_range (array, float): array of x values (wavelengths)
        y_vals (array, float): array of accumulated y values (i.e., the
            sum of luminosities or fluxes from each voigt profile)
        '''

        # TODO noiseless profile, Poisson noise
        # TODO gas kinematics, blur with telescope

        x_range = np.linspace(min(centers) - pad, max(centers) + pad, 1000)
        y_vals = np.zeros_like(x_range) + noise_lvl

        for amp, center, sigma, gamma in \
            zip(amplitudes, centers, sigmas, gammas):
            y_vals += (amp) * voigt_profile(x_range - center, sigma, gamma)

            #if amp > noise_lvl:
                #y_vals += (amp-noise_lvl)*voigt_profile(x_range - center,
                #   sigma, gamma) # - noise after no sub

        #y_vals += noise_lvl

        return x_range, y_vals
    

# Lims Dict in Class
# TODO consistent fontsize

# TODO change panel plot frb approach
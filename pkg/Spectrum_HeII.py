"""
S.N. Kolwa
ESO (2019) 

"""

import numpy as np
import matplotlib.pyplot as pl
import matplotlib
from math import*

import mpdaf.obj as mpdo

from lmfit import *
import lmfit.models as lm

from Gaussian import * 

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ('ignore', category=AstropyWarning         )


class Spectrum_HeII:

	def __init__( self, source=None, plot_dir=None ):
		"""
		Parameters 
		----------
		source : Source name

		plot_dir : Dirtectory for output plots

		""" 
		self.source = source
		self.plot_dir = plot_dir
		self.c = 2.9979245800e5 	#speed of light in km/s

	def convert_wav_to_vel( self, wav_obs, wav_em, z ):
		"""
		Convert an observed wavelength to a velocity 
		in the observer-frame at redshift
	
		Parameters 
		----------
		wav_obs : Observed wavelength
	
		wav_em : Emitted wavelength
	
		z : Observer Redshift
	
		Returns 
		-------
		velocity (km/s)
		
		"""
		v = self.c*((wav_obs/wav_em/(1.+z)) - 1.)
		return v

	def get_redshift( self, dec, ra, size, lam1, lam2, muse_file, 
		p, wav_em):
		"""
		Calculate the systemic redshift from a line in the MUSE spectrum

		Parameters 
		----------
		dec : DEC (pixel) of aperture centre for extracted MUSE spectrum

		ra : RA (pixel) of aperture centre for extracted MUSE spectrum

		size : Radius of aperture for extracted MUSE spectrum

		lam1 : Wavelength (Angstroms) at the lower-end of spectral range 
			of the subcube

		lam2 : Wavelength (Angstroms) at the upper-end of spectral range 
			of the subcube

		muse_file : Path and filename of MUSE datacube

		p : Initial guesses for fit parameters

		wav_em : Rest wavelength of HeII 1640

		source : Name of source

		Returns 
		-------
		Systemic redshift of the galaxy and its velocity : list
		
		"""
		#Import MUSE data
		muse_cube = mpdo.Cube(muse_file, ext=1)
		muse_var = mpdo.Cube(muse_file, ext=2)
		
		# Make HeII subcube
		m1,m2 	= muse_cube.sum(axis=(1,2)).wave.pixel([lam1,lam2], nearest=True) 
		muse_cube = muse_cube[ m1:m2, :, : ]
		muse_var = muse_var[ m1:m2, :, : ]

		# Aperture subcube for 1D spectral extraction
		subarp_data = muse_cube.subcube_circle_aperture( (dec, ra), size, unit_center=None, 
			unit_radius=None )
		subarp_var = muse_var.subcube_circle_aperture( (dec, ra), size, unit_center=None, 
			unit_radius=None )

		# data spectrum
		spec = subarp_data.sum(axis=(1,2))
		wav = list(spec.wave.coord())
		flux = spec.data
		normed_flux = [ flux[i]/max(flux) for i in range(len(flux)) ]

		# variance spectrum
		spec_var  = subarp_var.sum(axis=(1,2))
		var_flux  = spec_var.data
		normed_var_flux = [ var_flux[i]/max(var_flux) for i in range(len(var_flux)) ]
		inv_noise = [ 1./normed_var_flux[i] for i in range(len(var_flux)) ] 

		# ignore sky line in 4C +03.24
		if p[0] == 7488.:
			skyline_ind = [ i for i in range(len(wav)) if (wav[i] > 7518. and wav[i] < 7526.) ]

			for ind in skyline_ind: 
				normed_flux[ind] = 0.	
				normed_var_flux[ind] = 0.

		pars = Parameters()
		pars.add_many( 
			('g_cen', p[0], True, p[0]-10., p[0]+10.),
			('a', p[1], True, 0.),	
			('wid', p[2], True, 0.),  	#GHz
			('cont', p[3], True ))
		
		mod 	= lm.Model(Gaussian.gauss) 
		fit 	= mod.fit(normed_flux, pars, x=wav , weights=inv_noise)
		print( fit.fit_report() )

		res = fit.params
	
		wav_obs = res['g_cen'].value

		z = (wav_obs/wav_em - 1.) 

		if res['g_cen'].stderr is not None:
			z_err = z*( res['g_cen'].stderr / res['g_cen'].value )*np.median(var_flux)

		else:
			z_err = 0.0

		vel_glx = self.c*z					#velocity (ref frame of observer)
	
		vel_arr = [ self.convert_wav_to_vel( wav[i], wav_em, z ) for i in range(len(wav)) 	] # at z=z_sys
		
		fig, ax = pl.subplots(2, 1, figsize=(5, 4), sharex=True, 
			constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})
		pl.subplots_adjust(hspace=0, wspace=0.01) 

		matplotlib.rc('axes',edgecolor='grey')
		fs = 12
		ax[0].plot(vel_arr, Gaussian.gauss(wav, res['a'], res['wid'], 
			res['g_cen'], res['cont']), c='red', alpha=0.6)
		ax[0].plot(vel_arr, normed_flux, c='k', drawstyle='steps-mid', fillstyle='right', alpha=0.6)
		ax[0].plot([-2000.,2000.], [0.,0.], c='red', ls='--', alpha=0.6)

		ax[1].plot(vel_arr, normed_var_flux, c='grey', drawstyle='steps-mid', alpha=0.5)
		
		ax[0].set_ylim([-0.05, max(normed_flux) + 0.1])

		if p[0] == 7488.:
			for i in range(2):
				ax[i].axvspan(1100,1400,ymax=1.0,color='#cfdb60', zorder=10, alpha=0.8)

		if p[0] == 6434.:
			ax[1].set_ylim([-0.05, 0.3])
		else: 
			ax[1].set_ylim([-0.05, max(normed_var_flux) + 0.1])
			ax[1].set_yticks(np.arange(0., 1.5, 0.5))

		ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

		ax[0].set_ylabel(r'Norm. Flux', fontsize=fs)
		ax[1].set_ylabel(r'$\sigma$', fontsize=fs)

		for ax in ax:
			ax.set_facecolor('#e8e8e8')
			ax.set_xlim([-2000.,2000.])
			ax.grid(color='white')

			# tick formatting
			ax.set_xticks(np.arange(-2000, 3000, 1000))
			ax.tick_params(direction='in', which='major', length=10,  width=0.5,
	               grid_color='white', grid_alpha=0.4, top=1, right=1)
			ax.tick_params(direction='in', which='minor', length=5,  width=0.5,
	               grid_color='white', grid_alpha=0.4)

		pl.savefig(self.plot_dir+self.source+'_HeII.png')
	
		print( "Systemic redshift ("+self.source+"): %.4f +/- %.4f " %( z, z_err ) 	)
		return [z, z_err, vel_glx]


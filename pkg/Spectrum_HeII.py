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

	def get_redshift( self, dec, ra, size, lam1, lam2, muse_spectrum, 
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
		
		# muse_cube = mpdo.Cube(muse_file, ext=1)
		# muse_var = mpdo.Cube(muse_file, ext=2)
		
		# # Make HeII subcube
		# m1,m2 	= muse_cube.sum(axis=(1,2)).wave.pixel([lam1,lam2], nearest=True) 
		# muse_cube = muse_cube[ m1:m2, :, : ]
		# muse_var = muse_var[ m1:m2, :, : ]

		# # Aperture subcube for 1D spectral extraction
		# subarp_data = muse_cube.subcube_circle_aperture( (dec, ra), size, unit_center=None, 
		# 	unit_radius=None )
		# subarp_var = muse_var.subcube_circle_aperture( (dec, ra), size, unit_center=None, 
		# 	unit_radius=None )

		# data spectrum
		# spec = subarp_data.sum(axis=(1,2))

		spec = mpdo.Spectrum(muse_spectrum, ext=1) 
		spec.info()
		wav1 = spec.wave.coord()
		flux1 = spec.data

		var_spec = mpdo.Spectrum(muse_spectrum, ext=2)
		var_wav1 = var_spec.wave.coord()
		var_flux1 = var_spec.data

		N = len(wav1)

		wav, flux = [], []
		var_wav, var_flux = [], []
		for i, item in enumerate(wav1): 
			if item > lam1 and item < lam2: 
				wav.append( wav1[i] )
				flux.append( flux1[i] )
				var_wav.append( var_wav1[i] )
				var_flux.append( var_flux1[i] )

		normed_flux = [ flux[i]/max(flux) for i in range(len(flux)) ]

		normed_var_flux = [ var_flux[i]/max(var_flux) for i in range(len(var_flux)) ]
		inv_noise = [ 1./normed_var_flux[i] for i in range(len(var_flux)) ]

		print( '-'*len('   '+self.source+'   '))
		print('   '+self.source+'   ')
		print( '-'*len('   '+self.source+'   '))


		if self.source == 'TNJ1338':
			pars = Parameters()
			red_g_cen = p[0] + 20.
			pars.add_many(
				('a1', p[1], True, 0., 10.), 
				('g_cen1', p[0], True, p[0]-5., p[0]+5.),
				('wid1', p[2], True, 0., 20.),  	
				('a2', p[1], True, 0., 10.), 
				('g_cen2', red_g_cen, True, red_g_cen -5., red_g_cen + 5.),
				('wid2', p[2], True, 0., 20.), 
				('cont', p[3], True ), 
				)

			mod 	= lm.Model(Gaussian.dgauss) 
			fit 	= mod.fit(normed_flux, pars, x=wav)
			print( fit.fit_report() )

			res = fit.params
		
			wav_obs, wav_obs_err = res['g_cen1'].value, res['g_cen1'].stderr

			z = (wav_obs/wav_em - 1.) 
			if res['g_cen1'].stderr is not None:
				z_err = z*( wav_obs_err / wav_obs )
			else:
				z_err = 0.0


		else: 
			pars = Parameters()
			pars.add_many(
				('a', p[1], True, 0.), 
				('g_cen', p[0], True, p[0]-5., p[0]+5.),
				('wid', p[2], True, 0., 20.),  	
				('cont', p[3], True ))

			
			mod 	= lm.Model(Gaussian.gauss) 
			fit 	= mod.fit(normed_flux, pars, x=wav)
			print( fit.fit_report() )

			res = fit.params
		
			wav_obs, wav_obs_err = res['g_cen'].value, res['g_cen'].stderr

			z = (wav_obs/wav_em - 1.) 
			if res['g_cen'].stderr is not None:
				z_err = z*( wav_obs_err / wav_obs )
			else:
				z_err = 0.0

		
		vel_glx = self.c*z					#velocity (ref frame of observer)
	
		vel_arr = [ self.convert_wav_to_vel( wav[i], wav_em, z ) for i in range(len(wav)) ] # at z=z_sys
		
		fig, ax = pl.subplots(2, 1, figsize=(6, 4), sharex=True, 
			constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})
		pl.subplots_adjust(hspace=0, wspace=0.01) 

		matplotlib.rc('axes',edgecolor='grey')
		fs = 12

		if self.source == 'TNJ1338': 
			ax[0].plot(vel_arr, Gaussian.dgauss(wav, res['a1'], res['wid1'], 
				res['g_cen1'], res['a2'], res['wid2'], 
				res['g_cen2'], res['cont']), c='red', alpha=0.6)
			ax[0].plot(vel_arr, Gaussian.gauss(wav, res['a1'], res['wid1'], 
				res['g_cen1'], res['cont']), ls='-.')
			ax[0].plot(vel_arr, Gaussian.gauss(wav, res['a2'], res['wid2'], 
				res['g_cen2'], res['cont']), ls='--')

		else: 
			ax[0].plot(vel_arr, Gaussian.gauss(wav, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red', alpha=0.6)

		ax[0].plot(vel_arr, normed_flux, c='k', drawstyle='steps-mid', fillstyle='right', alpha=0.6)
		ax[0].plot([-5000.,5000.], [0.,0.], c='red', ls='--', alpha=0.6)
		ax[1].plot(vel_arr, normed_var_flux, c='grey', drawstyle='steps-mid', alpha=0.5)

		# # set ylim on var plot
		# if self.source in ('MRC0943'):
		# 	ax[1].set_ylim([-0.05, 0.4])

		ax[1].set_ylim([-0.05, max(normed_var_flux) + 0.1])
		ax[1].set_yticks(np.arange(0., 1.5, 0.5))

		ax[0].set_ylim([-0.1, max(normed_flux) + 0.2])
		ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
		ax[0].set_ylabel(r'F$_\lambda$ (norm.)', fontsize=fs)
		ax[1].set_ylabel(r'$\sigma_\lambda$ (norm.)', fontsize=fs)

		for ax in ax:
			ax.set_facecolor('#e8e8e8')
			ax.set_xlim([-2500., 2500.])
			ax.grid(color='white')

			# tick formatting
			ax.tick_params(direction='in', which='major', length=10,  width=0.5,
	               grid_color='white', grid_alpha=0.4, top=1, right=1)
			ax.tick_params(direction='in', which='minor', length=5,  width=0.5,
	               grid_color='white', grid_alpha=0.4)

		pl.savefig(self.plot_dir+self.source+'_HeII.png')
	
		print( "Systemic redshift ("+self.source+"): %.4f +/- %.4f " %( z, z_err ) 	)
		return [z, z_err, vel_glx]
"""
S.N. Kolwa
ESO (2019) 

"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as pl
from math import*

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import Distance, Angle
from astropy.cosmology import Planck15

from itertools import chain 

from lmfit import *
import lmfit.models as lm

import pyregion as pyr
from matplotlib.patches import Ellipse, Rectangle

import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ('ignore', category=AstropyWarning)


class Image_CI:
	
	def __init__( self, CI_path=None, input_dir=None, output_dir=None, plot_dir=None ):
		"""
		Parameters 
		----------
		CI_path : Path of ALMA [CI] datacubes

		input_dir : Directory of input files

		output_dir : Directory of output files

		"""

		self.CI_path = CI_path
		self.input_dir = input_dir
		self.output_dir = output_dir
		self.plot_dir = plot_dir

	def make_narrow_band( self, CI_moment0, 
		CI_rms, regions, dl, beam_pos, source ):
		"""
		Visualise narrow-band ALMA [CI] moment-0 map generated with CASA. 

		Parameters 
		----------
		CI_moment0 : [CI] moment-0 map 

		CI_rms : Minimum threshold value of [CI] contours

		regions : Region names 

		dl : Length of distance scale bar

		source : Source name
		

		Returns 
		-------
		Moment-0 map : image
		
		"""
		[img_arr, wcs, hdr] = self.process_mom0(CI_moment0)

		# Save moment-0 map with colourbar
		fig = pl.figure(figsize=(5.5,4.5))
		ax = fig.add_axes([0.15, 0.12, 0.78, 0.8], projection=wcs)

		# Annotate regions
		regions = [ self.input_dir+i for i in regions]
		
		for x in regions:
			r = pyr.open(x)
			patch, text = r.get_mpl_patches_texts()

			for p in patch:
				ax.add_patch(p)

			for t in text:
				ax.add_artist(t)

		# Add clean/synthesised beam
		pix_deg = abs(hdr['cdelt1'])			# degrees per pixel
		bmaj, bmin = hdr['bmaj'], hdr['bmin']  	# clean beam Parameters
		ellip = Ellipse( beam_pos, (bmaj/pix_deg), (bmin/pix_deg), 0.,
		fc='cyan', ec='black' )
		ax.add_artist(ellip)

		# Add projected distance-scale
		ax.text(80, 10, '10 kpc', color='red', fontsize=10, 
			bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
		ax.plot([83., 83.+dl], [8.5, 8.5], c='red', lw='2', zorder=10.)

		# Change format of right-ascension units
		ra 	= ax.coords[0]
		ra.set_major_formatter('hh:mm:ss.s')

		# Add contours
		[CI_data, CI_wcs, ci_contours] = self.CI_contours(self.CI_path, CI_moment0, CI_rms)

		ax.contour(img_arr, levels=ci_contours*1.e3, colors='k',
		 label='[CI](1-0)', zorder=-5, alpha=0.25, lw=0.2)

		# Optimise image colour-scale of host galaxy
		pix = list(chain(*img_arr))
		pix_rms = np.sqrt(np.mean(np.square(pix)))
		pix_med = np.median(pix)
		vmax = 1.5*(pix_med + pix_rms) 
		vmin = 0.1*(pix_med - pix_rms) 
		CI_map = ax.imshow(img_arr, origin='lower', cmap='viridis', 
			vmin=vmin, vmax=vmax, zorder=-10)

		ax.set_xlabel(r'$\alpha$ (J2000)', size=12)
		ax.set_ylabel(r'$\delta$ (J2000)', size=12)

		ax.set_xlim(8,42)
		ax.set_ylim(8,42)

		cbaxes = fig.add_axes([0.86, 0.12, 0.02, 0.8])
		cb = pl.colorbar(CI_map, cax=cbaxes)
		cb.set_label(r'mJy / beam / GHz',rotation=90, fontsize=10)
		cb.ax.tick_params(labelsize=10)

		#draw MW image boundaries 
		if (source=='4C03' or source=='MRC0943'):
			# hst = Rectangle((10,10), 30, 30, fill=0, color='white', lw=2, alpha=0.6)
			# ax.add_artist(hst)
			# ax.text(10, 8, 'HST FOV', c='white', fontsize=12, weight='bold')

			if source=='4C03':
				muse = Rectangle((15,15), 17, 17, fill=0, color='white', lw=2, alpha=0.6)
				ax.add_artist(muse)
				ax.text(15, 13, 'MUSE FOV', c='white', fontsize=12, weight='bold')

			elif source=='MRC0943': 
				muse = Rectangle((10,10), 30, 30, fill=0, color='white', lw=2, alpha=0.6)
				ax.add_artist(muse)
				ax.text(10, 10, 'MUSE FOV', c='white', fontsize=12, weight='bold')

		else: 
			irac = Rectangle((10,10), 30, 30, fill=0, color='white', lw=2, alpha=0.6)
			ax.add_artist(irac)
			ax.text(18, 8,'IRAC FOV', c='white', fontsize=12, weight='bold')

			muse = Rectangle((15,15), 20, 20, fill=0, color='white', lw=2, alpha=0.6)
			ax.add_artist(muse)
			ax.text(27,13,'MUSE FOV', c='white', fontsize=12, weight='bold')

		pl.savefig(self.plot_dir+source+'_CI_moment0.png', dpi=300)

	def process_mom0( self, CI_moment0 ): 
		# Moment-0 map from CASA
		moment0 = fits.open(self.CI_path+CI_moment0)

		# WCS header
		hdr = moment0[0].header
		wcs = WCS(hdr)
		wcs = wcs.sub(axes=2)

		# Get correct orientation of data
		img_arr = moment0[0].data[0,0,:,:]
		img_arr = np.rot90(img_arr, 1)
		img_arr = np.flipud(img_arr)

		# Convert from Jy/beam to mJy/beam
		img_arr = img_arr[:,:]
		N1,N2 = img_arr.shape[0], img_arr.shape[1]
		img_arr = [[ img_arr[i][j]*1.e3 for i in range(N1) ] for j in range(N2)]
		return [img_arr, wcs, hdr]

	def get_mass( self, z, z_err, SdV, SdV_err, nu_obs, nu_obs_err, fit ):
		"""
		Calculate L_CI, M_CI, M_H2

		Parameters 
		----------
		z : Redshift
			
		z_err : Redshift Error

		SdV : Integrated flux in Jy km/s
			
		SdV_err : Error in integrated flux

		nu_obs : Observed Frequency in GHz

		nu_obs_err : Observed Frequency Error

		fit (bool): True for line detection
			
		Returns 
		-------
		Inferred H_2 mass

		"""
		Dl = (Distance(z=z, unit=u.Mpc, cosmology=Planck15)).value
		X_CI = 3.e-5
		A_10 = 7.93e-8
		Q_10 = 0.48

		L_CI = 3.25e7*SdV*1.e-3*Dl**2/(nu_obs**2*(1+z)**3)  # L' in K km/s pc^2
		L_CI_err = L_CI*np.sqrt( (SdV_err/SdV)**2 + (nu_obs_err/nu_obs)**2 + (z_err/z)**2)

		M_H2 = (1375.8*Dl**2*(1.e-5/X_CI)*(1.e-7/A_10)*SdV*1.e-3)/((1.+z)*Q_10) # solar masses
		M_H2_err = M_H2*np.sqrt( (z_err/z)**2 + (SdV_err/SdV)**2 )

		if fit:
			print('L_CI = %.3e +/- %.3e' %(L_CI, L_CI_err))
			print( 'M_H2/M_sol = %.3e +/- %.3e' %(M_H2, M_H2_err) )
		else: 
			print('L_CI <= %.3e' %L_CI)
			print( 'M_H2/M_sol (upper limit) <= %.3e' %M_H2 )

		return M_H2

	def get_upper_limit( self, CI_datacube, CI_moment0,
		source, region, s, z, z_err, vlims, input_dir ):
		"""
		Visualise narrow-band ALMA [CI] moment-0 map generated with CASA
	
		Parameters 
		----------
			
		CI_spectrum : 1D Spectrum
	
		source : Source name

		CI_rms : Minimum threshold value of [CI] contours

		s : [SFR, SFR upper error, SFR lower error]

		z : Redshift of source
			
		z_err : Redshift error of source

		input_dir : Directory of input files


		Returns 
		-------
		Flux density
		RMS 
		
		"""	
		c = 2.9979245800e5 		#speed of light in km/s
		freq_em = 492.161		#rest frequency of [CI](1-0) in GHz

		print("-"*len("   "+source+" "+region+"   "))
		print("   "+source+" "+region+"   ")
		print("-"*len("   "+source+" "+region+"   "))

		hdu = fits.open(self.CI_path+CI_datacube)
		data = hdu[0].data[0,:,:,:]
		hdr = hdu[0].header	
	
		m0_hdu = fits.open(self.CI_path+CI_moment0)
		m0_hdr = m0_hdu[0].header
	
		# Synthesized beam-size
		bmaj, bmin = m0_hdr['bmaj'], m0_hdr['bmin']
		bmaj, bmin = bmaj*3600, bmin*3600			# arcsec^2
	
		print('bmaj: %.2f arcsec, bmin: %.2f arcsec' %(bmaj, bmin))
	
		# Spectrum for source
		alma_spec = np.genfromtxt(input_dir+source+'_spec_'+region+'_freq.txt')
	
		freq = alma_spec[:,0]			# GHz
		flux = alma_spec[:,1]*1.e3		# mJy
		rms  = alma_spec[:,2]*1.e3		# mJy 
		
		M = len(freq)
		
		# radio velocity
		v_radio = [ c*(1. - freq[i]/freq_em) for i in range(M) ] 
	
		# systemic frequency
		freq_o 		= freq_em/(1. + z) 
		freq_o_err 	= freq_o*( z_err/z )
	
		# systemic velocity
		vel0 = c*(1. - freq_o/freq_em)	
	
		# velocity offset
		voff = [ v_radio[i] - vel0 for i in range(M) ]	

		# indices for velocities within limit
		ind_limit = [ voff.index(v) for v in voff if v > vlims[0] and v < vlims[1] ]

		rms = [ rms[i] for i in ind_limit ]
		mean_rms = np.mean(rms)
		
		# host galaxy
		freq_host 	  = freq_em/(1.+z)
		freq_host_err = (z_err/z)*freq_host
	
		SdV = 5.*mean_rms*np.sqrt(10 * 100)				# 5*sigma flux estimate on 10 km/s binned spectra in mJy km/s (assuming FWHM=100 km/s)
	
		print('SdV = %f mJy km/s' %SdV)	
		print('Frequency of host galaxy (from systemic z) = %.2f +/- %.2f GHz' 
			%(freq_host, freq_host_err))
		
		M_H2 = self.get_mass(z, z_err, SdV, 0., freq_o, freq_o_err, 0)	# H_2 mass in solar masses

		try:
			print('SFR = %.0f + %.0f - %.0f' % (s[0], s[1], s[2]))
			print('tau_(depl.) = %.0f Myr' % ((M_H2/s[0])/1.e6))
	
		except: 
			print("No SFR Measured")
	
	def CI_contours( self, CI_path, CI_moment0, CI_rms ):
		"""
		Reads header and image data and
		generates [CI] contours from moment-0 map
	
		Parameters 
		----------
		CI_path : Path of ALMA [CI] datacubes
	
		CI_moment0 : [CI] moment-0 map 
	
		CI_rms : Minimum threshold value of [CI] contours
	
		Return
		------
		[CI] image array, WCS and contours : 1d array
	
		"""
		moment0 = fits.open(CI_path+CI_moment0)
		CI_wcs	= WCS(moment0[0].header)
	
		# select RA,DEC axes only 
		CI_new_wcs = CI_wcs.sub(axes = 2)	
		CI_img_arr = moment0[0].data[0,0,:,:]
		
		#define contour parameters	
		n_contours 		=	4
		n 				=   1
		
		contours 		= np.zeros(n_contours)
		contours[0] 	= CI_rms
		
		for i in range(1,n_contours):
			contours[i] = CI_rms*np.sqrt(2)*n
			n			+= 1
	
		return [ CI_img_arr, CI_new_wcs, contours ]
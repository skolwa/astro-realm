"""
S.N. Kolwa
ESO (2019) 

"""

import numpy as np
from math import*

import matplotlib.pyplot as pl
from astropy.io import fits
import mpdaf.obj as mpdo
import scipy.ndimage as ndimage
from astropy.wcs import WCS

import time

from Image_CI import *


class Multiwavelength_Image:

	def __init__( self, output_dir=None, plot_dir=None ):
		"""
		output_dir : Directory for output files

		plot_dir : Directory for output plots

		"""
		self.output_dir = output_dir
		self.plot_dir = plot_dir

	def VLA_contours( self, vla_path, vla_img, vla_rms ):
		"""
		Reads header and image data and
		generates VLA contours from VLA image

		Parameters 
		----------
		vla_path : Path of ALMA [CI] datacubes

		vla_img : Filename of VLA image

		vla_rms : Minimum threshold value of VLA contours

		Return
		------
		VLA image array, WCS and contours : 1D array

		"""
		vla_hdu 	= fits.open(vla_path+vla_img)[0]
		vla_hdr 	= vla_hdu.header
		vla_data 	= vla_hdu.data[0,0,:,:]
		
		w = WCS(vla_hdr)

		vla_wcs = w.celestial
		
		#VLA contour levels
		n_contours = 6		
		contours   = [ vla_rms ]
		
		for i in range(n_contours):
			contours.append( contours[i]*(np.sqrt(2))*2 )

		return [vla_data, vla_wcs, contours]
	
	def muse_lya_contours( self, muse_rms, Lya_img ):
		"""
		Generate MUSE Ly-alpha contours
	
		Parameters 
		----------
		muse_rms : Minimum threshold value of MUSE Ly-alpha contours	
	
		Lya_img : Filename for Ly-alpha image
	
		Return
		------
		MUSE Ly-alpha contours : 1D array
	
		"""
		muse_hdu = fits.open(Lya_img)
		muse_hdr = muse_hdu[1].header
		muse_wcs = WCS(muse_hdr).celestial
		muse_data = muse_hdu[1].data[:,:]
	
		#MUSE Lya contour levels
		n_contours = 6	
		n = 1	
			
		contours    = np.zeros(n_contours)
		contours[0] = muse_rms
		
		for i in range(1, n_contours):
			contours[i] = muse_rms*np.sqrt(2)*n
			n += 2
	
		return [muse_data, muse_wcs, contours]
	
	def irac_contours( self, irac_path, irac_img, irac_rms, n_contours ):
		"""
		Reads header and image data and
		generates IRAC contours from IRAC image

		Parameters 
		----------
		irac_path : Path of IRAC path

		irac_img : Filename of IRAC image

		irac_rms : Minimum threshold value of IRAC contours

		n_contours: Number of contours

		Return
		------
		VLA image array, WCS and contours : 1D array

		"""
		irac_hdu = fits.open(irac_path+irac_img)
		irac_hdr = irac_hdu[0].header
		irac_wcs = WCS(irac_hdr).celestial

		irac_data = irac_hdu[0].data[:,:]
			
		#IRAC contour levels 
		n = 1		
			
		contours    = np.zeros(n_contours)
		contours[0] = irac_rms
		
		for i in range(1, n_contours):
			contours[i] = contours[0]*np.sqrt(2)*n
			n += 1

		return [irac_data, irac_wcs, contours]
	
	def hst_vla_CI( self, vla_path, vla_img, vla_rms, CI_path, CI_moment0, CI_rms,
		hst_path, hst_img, source, dl ):
		"""
		Visualise HST narrow-band data with ALMA [CI] and 
		VLA radio contours

		Parameters 
		----------
		vla_path : Path of VLA image

		vla_img : VLA  image name

		vla_rms : Minimum threshold value of VLA contours
		
		CI_path : Path of [CI] moment-0 map

		CI_moment0 : Filename of moment-0 map

		CI_rms : Minimum threshold value of [CI] contours

		hst_path : Path of HST data

		hst_img : Filename of HST image

		source : Name of source

		dl : Length of distance scale bar

		Return
		------
		Multiwavelength narrow-band image of HST observation overlaid
		with ALMA [CI] and VLA radio contours 

		"""
		fig = pl.figure(figsize=(5.5,5))
		
		[vla_data, vla_wcs, vla_contours] = self.VLA_contours(vla_path, vla_img, vla_rms)

		[CI_data, CI_wcs, contours] = Image_CI.CI_contours(self, CI_path, CI_moment0, CI_rms)

		ax = fig.add_axes([0.2, 0.12, 0.8, 0.8], projection=CI_wcs)

		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize('16')

		ax.set_xlabel(r'$\alpha$ (J2000)', fontsize=12)
		ax.set_ylabel(r'$\delta$ (J2000)', fontsize=12)

		ax.contour(vla_data, levels=vla_contours, colors='green',
		 transform=ax.get_transform(vla_wcs))

		ax.contour(CI_data, levels=contours, colors='blue', label='[CI](1-0)')

		ax.set_xlim(10, 40)
		ax.set_ylim(10, 40)

		hdu = fits.open(hst_path+hst_img)
		hst_hdr  = hdu[1].header
		hst_data = hdu[1].data[:,:]

		hst_wcs = WCS(hst_hdr)

		x = len(hst_data)
		photflam = hdu[1].header['photflam']  #erg/s/cm^2/Ang

		hst_arr = [ [hst_data[i][j]*photflam*1.e22 for j in range(x)] for i in range(x) ]
		hst_fig = ax.imshow(hst_arr, transform=ax.get_transform(hst_wcs), origin='lower', 
			interpolation='nearest', vmin=-5., vmax=15, cmap='gist_gray_r')

		# cbaxes = fig.add_axes([0.81, 0.11, 0.02, 0.85])
		# cb = pl.colorbar(hst_fig, orientation = 'vertical', cax=cbaxes)
		# cb.set_label(r'S.B. (10$^{-22}$ erg s$^{-1}$ cm$^{-2}$)',rotation=90, fontsize=10)

		ra = ax.coords[0]
		ra.set_major_formatter('hh:mm:ss.s')
		
		l = 12

		if source == 'MRC0943':
			ax.scatter(25, 25, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax.text(24.25, 22.5, 'Host', c='red', fontsize=12, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)
 
			ax.scatter(21.63, 34.97, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax.text(20.5, 32.47, 'NE', c='red', fontsize=12, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)

			ax.scatter(33, 20, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax.text(32., 17.5, 'Thor', c='red', fontsize=12, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)

			ax.scatter(37.46, 16.30, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax.text(35.5, 13.80, 'Loke', c='red', fontsize=12, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)

		# ax.text(l, l+dl, '10 kpc', color='red', fontsize=5, 
		# 	bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
		# ax.plot([l, l+dl], [l+1, l+1], c='red', lw='2', zorder=10.)

		return pl.savefig(self.plot_dir+source+'_hst_CI_VLA_img.png', dpi=300)
	
	def irac_vla_CI( self, vla_path, vla_img, vla_rms, CI_path, CI_moment0, CI_rms,
		irac_path, irac_img, source, l, dl, irac_vmin, N ):
		"""
		
		Visualise IRAC narrow-band image with ALMA [CI] and 
		VLA radio contours

		Parameters 
		----------
		vla_path : Path of VLA image

		vla_img : Filename of VLA image

		vla_rms : Minimum threshold of VLA contours

		CI_path : Path of [CI] moment-0 map

		CI_moment0 : Filename of moment-0 map

		CI_rms : Minimum threshold of [CI] contours
		
		irac_path : Path of IRAC image
		
		irac_img : IRAC image name

		source : Name of source

		dl : Length of distance scale bar  

		Return
		------
		Multiwavelength narrow-band image with HST data overlaid
		with ALMA-detected [CI] and VLA radio contours 

		"""

		[CI_data, CI_wcs, contours] = Image_CI.CI_contours(self, CI_path, CI_moment0, CI_rms)

		fig = pl.figure(figsize=(5.5,5))
		
		ax = fig.add_axes([0.2, 0.12, 0.8, 0.8], projection=CI_wcs)

		ax.contour(CI_data, levels=contours, colors='blue', 
		 label='[CI](1-0)')

		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize('16')

		ax.set_xlabel(r'$\alpha$ (J2000)', fontsize=12)
		ax.set_ylabel(r'$\delta$ (J2000)', fontsize=12)

		[vla_data, vla_wcs, vla_contours] = self.VLA_contours(vla_path, vla_img, vla_rms)

		ax.contour(vla_data, levels=vla_contours, colors='green',
			transform=ax.get_transform(vla_wcs))

		ax.set_xlim(10, 40)
		ax.set_ylim(10, 40)

		hdu = fits.open(irac_path+irac_img)[0] 
		irac_hdr  = hdu.header
		irac_data = hdu.data[:,:]

		irac_wcs = WCS(irac_hdr)

		pix = list(chain(*irac_data))
		pix = [ x for x in pix if str(x) != 'nan' ]
		pix_rms = np.sqrt(np.mean(np.square(pix)))
		pix_med = np.median(pix)
		vmax = (pix_med + N*pix_rms) 
		vmin = irac_vmin

		irac_fig = ax.imshow(irac_data, transform=ax.get_transform(irac_wcs), origin='lower', 
			interpolation='nearest', vmin=vmin, vmax=vmax, cmap='gist_gray_r')

		# cbaxes = fig.add_axes([0.81, 0.11, 0.02, 0.85])
		# cb = pl.colorbar(irac_fig, orientation = 'vertical', cax=cbaxes)
		# cb.set_label(r'S.B. (MJy sr$^{-1}$)',rotation=90, fontsize=10)

		ra = ax.coords[0]
		ra.set_major_formatter('hh:mm:ss.s')

		# ax.text(l, l, '10 kpc', color='red', fontsize=8, 
		# 	bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5, ha='center')

		# m = l - 1.0
		# n = l - 0.5
		# ax.plot([m, m+dl], [n, n], c='red', lw='2', zorder=10.)

		return pl.savefig(self.plot_dir+source+'_irac_CI_VLA_img.png', dpi=300)
	
	def muse_lya_irac_CI( self, Lya_subcube, Lya_img, muse_lya_rms, lam1, lam2,
		irac_path, irac_img, irac_rms, CI_path, CI_moment0, CI_rms, 
		radio_hotspots, dl, img_dim, source ):
		"""
		Overlay [CI] contours over Spitzer/IRAC imaging

		Parameters 
		----------
		Lya_subcube : Continuum-subtracted Ly-alpha subcube

		Lya_img : Shorthand name of Ly-alpha subcube

		muse_lya_rms : Minimum threshold of Ly-alpha contours

		lam1 : Lower limit of narrow-band of Ly-alpha image

		lam2 : Upper limit of narrow-band of Ly-alpha image

		irac_path : Path of IRAC image

		irac_img : Filename of IRAC image

		irac_rms : Minimum threshold of IRAC rms

		CI_path : Path of ALMA [CI] datacubes

		CI_moment0 : Filename of [CI] moment-0 map

		CI_rms : Minimum threshold of [CI] contours

		radio_hotspots : Pixel co-ordinates of radio hotspots in MUSE image

		dl : Length of distance scale bar  

		Returns 
		-------
		Ly-alpha narrow-band image with [CI] and Spitzer/IRAC contours overlaid 

		"""
		[CI_data, CI_wcs, contours] = Image_CI.CI_contours(self, CI_path, CI_moment0, CI_rms)

		fig = pl.figure(figsize=(5.5,4.5))
		
		ax1 = fig.add_axes([0.15, 0.12, 0.78, 0.8], projection=CI_wcs)

		ax1.contour(CI_data, levels=contours, colors='blue', label='ALMA [CI]', zorder=5)

		muse_Lya = mpdo.Cube(self.output_dir+Lya_subcube, ext=1)

		# collapse along Lya profile axis to form image
		spec = muse_Lya.sum(axis=(1,2))
		p1,p2 = spec.wave.pixel([lam1,lam2], nearest=True)

		muse_img = muse_Lya[p1:p2+1, :, :].sum(axis=0)
		# print(muse_img.info())
		muse_img_arr = muse_img.data[:,:]
		x = len(muse_img_arr)
		y = len(muse_img_arr[0])
		delta_lam = abs(lam2-lam1)		#bandwidth of image

		muse_img_arr = [ [muse_img_arr[i][j]/delta_lam/0.04e2 for j in range(y)] for i in range(x) ]

		muse_img.write(self.output_dir+Lya_img+'_'+str(int(lam1))+'_'+str(int(lam2))+'.fits')		#save to create new WCS		

		muse_hdu = fits.open(self.output_dir+Lya_img+'_'+str(int(lam1))+'_'+str(int(lam2))+'.fits')	#open saved narrow-band image

		muse_hdr = muse_hdu[1].header
		muse_wcs = WCS(muse_hdr).celestial

		pix = list(chain(*muse_img_arr))
		pix_rms = np.sqrt(np.mean(np.square(pix)))
		pix_med = np.median(pix)

		vmax = 5*(pix_med + pix_rms) 
		vmin = 0.2*(pix_med - pix_rms) 

		muse_smoothed = ndimage.gaussian_filter(muse_img_arr, sigma=(1, 1), order=0)
		muse_fig = ax1.imshow(muse_smoothed, transform=ax1.get_transform(muse_wcs), origin='lower', 
			interpolation='nearest', 
			cmap='gist_gray_r', vmin=vmin, vmax=vmax)

		lya_fn = self.muse_lya_contours( muse_lya_rms, self.output_dir+Lya_img+'_'+str(int(lam1))+'_'+str(int(lam2))+'.fits' )
		lya_data, lya_wcs, contours = lya_fn[0], lya_fn[1], lya_fn[2]
		# print(contours)

		ax1.contour( lya_data, levels=contours, colors='grey', alpha=0.4,
		transform=ax1.get_transform(muse_wcs), label=r'MUSE Ly$\alpha$' ) 

		for (h1,h2) in radio_hotspots:
			ax1.scatter(h1, h2, marker='X', transform=ax1.get_transform(muse_wcs), facecolor='green', 
				s=100, zorder=10, edgecolor='black')

		cbaxes = fig.add_axes([0.865, 0.12, 0.02, 0.8])
		cb = pl.colorbar(muse_fig, orientation = 'vertical', cax=cbaxes)
		# cb.set_label(r' x 10$^{-18}$ erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$',rotation=90, fontsize=8)

		[irac_data, irac_wcs, contours] = self.irac_contours( irac_path, irac_img, irac_rms, 6 )

		ax1.contour( irac_data, levels=contours, colors='red', alpha=0.5,
		transform=ax1.get_transform(irac_wcs), label='IRAC 2' )

		# l1,l2 are (x,y) of [CI]1-0 image
		l1,l2 = img_dim
		ax1.set_xlim(l1,l2)
		ax1.set_ylim(l1,l2)

		# distance scale bar
		l = l1 + 2.5

		# ax1.text( l, l,'10 kpc', color='red', fontsize=10, 
		# 	bbox={'facecolor':'white', 'alpha':0.7, 'pad':15}, zorder=5, 
		# 	transform=ax1.get_transform(CI_wcs), ha='center')

		# m = l - 0.75
		# n = l - 0.5
		# ax1.plot( [m, m+dl], [n, n], c='red', lw='2', zorder=10.)

			# add region labels for MRC 0943-242

		fs = 10
		if source == 'MRC0943':
			ax1.scatter(25, 25, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax1.text(23.75, 22.5, 'Host', c='red', fontsize=fs, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)
 
			ax1.scatter(21.63, 34.97, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax1.text(20.25, 32.47, 'A. NE', c='red', fontsize=fs, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)

			ax1.scatter(33, 20, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax1.text(31., 17.5, 'B. Thor', c='red', fontsize=fs, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)

			ax1.scatter(37.46, 16.30, marker='X', facecolor='red', s=100, zorder=5, edgecolor='k')
			ax1.text(34.5, 13.80, 'C. Loke', c='red', fontsize=fs, weight='bold', 
				backgroundcolor='white',
				bbox={'facecolor':'white', 'alpha':0.8, 'pad':4}, zorder=5)

		ax1.set_xlabel(r'$\alpha$ (J2000)', fontsize=12)
		ax1.set_ylabel(r'$\delta$ (J2000)', fontsize=12)

		ra = ax1.coords[0]
		ra.set_major_formatter('hh:mm:ss.s')

		return pl.savefig(self.plot_dir+irac_img[:-5]+'_CI.png', dpi=300)
	
	def muse_lya_continuum_subtract( self, muse_cube, lam1, lam2, mask_lmin, 
		mask_lmax, ra_lim, dec_lim, source ): 
		"""
		Continuum-subtract MUSE Lya subcube

		Parameters 
		----------
		muse_cube : File path for MUSE datacube 

		lam1 : Lower wavelength limit of subcube

		lam2 : Upper wavelength limit of sucbube 

		mask_lmin : Lower wavelength limit of spectral region mask

		mask_lmax : Upper wavelength limit of spectral region mask

		ra_lim : Right ascension pixel of subcube 

		dec_lim : Declination pixel range of subcube
		
		source : Name of source

		Returns 
		-------
		Continuum subtracted Ly-alpha subcube

		"""
		# Open MUSE Lya image
		start_time = time.time()

		muse_hdu = fits.open(muse_cube)

		muse_mpdaf_cube = mpdo.Cube(muse_cube, mmap=True)
		# select F.O.V.
		rg = muse_mpdaf_cube[:, dec_lim[0]:dec_lim[1], ra_lim[0]:ra_lim[1]]	

		# identify Ly-alpha line in spectrum
		spec = rg.sum(axis=(1,2))

		# select spectral range and smaller F.O.V. (in pixels)
		p1,p2 = spec.wave.pixel([lam1,lam2], nearest=True)

		Lya_emi = rg[ p1:p2+1, :, : ]				# select spectral region

		Lya_emi.write(self.output_dir+source+'_Lya_subcube.fits')

		print( 'Initialising empty cube on which to write continuum solution...' )

		#empty cube	
		cont 			= Lya_emi.clone(data_init = np.empty, var_init = np.empty) 
		#copy of cube	
		Lya_emi_copy 	= Lya_emi.copy()					

		print( 'Masking copy of sub-cube...' )

		for sp in mpdo.iter_spe(Lya_emi_copy):#mask Lya line emission in each spaxel
			sp.mask_region(lmin=mask_lmin,lmax=mask_lmax)

		print( 'Calculating continuum solution...' )

		for sp,co in zip(mpdo.iter_spe(Lya_emi_copy),mpdo.iter_spe(cont)):
			co[:] = sp.poly_spec(0)

		print( 'Subtracting continuum...' )

		# continuum subtracted cube
		Lya_cs = Lya_emi - cont	
		Lya_cs.write(self.output_dir+source+'_Lya_subcube_cs.fits')

		elapsed = (time.time() - start_time)/60.
		print( "Process complete. Total build time: %f mins" % elapsed )
	
	def muse_lya_astro_correct( self, muse_std, gaia_std, Lya_subcube ):
		"""
		MUSE astrometry correct

		Parameters 
		----------
		muse_std : ra, dec of field star in MUSE frame

		gaia_std : ra, dec of field star in GAIA frame

		Lya_subcube : Name of Ly-alpha subcube

		Returns 
		-------
		Astrometry-corrected Ly-alpha subcube

		"""
		hdu = fits.open(self.output_dir+Lya_subcube)
		hdr = hdu[1].header

		ra_offset = muse_std[0] - gaia_std[0]
		dec_offset = muse_std[1] - gaia_std[1]

		new_ra = hdr['crval1'] - ra_offset
		new_dec = hdr['crval2'] - dec_offset

		hdr.update(crval1=new_ra, crval2=new_dec)

		hdu.writeto(self.output_dir+Lya_subcube[:-5]+'_astrocorr.fits', 
			overwrite=1)
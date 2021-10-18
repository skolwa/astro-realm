"""
S.N. Kolwa
ESO (2019) 

"""

from astropy.io import fits
import numpy as np

from matplotlib import rcParams
import matplotlib.pyplot as pl
from matplotlib.cbook import get_sample_data
from matplotlib.ticker import FormatStrFormatter

import lmfit
import lmfit.models as lm

from Gaussian import * 
from Image_CI import *

import pickle

from itertools import chain

class Spectrum_CI:

	def __init__( self, input_dir=None, output_dir=None, plot_dir=None ):
		"""
		input_dir : Directory of input files

		output_dir : Directory of output files

		plot_dir : Directory of plot files (for manuscript)

		c : Speed of light (constant)

		"""

		self.input_dir = input_dir	
		self.output_dir = output_dir
		self.plot_dir = plot_dir

		self.c = 2.9979245800e5 	# in km/s

	def freq_to_vel( self, freq_obs, freq_em, z ):
		"""
		Convert an observed frequency to a velocity 
	
		Parameters 
		----------
		freq_obs : Observed frequency
	
		freq_em : Emitted frequency
	
		z : Observer's redshift
	
		Returns 
		-------
		Velocity (km/s)
		
		"""
		v = self.c*(1. - freq_obs/freq_em/(1.+z) )
		return v
	
	def convert_fwhm_kms( self, sigma, sigma_err, 
		freq_o, freq_o_err ):	
		"""
		Convert velocity dispersion (km/s) to FWHM (km/s)

		Parameters 
		----------
		sigma : Velocity dispersion 

		sigma_err : Velocity dispersion error

		freq_o : Observed frequency

		freq_o_err : Observed frequency error

		Returns 
		-------
		FWHM and FWHM error (km/s) : 1D array
		
		"""
		freq_em = 492.161		#rest frequency of [CI](1-0) in GHz

		fwhm 		= 2.*np.sqrt(2*np.log(2))*sigma 
		fwhm_err 	= (sigma_err/sigma)*fwhm
		
		fwhm_kms	= (fwhm/freq_em)*self.c
		fwhm_kms_err = fwhm_kms*(fwhm_err/fwhm) 
		
		return (fwhm_kms, fwhm_kms_err)
	
	def make_spectrum( self, CI_path, CI_moment0, 
		CI_region, source, p, z, z_err, freq_obs_mt0):
		"""
		Show [CI] line spectra and line-fits
	
		Parameters 
		----------
		CI_path : Path of [CI] spectrum 
	
		CI_moment0 : Filename of [CI] moment-0 map

		CI_region : [CI] regions (array)

		source : Short-hand source name

		p : Initial parameters for line-fit

		z : Redshift

		z_err : Redshift error

		freq_obs_mto : Observed frequency range of [CI] moment-0 map

		npb_rms : RMS in non-primary-beam-corrected image
		
		"""
		self.CI_path = CI_path
		self.CI_moment0 = CI_moment0
		self.CI_region = CI_region
		self.source = source
		self.p = p
		self.z = z
		self.z_err = z_err
		self.freq_obs_mt0 = freq_obs_mt0

		freq_em = 492.161		#rest frequency of [CI](1-0) in GHz

		moment0 = fits.open(self.CI_path+CI_moment0)
		hdr =  moment0[0].header
	
		bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees

		print('{:.2f} x {:.2f} arcsec^2'.format(bmaj*3600, bmin*3600))

		k = -1 
		for spec in CI_region:
			k += 1

			# 10 km/s binned 1D spectrum
			alma_spec = np.genfromtxt(self.input_dir+source+'_spec_'+spec+'_freq.txt')
	
			freq = alma_spec[:,0]	# GHz		
			flux = [ alma_spec[:,1][i]*1.e3 for i in range(len(freq)) ] # mJy/beam
			rms  = [ alma_spec[:,2][i]*1.e3 for i in range(len(freq)) ] # mJy/beam
			inv_noise = [ rms[i]**-1 for i in range(len(freq)) ]

			# Draw spectrum
			fig = pl.figure(figsize=(7,5))
			fig.add_axes([0.15, 0.1, 0.8, 0.8])
			ax = pl.gca()

			print( '-'*len('   '+spec+' '+source+'   '))
			print('   '+spec+' '+source+'   ')
			print( '-'*len('   '+spec+' '+source+'   '))

			# For [CI] detections
			if source=='4C03' or (source=='MRC0943' and spec=='SW'): 
				# Single component fit

				pars = Parameters()

				pars.add_many( 
					('g_cen', p[k][1], True, 0.),
					('a', p[k][2], True, 0.),	
					('wid', p[k][3], True, 0.),  	
					('cont', p[k][4], True, 0., 0.001 )
					)
				
				mod 	= lm.Model(Gaussian.gauss) 
				fit 	= mod.fit(flux, pars, x=freq, weights=inv_noise)

				print( fit.fit_report() )
				print( fit.ci_report(sigmas=[2]))
				conf_int = fit.conf_interval(sigmas=[2])
				
				res = fit.params

				# Line-width in km/s
				freq_o, freq_o_err = res['g_cen'].value, res['g_cen'].stderr
				# freq_o_err_lo, freq_o_err_hi = freq_o - conf_int['g_cen'][0][1], conf_int['g_cen'][2][1] - freq_o

				sigma = res['wid'].value
				sigma_err = res['wid'].stderr
				# sigma_err_lo, sigma_err_hi	= sigma - conf_int['wid'][0][1], conf_int['wid'][2][1] - sigma
				
				sigma_kms = (sigma/freq_em)*self.c
				sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )

				# sigma_kms_err_lo = sigma_kms*np.sqrt ( (sigma_err_lo/sigma)**2 + (freq_o_err_lo/freq_o)**2 )
				# sigma_kms_err_hi = sigma_kms*np.sqrt ( (sigma_err_hi/sigma)**2 + (freq_o_err_hi/freq_o)**2 )
	
				# FWHM in km/s
				fwhm_kms, fwhm_kms_err = self.convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

				# fwhm_kms, fwhm_kms_err_lo = self.convert_fwhm_kms( sigma, sigma_err_lo, freq_o, freq_o_err_lo )
				# fwhm_kms, fwhm_kms_err_hi = self.convert_fwhm_kms( sigma, sigma_err_hi, freq_o, freq_o_err_hi )
				
				print( "Line centre (GHz) = %.3f +/- {+%.3f}"%(freq_o, freq_o_err))
				print( "Sigma (km/s) = %.3f +/- {+%.3f}" %(sigma_kms, sigma_kms_err))
				print( "FWHM (km/s) = %.0f +/- {+%.0f}" %(fwhm_kms, fwhm_kms_err))
	
				flux_peak, flux_peak_err = res['a'].value, res['a'].stderr
	
				# Integrated Flux in mJy km/s
				SdV 		= flux_peak * fwhm_kms 				
				SdV_err 	= SdV * (flux_peak_err/flux_peak)
	
				print( "Flux peak (mJy) = %.3f +/- %.3f" %(flux_peak, flux_peak_err) )
				print( "Flux integrated (mJy km/s) = %.0f +/- %.0f" %(SdV, SdV_err) )
	
				# Inferred H_2 mass
				M_H2 = Image_CI.get_mass(self, z, z_err, SdV, SdV_err, freq_o, freq_o_err, 1)	
	
				# Velocity shifts
				v_sys = self.c*z / (1.+z)
				v_obs = self.c*(1. - freq_o/freq_em)
	
				vel_offset = v_obs - v_sys
				vel_offset_err = vel_offset * (freq_o_err/freq_o)
				
				print( "Velocity shift (km/s) = %.3f +/- %.3f" %( vel_offset, vel_offset_err ) )
	
				# Frequency range of moment-0 map
				v_obs_mt0 = ( self.c*(1. - freq_obs_mt0[0]/freq_em) - v_sys, 
					self.c*(1. - freq_obs_mt0[1]/freq_em) - v_sys )
	
				print('Moment-0 map velocity range: %.2f to %.2f km/s' %(v_obs_mt0[1], v_obs_mt0[0]) )
	
				freq_ax = np.linspace( freq.min(), freq.max(), num=len(freq))

				ax.plot(freq_ax, Gaussian.gauss(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

				fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

				np.savetxt(self.output_dir+spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
					header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

			# No [CI] line detection
			else:
				print("[CI] line not detected")
			
			ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
			ax.plot( freq, rms, c='grey', alpha=1.0, drawstyle='steps-mid')

			if (source=='MRC0943' and (spec=='Host' or spec=='NE')):
				alma_spec_50 = np.genfromtxt(self.input_dir+source+'_spec_Host_50kms_freq.txt')

				freq_50 = alma_spec_50[:,0]	
				flux_50 = [ alma_spec_50[:,1][i]*1.e3 for i in range(len(freq_50)) ]
				ax.plot( freq_50, flux_50, c='#0a78d1', drawstyle='steps-mid' )

			elif (source=='MRC0943' and spec=='Loke'): 
				alma_spec_Loke_50 = np.genfromtxt(self.input_dir+source+'_spec_Loke_50kms_freq.txt')

				freq_Loke = alma_spec_Loke_50[:,0]	
				flux_Loke = [ alma_spec_Loke_50[:,1][i]*1.e3 for i in range(len(freq_Loke)) ]
				ax.plot( freq_Loke, flux_Loke, c='#0a78d1', drawstyle='steps-mid' )

			elif source=='4C03':
				pbcor_spec = np.genfromtxt(self.input_dir+source+'_spec_Host_freq_pbcor.txt')

				freq_pb = pbcor_spec[:,0]
				flux_pb = [ pbcor_spec[:,1][i]*1.e3 for i in range(len(freq_pb)) ]
				ax.plot( freq_pb, flux_pb, c='#d2e5f7', drawstyle='steps-mid' )

			else: 
				alma_spec_50 = np.genfromtxt(self.input_dir+source+'_spec_Host_50kms_freq.txt')

				freq_50 = alma_spec_50[:,0]	
				flux_50 = [ alma_spec_50[:,1][i]*1.e3 for i in range(len(freq_50)) ]
				ax.plot( freq_50, flux_50, c='#0a78d1', drawstyle='steps-mid' )


			pl.savefig(self.output_dir+'CI_spec_'+spec+'.png')
		
			# Pickle figure and save to re-open again
			pickle.dump( fig, open( self.output_dir+'CI_spec_'+spec+'.pickle', 'wb' ) )


		# Get the axes to put the data into subplots
		data1, data2, data3, data4 = [], [], [], []

		for spec in CI_region:
			mpl_ax = pickle.load(open(self.output_dir+'CI_spec_'+spec+'.pickle', 'rb'))

			if source=='4C03':

				data1.append(mpl_ax.axes[0].lines[0].get_data())	
				data2.append(mpl_ax.axes[0].lines[1].get_data())	
				data3.append(mpl_ax.axes[0].lines[2].get_data())
				data4.append(mpl_ax.axes[0].lines[3].get_data())

			else: 
				data1.append(mpl_ax.axes[0].lines[0].get_data())	
				data2.append(mpl_ax.axes[0].lines[1].get_data())	
				data3.append(mpl_ax.axes[0].lines[2].get_data())

		# Frequency at the systemic redshift (from HeII)
		freq_sys = freq_em/(1.+z)				
		vel0 = self.freq_to_vel(freq_sys, freq_em, 0.)

		# Velocity axes 
		if source=='4C03' :
			v_radio1 = [ self.freq_to_vel(data1[0][0][i], freq_em, 0.) for i in range(len(data1[0][0])) ] #model
			v_radio2 = [ self.freq_to_vel(data2[0][0][i], freq_em, 0.) for i in range(len(data2[0][0])) ] #data 


			voff1 = [ v_radio1[i] - vel0 for i in range(len(v_radio1)) ]
			voff2 = [ v_radio2[i] - vel0 for i in range(len(v_radio2)) ]

		elif source=='MRC0943':
			# plot 1
			v_radio11 = [ self.freq_to_vel(data1[0][0][i], freq_em, 0.) for i in range(len(data1[0][0])) ] #flux data
			v_radio12 = [ self.freq_to_vel(data2[0][0][i], freq_em, 0.) for i in range(len(data2[0][0])) ] #rms data
			v_radio13 = [ self.freq_to_vel(data3[0][0][i], freq_em, 0.) for i in range(len(data3[0][0])) ] #wide binning


			voff11 = [ v_radio11[i] - vel0 for i in range(len(v_radio11)) ]
			voff12 = [ v_radio12[i] - vel0 for i in range(len(v_radio12)) ]	
			voff13 = [ v_radio13[i] - vel0 for i in range(len(v_radio13)) ]

			# plot 2	
			v_radio21 = [ self.freq_to_vel(data1[1][0][i], freq_em, 0.) for i in range(len(data1[1][0])) ] #model 
			v_radio22 = [ self.freq_to_vel(data2[1][0][i], freq_em, 0.) for i in range(len(data2[1][0])) ] #flux data 
			v_radio23 = [ self.freq_to_vel(data3[1][0][i], freq_em, 0.) for i in range(len(data3[1][0])) ] #rms data 


			voff21 = [ v_radio21[i] - vel0 for i in range(len(v_radio21)) ]
			voff22 = [ v_radio22[i] - vel0 for i in range(len(v_radio22)) ]	
			voff23 = [ v_radio23[i] - vel0 for i in range(len(v_radio23)) ]

			#plot 3	
			v_radio31 = [ self.freq_to_vel(data1[2][0][i], freq_em, 0.) for i in range(len(data1[2][0])) ] #flux data
			v_radio32 = [ self.freq_to_vel(data2[2][0][i], freq_em, 0.) for i in range(len(data2[2][0])) ] #rms data
			v_radio33 = [ self.freq_to_vel(data3[2][0][i], freq_em, 0.) for i in range(len(data3[2][0])) ] #wide binning

			voff31 = [ v_radio31[i] - vel0 for i in range(len(v_radio31)) ]
			voff32 = [ v_radio32[i] - vel0 for i in range(len(v_radio32)) ]	
			voff33 = [ v_radio33[i] - vel0 for i in range(len(v_radio33)) ]

			#plot 4
			v_radio41 = [ self.freq_to_vel(data1[3][0][i], freq_em, 0.) for i in range(len(data1[3][0])) ] #flux data
			v_radio42 = [ self.freq_to_vel(data2[3][0][i], freq_em, 0.) for i in range(len(data2[3][0])) ] #rms data
			v_radio43 = [ self.freq_to_vel(data3[3][0][i], freq_em, 0.) for i in range(len(data3[3][0])) ] #wide binning

			voff41 = [ v_radio41[i] - vel0 for i in range(len(v_radio41)) ]
			voff42 = [ v_radio42[i] - vel0 for i in range(len(v_radio42)) ]	
			voff43 = [ v_radio43[i] - vel0 for i in range(len(v_radio43)) ]

		else:
			v_radio1 = [ self.freq_to_vel(data1[0][0][i], freq_em, 0.) for i in range(len(data1[0][0])) ] #model
			v_radio2 = [ self.freq_to_vel(data2[0][0][i], freq_em, 0.) for i in range(len(data2[0][0])) ] #data 
			v_radio3 = [ self.freq_to_vel(data3[0][0][i], freq_em, 0.) for i in range(len(data3[0][0])) ] #data 


			voff1 = [ v_radio1[i] - vel0 for i in range(len(v_radio1)) ]
			voff2 = [ v_radio2[i] - vel0 for i in range(len(v_radio2)) ]	
			voff_wide = [ v_radio3[i] - vel0 for i in range(len(v_radio3)) ]	

		# round up to nearest 0.5
		def roundup(x):
			return int(ceil(x/10.0))*0.5

		# Global plot parameters
		fs = 12
		x_pos = 0.05
		y_pos = 0.92

		rcParams['font.sans-serif'] = ['Helvetica']


		# Custom plot parameters per source
		if source == '4C03':
			fig, ax = pl.subplots(2, 1, figsize=(4, 3), sharex=True, 
			constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})

			pl.subplots_adjust(hspace=0, wspace=0.01) 
			host_cont = np.genfromtxt(self.output_dir+'Host_fit_params.txt')[6]
			# ax[0].plot(voff2, data4[0][1], c='#d2e5f7', drawstyle='steps-mid', alpha=0.8, lw=0.8) 
			ax[0].plot(voff2, data2[0][1], c='k', drawstyle='steps-mid', alpha=0.9, lw=0.8)
			ax[0].plot(voff1, data1[0][1], c='red', alpha=0.7, lw=0.8)
			indices = [ i for i in range(len(voff2)) if (voff2[i] > 220. and voff2[i] < 320.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data1_2_sub = [ data2[0][1][i] for i in indices ]
			ax[0].fill_between( voff2_sub, data1_2_sub, host_cont, where=data1_2_sub > host_cont , 
				interpolate=1, color='yellow', alpha=0.5 )
			ax[0].text(x_pos, y_pos, '4C+03.24 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')		

			miny,maxy,dy = -1.0, 1.6, 0.5
			ax[0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
			ax[0].plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5)
			ax[0].set_yticks( np.arange(miny, maxy, dy) )
			ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
			ax[0].set_ylim([miny+0.1, maxy])
			ax[0].set_xlim([-400,400])
			
			miny,maxy,dy = 0.0, 0.7, 0.3
			ax[1].set_yticks( np.arange(miny, maxy, dy) )
			ax[1].plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5)
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[1].set_ylabel(r'$\sigma_{\nu}$', fontsize=fs )
			ax[1].plot(voff2, data3[0][1], c='grey', alpha=0.5, drawstyle='steps-mid', lw=0.8)

			for ax in ax: 
				ax.tick_params(direction='in', length=4, right=1, top=1)

				for pos in ['top','left','bottom', 'right']:
					ax.spines[pos].set_linewidth(1.2)
					ax.spines[pos].set_color('k')

			pl.savefig(self.plot_dir+'4C03_CI_spectrums.png', bbox_inches = 'tight',
    		pad_inches = 0.1)

		elif source == 'MRC0943':
			fig, ax = pl.subplots(2, 1, figsize=(4, 3), sharex=True, 
			constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})

			pl.subplots_adjust(hspace=0, wspace=0.01) 
			ax[0].plot(voff11, data1[0][1], c='k', drawstyle='steps-mid', alpha=1.0, lw=0.8)
			ax[0].plot(voff13, data3[0][1], c='#0a78d1', drawstyle='steps-mid', alpha=0.8, lw=0.8)
			ax[0].text(x_pos, y_pos, 'MRC 0943-242 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')		

			miny,maxy,dy = -1.0, max(data1[0][1])+0.5, 0.5
			ax[0].plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5)
			ax[0].set_yticks( np.arange(miny, maxy, dy) )
			ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
			ax[0].set_ylim([miny+0.1, maxy])
			ax[0].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
			ax[0].set_xlim([-400,400])

			miny,maxy,dy = 0.0, 1.25, 0.5
			ax[1].set_yticks( np.arange(miny, maxy, dy) )
			ax[1].plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5)
			ax[1].set_ylabel(r'$\sigma_{\nu}$', fontsize=fs )
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[1].plot(voff12, data2[0][1], c='grey', alpha=0.5, drawstyle='steps-mid', lw=0.8)

			for ax in ax: 
				ax.tick_params(direction='in', length=4, right=1, top=1, width=1.)
				for pos in ['top','left','bottom', 'right']:
					ax.spines[pos].set_linewidth(1.2)
					ax.spines[pos].set_color('k')

			pl.savefig(self.plot_dir+'MRC0943_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1, dpi=300)

			fig, ax = pl.subplots(4, 2, figsize=(8, 6), sharex=False, sharey=False,
			constrained_layout=True, gridspec_kw={'height_ratios': [1.5,0.5,1.5,0.5]})

			pl.subplots_adjust(wspace=0, hspace=0) 

			# plot 1
			ax[0][0].plot(voff41, data1[3][1], c='k', drawstyle='steps-mid', alpha=1, lw=0.8)
			ax[0][0].plot(voff43, data3[3][1], c='#0a78d1', drawstyle='steps-mid', alpha=0.8, lw=0.8)
			ax[0][0].text(x_pos, y_pos, 'A. North-East (E11)', ha='left', transform=ax[0][0].transAxes, fontsize=fs, color='k')
			ax[0][0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
			ax[0][0].tick_params(axis='y', labelsize=fs)

			ax[1][0].set_ylabel(r'$\sigma_{\nu}$', fontsize=fs )
			ax[1][0].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[1][0].plot(voff42, data2[3][1], c='grey', alpha=1.0, drawstyle='steps-mid', lw=0.8)
			ax[1][0].tick_params(axis='y', labelsize=fs)
			ax[1][0].tick_params(axis='x', labelsize=fs)

			# plot 2
			SW_cont = np.genfromtxt(self.output_dir+'SW_fit_params.txt')[6]
			ax[0][1].plot(voff21, data1[1][1], c='red', alpha=1, lw=0.8)
			# ax[0][1].plot([min(voff21),max(voff21)], [0, 0], c='red', ls='--', lw=0.8)
			ax[0][1].plot(voff22, data2[1][1], c='k', drawstyle='steps-mid', alpha=1.0, lw=0.8)
			indices = [ i for i in range(len(voff22)) if (voff22[i] > -240. and voff22[i] < 180.)  ]
			voff2_sub = [ voff22[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[0][1].fill_between( voff2_sub, 0.0, data2_2_sub, where=data2_2_sub > SW_cont, interpolate=1, color='yellow', alpha=0.5)
			ax[0][1].text(x_pos, y_pos, 'B. Thor (G16a) ', ha='left', transform=ax[0][1].transAxes, fontsize=fs, color='k')
			ax[0][1].set_yticklabels([])

			ax[1][1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[1][1].plot(voff23, data3[1][1], c='grey', alpha=1.0, drawstyle='steps-mid', lw=1)
			ax[1][1].set_yticklabels([])
			ax[1][1].tick_params(axis='x', labelsize=fs)

			# plot 3
			ax[2][1].plot(voff31, data1[2][1], c='k', drawstyle='steps-mid', alpha=1.0, lw=0.8)
			ax[2][1].plot(voff33, data3[2][1], c='#0a78d1', drawstyle='steps-mid', alpha=0.8, lw=0.8)
			ax[2][1].text(x_pos, y_pos, 'C. Loke (G16a)', ha='left', transform=ax[2][1].transAxes, fontsize=fs, color='k')
			ax[2][1].set_yticklabels([])

			ax[3][1].set_yticklabels([])
			ax[3][1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[3][1].plot(voff32, data2[2][1], c='grey', alpha=1.0, drawstyle='steps-mid', lw=1)
			ax[3][1].tick_params(axis='y', labelsize=fs)
			ax[3][1].tick_params(axis='x', labelsize=fs)
		
			ax[2][0].axis("off")
			ax[3][0].axis("off")

			# modify all axes
			spec_ax = [ax[0]] 
			spec_ax = list(chain(*spec_ax)) + [ax[2][1]]
		
			for sa in spec_ax :
				miny,maxy,dy = -1.0, 2.0, 0.5
				sa.set_yticks( np.arange(miny, maxy, dy) )
				sa.plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5, lw=0.8)
				sa.tick_params(direction='in', length=4, right=1, top=1, width=1.)
				sa.set_ylim([-0.99, 2.0])
				sa.set_xlim([-399, 399])

				for pos in ['top','left','bottom', 'right']:
					sa.spines[pos].set_linewidth(1.)
					sa.spines[pos].set_color('k')

			var_ax = [ax[1]] 
			var_ax = list(chain(*var_ax)) + [ax[3][1]]

			for va in var_ax: 
				miny,maxy,dy = -0.5, 1.2, 0.5
				va.set_yticks( np.arange(miny, maxy, dy) )
				va.plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5, lw=0.8)
				va.tick_params(direction='in', length=4, right=1, top=1, width=1.)
				va.set_ylim([miny+0.1, maxy])
				va.set_xlim([-399, 399])

				for pos in ['top','left','bottom', 'right']:
					va.spines[pos].set_linewidth(1.)
					va.spines[pos].set_color('k')

			im = pl.imread(get_sample_data(self.plot_dir+'MRC0943_CI_moment0.png'))
			newax = fig.add_axes([0.03, -0.06, 0.5, 0.5], zorder=-1)
			newax.imshow(im)
			newax.axis('off')

			pl.savefig(self.plot_dir+'MRC0943_CI_CGM_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1, dpi=300)

		else:
			fig, ax = pl.subplots(2, 1, figsize=(4, 3), sharex=True, 
			constrained_layout=True, gridspec_kw={'height_ratios': [3,1]})

			pl.subplots_adjust(hspace=0, wspace=0.01) 
			ax[0].plot(voff1, data1[0][1], c='k', drawstyle='steps-mid', alpha=1.0, lw=0.8)
			ax[0].plot(voff_wide, data3[0][1], c='#0a78d1', drawstyle='steps-mid', alpha=0.8, lw=0.8)
			
			if self.source == 'TNJ0205':
				ax[0].text(x_pos, y_pos, 'TN J0205+2422 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')	
			elif self.source == 'TNJ0121':
				ax[0].text(x_pos, y_pos, 'TN J0121+1320 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')	
			elif self.source == 'TNJ1338':
				ax[0].text(x_pos, y_pos, 'TN J1338-1942 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')	
			elif self.source == '4C04':
				ax[0].text(x_pos, y_pos, '4C.04+11 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')	
			elif self.source == '4C19':
				ax[0].text(x_pos, y_pos, '4C+19.71 Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')	

			vel_data = data1[0][1]
			if min(vel_data) > 0.:
				miny,maxy,dy = roundup(min(vel_data)-0.5), max(vel_data)+0.5, 0.5
			else: 
				miny,maxy,dy = -1.0, max(vel_data)+0.5, 0.5

			ax[0].plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5)
			ax[0].set_yticks( np.arange(miny, maxy, dy) )
			ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
			ax[0].set_ylim([miny+0.1, maxy])
			ax[0].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
			ax[0].set_xlim([-400,400])

			miny,maxy,dy = 0.0, 1.25, 0.5
			ax[1].set_yticks( np.arange(miny, maxy, dy) )
			ax[1].plot([0.,0.], [miny, maxy], c='grey', ls='--', alpha=0.5)
			ax[1].set_ylabel(r'$\sigma_{\nu}$', fontsize=fs )
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			ax[1].plot(voff2, data2[0][1], c='grey', alpha=0.5, drawstyle='steps-mid', lw=0.8)

			for ax in ax: 
				ax.tick_params(direction='in', length=4, right=1, top=1, width=1.)
				for pos in ['top','left','bottom', 'right']:
					ax.spines[pos].set_linewidth(1.2)
					ax.spines[pos].set_color('k')
		
			pl.savefig(self.plot_dir+source+'_CI_spectrums.png', bbox_inches = 'tight', pad_inches = 0.1)
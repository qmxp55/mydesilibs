#!/usr/bin/env python

import os
import astropy.io.fits as fits
import sys
import numpy as np
import random
import string
from astropy import units
from astropy.table import Table
from astropy.coordinates import SkyCoord
import datetime
import matplotlib
#import pymangle
import healpy as hp

TMPDIR   = '/global/cscratch1/sd/raichoor/tmpdir/'


def get_date():
	return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# get a unique id, used when creating temporary catalogues
# [safe, when the same function is called in scripts running at the same time]
def get_pid(size=20,chars=string.ascii_letters+string.digits):
	return ''.join(random.choice(chars) for _ in range(size))


# appending fits tables (with the same format!)
# http://docs.astropy.org/en/stable/io/fits/usage/table.html#appending-tables
# ! apparently it fails when the catalogues have 3D arrays (as e.g., DECAM_APFLUX in dr3) !
def append_fits(INFITSLIST,OUTFITS,COLNAMES=None,VERBOSE=False):
	# uniq id
	pid = get_pid()
	# if no COLNAMES is provided (COLNAMES==None), we take the columns from INFITSLIST[0]
	if (COLNAMES==None):
		COLNAMES = fits.open(INFITSLIST[0])[1].columns.names
	# nb of fits catalogues
	nfits = len(INFITSLIST)
	# trick because the method doesn t work for a large nb of fits
	# (too many hdu opened at the samed time...)
	# so, if many fits, we split in several steps, then merge
	if (nfits>1000):
		if (VERBOSE==True):
			print('too many catalogues (', nfits, ') => we split in slices of 1000 catalogues')
		nslice     = 1000 
		SLICELIST  = []
		count      = 0
		indmax     = -1
		while (indmax<nfits):
			indmin    = count * nslice
			indmax    = (1+count) * nslice
			if (indmax>nfits):
				indmax = nfits
			outfits_i = TMPDIR+'/tmp.slice.'+str(count)+'.fits_'+pid
			if (VERBOSE==True):
				print(outfits_i, indmin, indmax)
			SLICELIST.append(outfits_i)
			a      = append_fits(INFITSLIST[indmin:indmax],outfits_i,COLNAMES=COLNAMES)
			count += 1
		a = append_fits(SLICELIST,OUTFITS,COLNAMES=COLNAMES,VERBOSE=VERBOSE)
		# cleaning
		for SLICE in SLICELIST:
			if (VERBOSE==True):
				print('removing '+SLICE)
			os.remove(SLICE)
	# if not too many fits
	else:
		print(OUTFITS,VERBOSE,nfits)
		nrowslist = np.zeros(nfits,dtype='int')
		hdulist   = np.zeros(nfits,dtype='object')
		for i in xrange(nfits):
			print('reading '+INFITSLIST[i])
			hdulist[i]   = fits.open(INFITSLIST[i])
			# case empty fits...
			if (type(hdulist[i][1].data)!=type(None)):
				nrowslist[i] = len(hdulist[i][1].data)
		nrows = np.sum(nrowslist) # total nb of obj.
		if (VERBOSE==True):
			print('nrows = '+str(nrows))
		# building the table structure: we take the columns (name,format) from INFITSLIST[0]
		collist = []
		for name in COLNAMES:
			if (VERBOSE==True):
				print(name)
			ind = np.where(np.array(hdulist[0][1].columns.names)==name)[0][0]
			collist.append(fits.Column(name=name,format=hdulist[0][1].columns[ind].format,dim=hdulist[0][1].columns[ind].dim))
		cols  = fits.ColDefs(collist)
		hdu   = fits.BinTableHDU.from_columns(cols, nrows=nrows)
		for i in xrange(nfits):
			indstart = np.sum(nrowslist[:i])
			indend   = indstart + nrowslist[i]
			if (VERBOSE==True):
				print(i, INFITSLIST[i], indstart, indend)
			# non-empty?
			if (nrowslist[i]!=0):
				for colname in COLNAMES:
					hdu.data[colname][indstart:indend] = hdulist[i][1].data[colname]
		hdu.writeto(OUTFITS,overwrite=True)
	return True


# tiling directory
def get_tile_dir(CHUNK):
	return '/global/homes/r/raichoor/eBOSS_ELG_masks/'


# get sector infos
def get_sector(RA,DEC,CHUNK):
	# chunk infos
	tiledir = get_tile_dir(CHUNK)
	hdu     = fits.open(tiledir+'geometry-'+CHUNK+'.fits')
	geomsect= hdu[1].data['sector']
	hdu     = fits.open(tiledir+'sector-'+CHUNK+'.fits')
	data    = hdu[1].data
	sector  = data['sector']
	tiles   = data['tiles']
	ntiles  = data['ntiles']
	area    = data['area']
	# objects
	nobj       = len(RA)
	mng        = pymangle.Mangle(tiledir+'geometry-'+CHUNK+'.ply')
	objpolyid  = mng.polyid(RA,DEC)
	objsector  = np.zeros(nobj,dtype='int')-1.
	objtiles   = np.zeros((nobj,tiles.shape[1]),dtype='int')-1
	objntiles  = np.zeros(nobj,dtype='int')
	objarea    = np.zeros(nobj)
	tmparr     = np.arange(len(objtiles)) #
	for i in xrange(len(sector)):
		ind_i = np.where(geomsect==sector[i])[0]
		#print 'i=',i,'sector[i]=',sector[i],'ind_i=',ind_i
		for j in ind_i:
			tmp          = (objpolyid==j)
			#print '\tj=',j,'tmp[tmp].shape=',tmp[tmp].shape
			objsector[tmp] = sector[i]
			objtiles[tmparr[tmp]] = tiles[i,:]
			objntiles[tmp] = ntiles[i]
			objarea[tmp]   = area[i]
	return objsector,objtiles,objntiles,objarea


# ELG masks
def get_elgmaskdict():
	maskdir    = '/global/homes/r/raichoor/eBOSS_ELG_masks/'
	dict_mask  = {}
	dict_mask['rykoff']          = maskdir+'bright_object_mask_rykoff_pix.ply'
	dict_mask['Tycho20Vmag10']   = maskdir+'tycho2mask-0Vmag10.pol'
	dict_mask['Tycho210Vmag11']  = maskdir+'tycho2mask-10Vmag11.pol'
	dict_mask['Tycho211Vmag115'] = maskdir+'tycho2mask-11Vmag115.pol'
	return dict_mask


# bugged depth_ivar flagging
def get_isbugged(bugged_depth_ivar_g,bugged_depth_ivar_r,bugged_depth_ivar_z):
	isbugged_sgc = (bugged_depth_ivar_g<62.79) & (bugged_depth_ivar_r<30.05) & (bugged_depth_ivar_z<12.75)
	isbugged_ngc = (bugged_depth_ivar_g<62.79) | (bugged_depth_ivar_r<30.05) | (bugged_depth_ivar_z<11.00)
	return isbugged_sgc,isbugged_ngc


def get_drbrn(chunklist=None): # chunklist=['eboss21','eboss22']                                                                   
	hdu     = fits.open('../Catalogs/ELG_TS_master.fits')
	data    = hdu[1].data
	if (chunklist!=None):
		keep = np.zeros(len(data),dtype=bool)
		for chunk in chunklist:
			keep[data['CHUNK']==chunk] = True
		data = data[keep]
	alldrbrns= np.array([d+'/'+b for d,b in zip(data['decals_dr'],data['brickname'])])
	drbrns   = np.unique(alldrbrns)
	drs      = np.array([db.split('/')[0] for db in drbrns])
	brns     = np.array([db.split('/')[1] for db in drbrns])
	return drs,brns


# copied from https://github.com/rongpu/desi-examples/blob/master/bright_star_contamination/match_coord.py
def search_around(ra1, dec1, ra2, dec2, search_radius=1., verbose=True):
	'''
	Using the astropy.coordinates.search_around_sky module to find all pairs within
	some search radius.
	Inputs: 
	RA and Dec of two catalogs;
	search_radius (arcsec);
	Outputs: 
		idx1, idx2: indices of matched objects in the two catalogs;
		d2d: angular distances (arcsec);
		d_ra, d_dec: the differences in RA and Dec (arcsec); 
	'''
	# protect the global variables from being changed by np.sort
	ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])
	# Matching catalogs
	sky1 = SkyCoord(ra1*units.degree,dec1*units.degree, frame='icrs')
	sky2 = SkyCoord(ra2*units.degree,dec2*units.degree, frame='icrs')
	idx1, idx2, d2d, d3d = sky2.search_around_sky(sky1, seplimit=search_radius*units.arcsec)
	if verbose:
		print('%d nearby objects'%len(idx1))
	# convert distances to numpy array in arcsec
	d2d   = np.array(d2d.to(units.arcsec))
	d_ra  = (ra2[idx2]-ra1[idx1])*3600.    # in arcsec
	d_dec = (dec2[idx2]-dec1[idx1])*3600. # in arcsec
	##### Convert d_ra to actual arcsecs #####
	mask       = d_ra > 180*3600
	d_ra[mask] = d_ra[mask] - 360.*3600
	mask       = d_ra < -180*3600
	d_ra[mask] = d_ra[mask] + 360.*3600
	d_ra       = d_ra * np.cos(dec1[idx1]/180*np.pi)
	##########################################
	return idx1, idx2, d2d, d_ra, d_dec


# https://desi.lbl.gov/svn/docs/technotes/targeting/target-truth/trunk/python/match_coord.py
# slightly edited (plot_q and keep_all_pairs removed; u => units)
def match_coord(ra1, dec1, ra2, dec2, search_radius=1., nthneighbor=1, verbose=True):
	'''
	Match objects in (ra2, dec2) to (ra1, dec1). 

	Inputs: 
		RA and Dec of two catalogs;
		search_radius: in arcsec;
		(Optional) keep_all_pairs: if true, then all matched pairs are kept; otherwise, if more than
		one object in t2 is match to the same object in t1 (i.e. double match), only the closest pair
		is kept.

	Outputs: 
		idx1, idx2: indices of matched objects in the two catalogs;
		d2d: distances (in arcsec);
		d_ra, d_dec: the differences (in arcsec) in RA and Dec; note that d_ra is the actual angular 
		separation;
	'''
	t1 = Table()
	t2 = Table()
	# protect the global variables from being changed by np.sort
	ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])
	t1['ra'] = ra1
	t2['ra'] = ra2
	t1['dec'] = dec1
	t2['dec'] = dec2
	t1['id'] = np.arange(len(t1))
	t2['id'] = np.arange(len(t2))
	# Matching catalogs
	sky1 = SkyCoord(ra1*units.degree,dec1*units.degree, frame='icrs')
	sky2 = SkyCoord(ra2*units.degree,dec2*units.degree, frame='icrs')
	idx, d2d, d3d = sky2.match_to_catalog_sky(sky1, nthneighbor=nthneighbor)
	# This finds a match for each object in t2. Not all objects in t1 catalog are included in the result. 

	# convert distances to numpy array in arcsec
	d2d = np.array(d2d.to(units.arcsec))
	matchlist = d2d<search_radius
	if np.sum(matchlist)==0:
		if verbose:
			print('0 matches')
		return np.array([], dtype=int), np.array([], dtype=int), np.array([]), np.array([]), np.array([])
	t2['idx'] = idx
	t2['d2d'] = d2d
	t2 = t2[matchlist]
	init_count = np.sum(matchlist)
	#--------------------------------removing doubly matched objects--------------------------------
	# if more than one object in t2 is matched to the same object in t1, keep only the closest match
	t2.sort('idx')
	i = 0
	while i<=len(t2)-2:
		if t2['idx'][i]>=0 and t2['idx'][i]==t2['idx'][i+1]:
			end = i+1
			while end+1<=len(t2)-1 and t2['idx'][i]==t2['idx'][end+1]:
				end = end+1
			findmin = np.argmin(t2['d2d'][i:end+1])
			for j in range(i,end+1):
				if j!=i+findmin:
					t2['idx'][j]=-99
			i = end+1
		else:
			i = i+1

	mask_match = t2['idx']>=0
	t2 = t2[mask_match]
	t2.sort('id')
	if verbose:
		print('Doubly matched objects = %d'%(init_count-len(t2)))
	# -----------------------------------------------------------------------------------------
	if verbose:
		print('Final matched objects = %d'%len(t2))
	# This rearranges t1 to match t2 by index.
	t1 = t1[t2['idx']]
	d_ra = (t2['ra']-t1['ra']) * 3600.    # in arcsec
	d_dec = (t2['dec']-t1['dec']) * 3600. # in arcsec
	##### Convert d_ra to actual arcsecs #####
	mask = d_ra > 180*3600
	d_ra[mask] = d_ra[mask] - 360.*3600
	mask = d_ra < -180*3600
	d_ra[mask] = d_ra[mask] + 360.*3600
	d_ra = d_ra * np.cos(t1['dec']/180*np.pi)
	##########################################
	return np.array(t1['id']), np.array(t2['id']), np.array(t2['d2d']), np.array(d_ra), np.array(d_dec)



# https://matplotlib.org/examples/api/colorbar_only.html
def mycmap(name,n,cmin,cmax):
	cmaporig = matplotlib.cm.get_cmap(name)
	mycol    = cmaporig(np.linspace(cmin, cmax, n))
	cmap     = matplotlib.colors.ListedColormap(mycol)
	cmap.set_under(mycol[0])
	cmap.set_over (mycol[-1])
	return cmap



# fast method to check if ra,dec is in des
'''
def get_isdes(ra,dec):
	nside  = 256
	npix   = hp.nside2npix(nside)
	# checking hp pixels
	mng       = pymangle.Mangle('/global/cscratch1/sd/raichoor/desits/des.ply')
	theta,phi = hp.pix2ang(nside,np.arange(npix),nest=False)
	hpra,hpdec= 180./np.pi*phi,90.-180./np.pi*theta
	hpindes   = (mng.polyid(hpra,hpdec)!=-1).astype(int)
	# pixels with all neighbours in des
	hpindes_secure = np.array([i for i in xrange(npix) 
				if hpindes[i]+hpindes[hp.get_all_neighbours(nside,i)].sum()==9])
	# pixels with all neighbours outside des
	hpoutdes_secure = np.array([i for i in xrange(npix) 
				if hpindes[i]+hpindes[hp.get_all_neighbours(nside,i)].sum()==0])
	# hpind to be checked
	tmp    = np.ones(npix,dtype=bool)
	tmp[hpindes_secure] = False
	tmp[hpoutdes_secure]= False
	hp_tbc = np.arange(npix)[tmp]
	# now checking indiv. obj. in the tbc pixels
	hppix     = hp.ang2pix(nside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=False)
	hpind     = np.unique(hppix)
	#
	isdes     = np.zeros(len(ra),dtype=bool)
	isdes[np.in1d(hppix,hpindes_secure)] = True
	tbc       = np.where(np.in1d(hppix,hp_tbc))[0]
	tbcisdes  = (mng.polyid(ra[tbc],dec[tbc])!=-1)
	isdes[tbc][tbcisdes] = True
	#
	return isdes
'''
def get_isdes(ra,dec):
	hdu = fits.open('/global/cscratch1/sd/raichoor/desits/des_hpmask.fits')
	nside,nest = hdu[1].header['HPXNSIDE'],hdu[1].header['HPXNEST']
	hppix     = hp.ang2pix(nside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
	isdes     = np.zeros(len(ra),dtype=bool)
	isdes[np.in1d(hppix,hdu[1].data['hppix'])] = True
	return isdes

#
def subsample(infits,subsample,outfits):
	hdu  = fits.open(infits)
	nobj = hdu[1].header['NAXIS2']
	keep = np.random.choice(nobj,size=int(subsample*nobj),replace=False)
	hdu[1].data = hdu[1].data[keep]
	hdu.writeto(outfits,overwrite=True)
	return True


#
def get_footprint(survey):
	mydict = {}
	if (survey=='hscdr1'):
		mydict['xmm']     = '29,40,-7,-3'
		mydict['gama09h'] = '130,138,-1,3'
		mydict['cosmos']  = '148,152,0,4'
		mydict['wide12h'] = '177,183,-2,2'
		mydict['gama15h'] = '213,221,-2,2'
		mydict['vvds']    = '331,342,-1,3'
		mydict['deep2-3'] = '350,354,-2,2'
		mydict['hectomap']= '242,249,42,45'
		mydict['aegis']   = '213,217,51,54'
		mydict['elais-n1']= '240,246,53,57'
	elif (survey=='hscdr2'):
		mydict['w05_1']   = '330,4,-2,3'
		mydict['w05_2']   = '330,345,3,7'
		mydict['w01']     = '15,23,-2,3'
		mydict['w02']     = '28,40,-8,0'
		mydict['w04']     = '172,195,-2.5,2.5'
		mydict['w06']     = '212,251,41.5,45'
		mydict['w07']     = '213,217,51,54'
	elif (survey=='deep2'):
		mydict['deep2_1'] = '213,217,51,54'
		mydict['deep2_2'] = '251,254,34,36'
		mydict['deep2_3'] = '351,354,-1,1'
		mydict['deep2_4'] = '36,39,0,1'
	elif (survey=='kids'):
		mydict['s1']      = '0,53.5,-35.6,-25.7'
		mydict['s2']      = '329.5,360,-35.6,-25.7'
		mydict['n1']      = '156,225,-5,4'
		mydict['n2']      = '225,238,-3,4'
	elif (survey=='ebosselg'):
		mydict['eboss21'] = '315,360,-2,2'
		mydict['eboss22'] = '0,45,-5,5'
		mydict['eboss23a']= '126,143,16,29'
		mydict['eboss23b']= '136,157,13,27'
		mydict['eboss25a']= '131,166,29,32.5'
		mydict['eboss25b']= '142.5,166,27,29'
		mydict['eboss25c']= '157,166,23,27'
	elif (survey=='dr8b'):
		mydict['s82']     = '0,45,-1.25,1.25'
		mydict['hsc_sgc'] = '30,40,-6.5,-1.25'
		mydict['hsc_ngc'] = '177.5,182.5,-1,1'
		mydict['edr']     = '240,245,5,12'
		mydict['hsc_north']='240,250,42,45'
		mydict['egs']     = '213,216.5,52,54'
	elif (survey=='cosmos'):
		mydict['cosmos']  = '149,151.5,1,3.5'
	return mydict



	

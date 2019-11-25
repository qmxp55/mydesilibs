import numpy as np
import sys, os, time, argparse, glob
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.ticker import NullFormatter
import astropy.units as units
from astropy import units as u
import pandas as pd
from astropy.io import ascii
from photometric_def import get_stars, get_galaxies
from scipy import optimize

import raichoorlib
np.seterr(divide='ignore') # ignode divide by zero warnings
import astropy.io.fits as fits
import healpy as hp

#from desiutil.log import get_logger, DEBUG
#log = get_logger()
class Point:

    def __init__(self, xcoord=0, ycoord=0):
        self.x = xcoord
        self.y = ycoord

class Rectangle:
    def __init__(self, bottom_left, top_right, colour):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.colour = colour

    def intersects(self, other):
        return not (self.top_right.x <= other.bottom_left.x or self.bottom_left.x >= other.top_right.x or self.top_right.y <= other.bottom_left.y or self.bottom_left.y >= other.top_right.y)
    
    def plot(self, other):
        fig, ax = plt.subplots(figsize=(15,8))
        rect = patches.Rectangle((self.bottom_left.x,self.bottom_left.y), abs(self.top_right.x - self.bottom_left.x), abs(self.top_right.y - self.bottom_left.y),linewidth=1.5, alpha=0.5, color='r')
        rect2 = patches.Rectangle((other.bottom_left.x,other.bottom_left.y), abs(other.top_right.x - other.bottom_left.x), abs(other.top_right.y - other.bottom_left.y),linewidth=1.5, alpha=0.5, color='blue')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        xlims = np.array([self.bottom_left.x, self.top_right.x, other.bottom_left.x, other.top_right.x])
        ylims = np.array([self.bottom_left.y, self.top_right.y, other.bottom_left.y, other.top_right.y])
        ax.set_xlim(xlims.min()-1, xlims.max()+1)
        ax.set_ylim(ylims.min()-1, ylims.max()+1)
        
def cut(ramin, ramax, decmin, decmax, catalog):
    
    mask = np.logical_and(catalog['RA'] >= ramin, catalog['RA'] <= ramax)
    mask &= np.logical_and(catalog['DEC'] >= decmin, catalog['DEC'] <= decmax)
    cat = catalog[mask]
    
    return cat

def get_sweep_patch(patch, rlimit=None, dr='dr7'):
    """
    Extract data from DECaLS DR7 SWEEPS files only.
    
    Parameters
    ----------
    patch: :class:`array-like`
        Sky coordinates in RA and DEC of the rectangle/square patch in format [RAmin, RAmax, DECmin, DECmax]
    rlimit: :class:`float`
        magnitude limit of data in the r-band with extinction correction
    
    Returns
    -------
    Subsample catalogue of SWEEP data.
    The subsample catalogue will be also stored with name 'sweep_RAmin_RAmax_DECmin_DECmax_rmag_rlimit' and numpy format '.npy'
    
    """
    import time
    start = time.time()
    
    if len(patch) != 4:
        log.warning('This require an input array of four arguments of the form [RAmin, RAmax, DECmin, DECmax]')
        raise ValueError
        
    if rlimit is None:
        log.warning('rlimit input is required')
        raise ValueError

    #patch
    ramin, ramax, decmin, decmax = patch[0], patch[1], patch[2], patch[3]
    sweep_file_name = '%s_sweep_%s_%s_%s_%s_rmag_%s' %(dr, str(ramin), str(ramax), str(decmin), str(decmax), str(rlimit))
    sweep_file = os.path.isfile(sweep_file_name+'.npy')
    if not sweep_file:
        if dr is 'dr7':
            sweep_dir = os.path.join('/global/project/projectdirs/cosmo/data/legacysurvey/','dr7', 'sweep', '7.1')
        elif dr == 'dr8-south':
            #print('HERE!!!!!!!!')
            #sweep_dir = '/global/cscratch1/sd/adamyers/dr8/decam/sweep'
            sweep_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0'
            #sweep_dir = '/global/cscratch1/sd/ioannis/dr8/decam/sweep-patched'
            
        elif dr is 'dr8-north':
            sweep_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0'
            
        df = cut_sweeps(ramin, ramax, decmin, decmax, sweep_dir, rlimit=rlimit)
        np.save(sweep_file_name, df)
    else:
        print('sweep file already exist at:%s' %(os.path.abspath(sweep_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    get_area(patch, get_val = False)
    print('Weight of %s catalogue: %s' %(sweep_file_name+'.npy', convert_size(os.path.getsize(sweep_file_name+'.npy'))))
    
    if not sweep_file:
        
        return df
    else:
        return np.load(os.path.abspath(sweep_file_name+'.npy'))
    
def convert_size(size_bytes): 
    import math
    if size_bytes == 0: 
            return "0B" 
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB") 
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i) 
    size = round(size_bytes / power, 2) 
    return "%s %s" % (size, size_name[i])
    
def get_area(patch, get_val = False):
    
    alpha1 = np.radians(patch[0])
    alpha2 = np.radians(patch[1])
    delta1 = np.radians(patch[2])
    delta2 = np.radians(patch[3])
    
    A = (alpha2 - alpha1)*(np.sin(delta2) - np.sin(delta1))*(180/np.pi)**(2)
    print('Area of %g < RA < %g & %g < DEC < %g: %2.4g deg^2' %(patch[0],
                                    patch[1], patch[2], patch[3], A))
    if get_val:
        return A

    
def cut_sweeps(ramin, ramax, decmin, decmax, sweep_dir, rlimit=None):
    '''Main function to extract the data from the SWEEPS'''
    
    cat1_paths = sorted(glob.glob(os.path.join(sweep_dir, '*.fits')))
    j = 0
    
    for fileindex in range(len(cat1_paths)):

        cat1_path = cat1_paths[fileindex]
        filename = cat1_path[-26:-5]
        brick = cat1_path[-20:-5]
        ra1min = float(brick[0:3])
        ra1max = float(brick[8:11])
        dec1min = float(brick[4:7])
        if brick[3]=='m':
            dec1min = -dec1min
        dec1max = float(brick[-3:])
        if brick[-4]=='m':
            dec1max = -dec1max
        
        r1=Rectangle(Point(ramin,decmin), Point(ramax, decmax), 'red')
        r2=Rectangle(Point(ra1min, dec1min), Point(ra1max, dec1max), 'blue')
        
        if not r1.intersects(r2):
            continue
        
        if j == 0:
            cat = fitsio.read(cat1_path, ext=1)
            cat = cut(ramin, ramax, decmin, decmax, cat)
            if rlimit != None:
                rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                print('%s with %i objects out of %i at rmag=%2.2g' %(filename, len(cat[rflux > 10**((22.5-rlimit)/2.5)]), len(cat), rlimit))
                cat = cat[rflux > 10**((22.5-rlimit)/2.5)]
            else:
                print('%s with %i objects' %(filename, len(name)))
            j += 1
            continue
        
        name = fitsio.read(cat1_path, ext=1)
        name = cut(ramin, ramax, decmin, decmax, name)
        if rlimit != None:
                rflux2 = name['FLUX_R'] / name['MW_TRANSMISSION_R']
                print('%s with %i objects out of %i at rmag=%2.2g' %(filename, len(name[rflux2 > 10**((22.5-rlimit)/2.5)]), len(name), rlimit))
                name = name[rflux2 > 10**((22.5-rlimit)/2.5)]
        else:
            print('%s with %i objects' %(filename, len(cat)))
        
        cat = np.concatenate((cat, name))
        j += 1
        
    print('Bricks that matched: %i' %(j))
    print('Sample region # objects: %i' %(len(cat)))
    
    return cat

def get_random(N=3, sweepsize=None, dr='dr8'):
    
    import time
    start = time.time()
        
    if (N < 2):
        log.warning('Number of RANDOMS files must be greater than one')
        raise ValueError
    
    import glob
    #ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.29.0/' #issues with MASKBITS...
    
    dirpath = '/global/cscratch1/sd/qmxp55/'
    random_file_name = '%s_random_N%s' %(dr, str(N))
        
    random_file = os.path.isfile(dirpath+random_file_name+'.npy')
    if not random_file:
        if dr is 'dr7':
            ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.22.0/'
            randoms = glob.glob(ranpath + 'randoms*')
        elif (dr == 'dr8'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr8/0.31.0/randoms/'
            randoms = glob.glob(ranpath + 'randoms-inside*')
            
        randoms.sort()
        randoms = randoms[0:N]

        for i in range(len(randoms)):
            df_ran = fitsio.read(randoms[i], columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS'],upper=True, ext=1)
        
            if i == 0:
                df_ranS1 = df_ran
                continue
        
            df_ranS1 = np.concatenate((df_ranS1, df_ran))
            
        np.save(dirpath+random_file_name, df_ranS1)
            
        print('# objects in RANDOM: %i' %(len(df_ranS1)))
        if sweepsize is not None:
            print('fraction of RANDOM catalogue compared to SWEEP catalogue: %2.3g' 
                      %(len(df_ranS1)/sweepsize))
    else:
        print('RANDOM file already exist at:%s' %(os.path.abspath(dirpath+random_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    print('Weight of %s catalogue: %s' %(dirpath+random_file_name+'.npy', convert_size(os.path.getsize(dirpath+random_file_name+'.npy'))))
    
    if not random_file:
        return df_ranS1
    else:
        return np.load(dirpath+random_file_name+'.npy')
    

def get_random_patch(patch, N=3, sweepsize=None, dr='dr7'):
    
    import time
    start = time.time()
    
    if len(patch) != 4:
        log.warning('This require an input array of four arguments of the form [RAmin, RAmax, DECmin, DECmax]')
        raise ValueError
        
    if (dr is 'dr7') & (N < 2):
        log.warning('Number of RANDOMS files must be greater than one')
        raise ValueError
    
    import glob
    #ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.29.0/' #issues with MASKBITS...
    
    ramin, ramax, decmin, decmax = patch[0], patch[1], patch[2], patch[3]
    random_file_name = '%s_random_%s_%s_%s_%s_N_%s' %(dr, str(ramin), str(ramax), str(decmin), str(decmax), str(N))
        
    random_file = os.path.isfile(random_file_name+'.npy')
    if not random_file:
        if dr is 'dr7':
            ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.22.0/'
            randoms = glob.glob(ranpath + 'randoms*')
        elif (dr == 'dr8-south') or (dr == 'dr8-north'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr8/0.31.0/randoms/'
            randoms = glob.glob(ranpath + 'randoms-inside*')
            
        randoms.sort()
        randoms = randoms[0:N]

        for i in range(len(randoms)):
            df_ran = fitsio.read(randoms[i], columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS'],upper=True, ext=1)
            df_ranS = cut(ramin, ramax, decmin, decmax, df_ran)

            print('total Size of randoms in %s: %i (within patch: %2.3g %%)' 
                  %(randoms[i][-9:-5], len(df_ran), 100*len(df_ranS)/len(df_ran)))
        
            if i == 0:
                df_ranS1 = df_ranS
                continue
        
            df_ranS1 = np.concatenate((df_ranS1, df_ranS))
            
        #elif dr is 'dr8c':
        #    ranpath = '/project/projectdirs/desi/target/catalogs/dr8c/PR490/'
        #    df_ran = fitsio.read(ranpath+'randoms-dr8c-PR490.fits', columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS'],upper=True, ext=1)
        #    df_ranS1 = cut(ramin, ramax, decmin, decmax, df_ran)
            
        np.save(random_file_name, df_ranS1)
            
        print('# objects in RANDOM patch: %i' %(len(df_ranS1)))
        if sweepsize is not None:
            print('fraction of RANDOM catalogue in patch compared to SWEEP catalogue in patch: %2.3g' 
                      %(len(df_ranS1)/sweepsize))
    else:
        print('RANDOM file already exist at:%s' %(os.path.abspath(random_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    get_area(patch, get_val = False)
    print('Weight of %s catalogue: %s' %(random_file_name+'.npy', convert_size(os.path.getsize(random_file_name+'.npy'))))
    
    if not random_file:
        return df_ranS1
    else:
        return np.load(os.path.abspath(random_file_name+'.npy'))
    
def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

def get_mag_decals(df):
    
    #df = df[(df['FLUX_R'] > 0) & (df['FLUX_G'] > 0) & (df['FLUX_Z'] > 0) & (df['FLUX_W1'] > 0)]
    rmag =  flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
    gmag = flux_to_mag(df['FLUX_G']/df['MW_TRANSMISSION_G'])
    zmag = flux_to_mag(df['FLUX_Z']/df['MW_TRANSMISSION_Z'])
    w1mag = flux_to_mag(df['FLUX_W1']/df['MW_TRANSMISSION_W1'])
    Gmag = df['GAIA_PHOT_G_MEAN_MAG']
    rr = flux_to_mag(df['FLUX_R'])
    AEN = df['GAIA_ASTROMETRIC_EXCESS_NOISE']
    
    c = SkyCoord(df['RA']*units.degree,df['DEC']*units.degree, frame='icrs')
    galb = c.galactic.b.value # galb coordinate
    
    keep = np.ones_like(rmag, dtype='?')
    for i in [rmag, gmag, zmag, w1mag, Gmag, rr]:
        keep &= np.isfinite(i)
    #print('fraction of finite values: %i / %i' %(np.sum(keep), len(keep)))

    return gmag, rmag, zmag, w1mag, Gmag, rr, AEN, galb, keep

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
    sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')
    idx1, idx2, d2d, d3d = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
    if verbose:
        print('%d nearby objects ~ %g %%'%(len(idx1), 100*len(idx1)/len(ra2)))

    # convert distances to numpy array in arcsec
    d2d = np.array(d2d.to(u.arcsec))


    d_ra = (ra2[idx2]-ra1[idx1])*3600.    # in arcsec
    d_dec = (dec2[idx2]-dec1[idx1])*3600. # in arcsec
    ##### Convert d_ra to actual arcsecs #####
    mask = d_ra > 180*3600
    d_ra[mask] = d_ra[mask] - 360.*3600
    mask = d_ra < -180*3600
    d_ra[mask] = d_ra[mask] + 360.*3600
    d_ra = d_ra * np.cos(dec1[idx1]/180*np.pi)
    ##########################################

    return idx1, idx2, d2d, d_ra, d_dec

#
def getGeoCuts(df):
    
    #GEO cuts CATALOGUE
    BS = (np.uint64(df['MASKBITS']) & np.uint64(2**1))==0
    MS = (np.uint64(df['MASKBITS']) & np.uint64(2**11))==0
    GC = (np.uint64(df['MASKBITS']) & np.uint64(2**13))==0
    LG = ((np.uint64(df['MASKBITS']) & np.uint64(2**12))==0)
    allmask = ((df['MASKBITS'] & 2**6) == 0) | ((df['MASKBITS'] & 2**5) == 0) | ((df['MASKBITS'] & 2**7) == 0)
    nobs = (df['NOBS_G'] > 0) | (df['NOBS_R'] > 0) | (df['NOBS_Z'] > 0)
    
    GeoCut = {'BS':BS,
              'MS':MS,
              'GC':GC,
              'LG':LG,
              'allmask':allmask,
              'nobs':nobs
             }
    
    return GeoCut

#
def getPhotCuts(df):
    
    rmag = flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
    gmag = flux_to_mag(df['FLUX_G']/df['MW_TRANSMISSION_G'])
    zmag = flux_to_mag(df['FLUX_Z']/df['MW_TRANSMISSION_Z'])
    rfibmag = flux_to_mag(df['FIBERFLUX_R']/df['MW_TRANSMISSION_R'])
    nomask = np.zeros_like(df['RA'], dtype='?')

    #Photometric cuts CATALOGUE
    STARS = get_stars(gaiagmag=df['GAIA_PHOT_G_MEAN_MAG'], fluxr=df['FLUX_R'])
    GAL = get_galaxies(gaiagmag=df['GAIA_PHOT_G_MEAN_MAG'], fluxr=df['FLUX_R'])
    
    FMC = nomask.copy()
    FMC |= ((rfibmag < (2.9 + 1.2) + rmag) & (rmag < 17.1))
    FMC |= ((rfibmag < 21.2) & (rmag < 18.3) & (rmag > 17.1))
    FMC |= ((rfibmag < 2.9 + rmag) & (rmag > 18.3))
    
    CC = ~nomask.copy()
    CC &= ((gmag - rmag) > -1.)
    CC &= ((gmag - rmag) < 4.)
    CC &= ((rmag - zmag) > -1.)
    CC &= ((rmag - zmag) < 4.)

    QC_FM = ~nomask.copy()
    QC_FM &= (df['FRACMASKED_R'] < 0.4)
    QC_FM &= (df['FRACMASKED_G'] < 0.4)
    QC_FM &= (df['FRACMASKED_Z'] < 0.4)
    
    QC_FI = ~nomask.copy()
    QC_FI &= (df['FRACIN_R'] > 0.3) 
    QC_FI &= (df['FRACIN_G'] > 0.3) 
    QC_FI &= (df['FRACIN_Z'] > 0.3) 
    
    QC_FF = ~nomask.copy()
    QC_FF &= (df['FRACFLUX_R'] < 5.) 
    QC_FF &= (df['FRACFLUX_G'] < 5.)  
    QC_FF &= (df['FRACFLUX_Z'] < 5.) 
    
    QC_IVAR = ~nomask.copy()
    QC_IVAR &= (df['FLUX_IVAR_R'] > 0.)  
    QC_IVAR &= (df['FLUX_IVAR_G'] > 0.)  
    QC_IVAR &= (df['FLUX_IVAR_Z'] > 0.) 

    
    PhotCut = {'SG':GAL,
              'FMC':FMC,
              'CC':CC,
              'QC_FM':QC_FM,
              'QC_FI':QC_FI,
              'QC_FF':QC_FF,
              'QC_IVAR':QC_IVAR
             }
    
    return PhotCut

def get_bgs(df):
    
    geocuts = getGeoCuts(df)
    photcuts = getPhotCuts(df)
    bgscuts = geocuts
    bgscuts.update(photcuts)
    
    bgs = np.ones_like(df['RA'], dtype='?')
    for key, val in zip(bgscuts.keys(), bgscuts.values()):
        if (key == 'allmask') or (key == 'MS'): continue
        else: bgs &= val
        
    return bgs


def get_sweep_whole(dr='dr8-south', rlimit=None, maskbitsource=False, bgsbits=False, opt='1'):
    """
    Extract data from DECaLS DR7 SWEEPS files only.
    
    Parameters
    ----------
    patch: :class:`array-like`
        Sky coordinates in RA and DEC of the rectangle/square patch in format [RAmin, RAmax, DECmin, DECmax]
    rlimit: :class:`float`
        magnitude limit of data in the r-band with extinction correction
    
    Returns
    -------
    Subsample catalogue of SWEEP data.
    The subsample catalogue will be also stored with name 'sweep_RAmin_RAmax_DECmin_DECmax_rmag_rlimit' and numpy format '.npy'
    
    """
    import time
    start = time.time()
    
    namelab = []
    if rlimit is not None: namelab.append('rlimit_%s' %(str(rlimit)))
    if maskbitsource: namelab.append('maskbitsource')

    if len(namelab): sweep_file_name = '%s_sweep_whole_%s' %(dr, '_'.join(namelab))
    else: sweep_file_name = '%s_sweep_whole' %(dr)
        
    sweepdir = '/global/cscratch1/sd/qmxp55/sweep_files/'
    sweep_file = os.path.isfile(sweepdir+sweep_file_name+'.npy')
    sweep_dir_dr7 = os.path.join('/global/project/projectdirs/cosmo/data/legacysurvey/','dr7', 'sweep', '7.1')
    sweep_dir_dr8south = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0'
    sweep_dir_dr8north = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0'
    
    if not sweep_file:
        if dr is 'dr7': df = cut_sweeps(sweep_dir=sweep_dir_dr7, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr == 'dr8-south': df = cut_sweeps(sweep_dir=sweep_dir_dr8south, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr is 'dr8-north': df = cut_sweeps(sweep_dir=sweep_dir_dr8north, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
        elif dr is 'dr8':
            print('getting data in the SOUTH')
            dfsouth = cut_sweeps(sweep_dir=sweep_dir_dr8south, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
            print('getting data in the NORTH')
            dfnorth = cut_sweeps(sweep_dir=sweep_dir_dr8north, rlimit=rlimit, maskbitsource=maskbitsource, opt=opt)
            if opt == '1': df = np.concatenate((dfnorth, dfsouth))
                
        tab = Table()
        for col in df.dtype.names:
            if (col[:4] == 'FLUX') & (col[:9] != 'FLUX_IVAR'): tab[col[-1:]+'MAG'] = flux_to_mag(df['FLUX_'+col[-1:]]/df['MW_TRANSMISSION_'+col[-1:]])
            elif col[:2] == 'MW': continue
            elif col == 'FIBERFLUX_R': tab['RFIBERMAG'] = flux_to_mag(df[col]/df['MW_TRANSMISSION_R'])
            elif col == 'GAIA_PHOT_G_MEAN_MAG': tab['G'] = df[col]
            elif col == 'GAIA_ASTROMETRIC_EXCESS_NOISE': tab['AEN'] = df[col]
            else: tab[col] = df[col]
        tab['FLUX_R'] = df['FLUX_R']
        
        # create BGSBITS: bits associated to selection criteria
        if bgsbits:
            
            geocuts = getGeoCuts(df)
            photcuts = getPhotCuts(df)
            bgscuts = geocuts
            bgscuts.update(photcuts)
            
            BGSBITS = np.zeros_like(df['RA'], dtype='i8')
            BGSMASK = {}
            #[BGSMASK[key] = bit for bit, key in enumerate(bgscuts.keys())]
            
            print('---- BGSMASK key: ---- ')
            for bit, key in enumerate(bgscuts.keys()):
                
                BGSBITS |= bgscuts[key] * 2**(bit)
                BGSMASK[key] = bit
                print('\t %s, %i, %i' %(key, bit, 2**bit))
            # bgs selection
            bgs = np.ones_like(df['RA'], dtype='?')
            for key, val in zip(bgscuts.keys(), bgscuts.values()):
                if (key == 'allmask') or (key == 'MS'): continue
                else: bgs &= val
            BGSBITS |= bgs * 2**(15)
            BGSMASK['bgs'] = 15
            print('\t %s, %i, %i' %('bgs', 15, 2**15))
            
            tab['BGSBITS'] = BGSBITS
            
            # sanity check...
            print('---- Sanity Check ---- ')
            for bit, key in zip(BGSMASK.values(), BGSMASK.keys()):
                if key == 'bgs': print('\t %s, %i, %i' %(key, np.sum(bgs), np.sum((BGSBITS & 2**(bit)) != 0)))
                else: print('\t %s, %i, %i' %(key, np.sum(bgscuts[key]), np.sum((BGSBITS & 2**(bit)) != 0)))
                
            #print('\t %s, %i, %i' %('all', np.sum(bgs), np.sum(BGSBITS != 0)))
                
        print(sweepdir+sweep_file_name)
        np.save(sweepdir+sweep_file_name, tab)
    else:
        print('sweep file already exist at:%s' %(os.path.abspath(sweepdir+sweep_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    #get_area(patch, get_val = False)
    #print('Weight of %s catalogue: %s' %(sweep_file_name+'.npy', convert_size(os.path.getsize(sweep_file_name+'.npy'))))
    
    if not sweep_file: 
        if opt == '1': return tab
        if opt == '2': return dfsouth, dfnorth
    else: return np.load(os.path.abspath(sweepdir+sweep_file_name+'.npy'))
    
    
def cut_sweeps(sweep_dir, rlimit=None, maskbitsource=False, opt='1'):
    '''Main function to extract the data from the SWEEPS'''
    
    sweepfiles = sorted(glob.glob(os.path.join(sweep_dir, '*.fits')))
    cols = ['RA', 'DEC', 'FLUX_R', 'FLUX_G', 'FLUX_Z', 'FIBERFLUX_R', 'MW_TRANSMISSION_R', 
                'MW_TRANSMISSION_G', 'MW_TRANSMISSION_Z','MASKBITS', 'REF_CAT', 'REF_ID', 
                    'GAIA_PHOT_G_MEAN_MAG', 'GAIA_ASTROMETRIC_EXCESS_NOISE', 'FRACFLUX_G', 
                        'FRACFLUX_R', 'FRACFLUX_Z', 'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z',
                             'FRACIN_G', 'FRACIN_R', 'FRACIN_Z', 'TYPE', 'FLUX_IVAR_R', 'FLUX_IVAR_G',
                                   'FLUX_IVAR_Z', 'NOBS_G', 'NOBS_R', 'NOBS_Z']
    
    #sweepfiles = sweepfiles[:2]
    
    if opt == '1':
        print('--------- OPTION 1 ---------')
        for i, file in enumerate(sweepfiles):
        
            if i == 0:
                cat = fitsio.read(file, columns=cols, upper=True, ext=1)
                keep = np.ones_like(cat, dtype='?')
                if rlimit != None:
                    rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                    keep &= rflux > 10**((22.5-rlimit)/2.5)
                
                if maskbitsource: 
                    keep &= (cat['REF_CAT'] != b'  ')
                
                print('fraction: %i/%i objects' %(np.sum(keep), len(cat)))
                cat0 = cat[keep]
                continue
        
            cat = fitsio.read(file, columns=cols, upper=True, ext=1)
            keep = np.ones_like(cat, dtype='?')
            if rlimit != None:
                rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                keep &= rflux > 10**((22.5-rlimit)/2.5)
                    
            if maskbitsource:
                keep &= (cat['REF_CAT'] != b'  ')
                        
            print('fraction: %i/%i objects' %(np.sum(keep), len(cat)))
        
            cat0 = np.concatenate((cat[keep], cat0))
        print('Sample # objects: %i' %(len(cat0)))
            
    if opt == '2':
        print('--------- OPTION 2 ---------')
        cat0 = {}
        for i in cols: cat0[i] = []
        for i, file in enumerate(sweepfiles):
            
            cat = fitsio.read(file, columns=cols, upper=True, ext=1)
            keep = np.ones_like(cat, dtype='?')
            if rlimit != None:
                rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                keep &= rflux > 10**((22.5-rlimit)/2.5)
                
            if maskbitsource:
                keep &= (cat['REF_CAT'] != b'  ')
                
            print('fraction: %i/%i objects' %(np.sum(keep), len(cat)))
            
            for j in cols:
                cat0[j] += cat[j][keep].tolist()
                
        
        print('Sample # objects: %i' %(len(cat0[list(cat0.keys())[0]])))
    
    return cat0

def bgsmask():
    
    mask = {'BS':0,
            'MS':1,
            'GC':2,
            'LG':3,
            'allmask':4,
            'nobs':5,
            'SG':6,
            'FMC':7,
            'CC':8,
            'QC_FM':9,
            'QC_FI':10,
            'QC_FF':11,
            'QC_IVAR':12
            }
    
    return mask


def get_dict(cat=None, randoms=None, pixmapfile=None, hppix_ran=None, hppix_cat=None, maskrand=None, maskcat=None, 
                 getnobs=False, nside=None, npix=None, nest=None, pixarea=None, Nranfiles=None, 
                     ranindesi=None, catindesi=None, dec_resol_ns=32.375, namesels=None, galb=None, survey='main', desifootprint=True):

   # start = raichoorlib.get_date()
    # creating dictionary
    hpdict = {}
    
    if (nside is None) or (npix is None) or (nest is None) or (pixarea is None) & (pixmapfile is not None):
        
        hdr          = fits.getheader(pixmapfile,1)
        nside,nest   = hdr['hpxnside'],hdr['hpxnest']
        npix         = hp.nside2npix(nside)
        pixarea      = hp.nside2pixarea(nside,degrees=True)
    else: raise ValueErro('if not pixel information given, include pixmapfile to compute them.')
    
    if (cat is None) or (pixmapfile is None) or (namesels is None) or (Nranfiles is None):
        raise ValueError('cat, pixmapfile, namesels and Nranfiles can not be None.')
    
    if (getnobs) and (randoms is None):
        raise ValueError('random catalogue can not be None when getnobs is set True.')
        
    if (hppix_ran is None) and (randoms is not None):
        hppix_ran = hp.ang2pix(nside,(90.-np.array(randoms['DEC']))*np.pi/180.,np.array(randoms['RA'])*np.pi/180.,nest=nest)
    elif (hppix_ran is None) and (randoms is None):
        raise ValueError('include a random catalogue to compute their hp pixels indexes.')
        
    if (hppix_cat is None):
        hppix_cat = hp.ang2pix(nside,(90.-cat['DEC'])*np.pi/180.,cat['RA']*np.pi/180.,nest=nest) # catalogue hp pixels array
    
    if (ranindesi is None) and (randoms is not None):
        ranindesi = get_isdesi(randoms['RA'],randoms['DEC']) # True if is in desi footprint
    elif (ranindesi is None) and (randoms is None):
        raise ValueError('include a random catalogue to compute ranindesi.')
        
    if (catindesi is None):
        catindesi = get_isdesi(cat['RA'],cat['DEC']) # True is is in desi footprint
        
    if galb is None:
        c = SkyCoord(cat['RA']*units.degree,cat['DEC']*units.degree, frame='icrs')
        galb = c.galactic.b.value # galb coordinate
        
    theta,phi  = hp.pix2ang(nside,np.arange(npix),nest=nest)
    hpdict['ra'],hpdict['dec'] = 180./np.pi*phi,90.-180./np.pi*theta
    c = SkyCoord(hpdict['ra']*units.degree,hpdict['dec']*units.degree, frame='icrs')
    hpdict['gall'],hpdict['galb'] = c.galactic.l.value,c.galactic.b.value

    # is in desi tile?
    hpdict['isdesi'] = get_isdesi(hpdict['ra'],hpdict['dec'])
    print('positions and desifotprint DONE...')

    # propagating some keys from ADM pixweight
    hdu = fits.open(pixmapfile)
    data = hdu[1].data
    for key in ['HPXPIXEL', 'FRACAREA', 
            'STARDENS', 'EBV', 
            'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
            'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
            'PSFDEPTH_W1', 'PSFDEPTH_W2',
            'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']:
        if (key=='STARDENS'):
            hpdict[key.lower()] = np.log10(data[key])
        elif (key[:8]=='GALDEPTH'):
            hpdict[key.lower()] = 22.5-2.5*np.log10(5./np.sqrt(data[key]))
        else:
            hpdict[key.lower()] = data[key]
    print('systematics DONE...')
        
    # computing fracareas
    randdens = 5000*Nranfiles
    print('randdens = ', randdens, ' ; len randoms = ', len(hppix_ran))
    if desifootprint: mainfootprint = ranindesi
    else: mainfootprint = np.ones_like(hppix_ran, dtype='?')
    if maskrand is None:
        ind,c           = np.unique(hppix_ran[mainfootprint],return_counts=True)
    else:
        ind,c           = np.unique(hppix_ran[(maskrand) & (mainfootprint)],return_counts=True)
    hpdict['bgsfracarea']      = np.zeros(npix)
    hpdict['bgsfracarea'][ind] = c / randdens / pixarea
    print('bgsfracarea DONE...')
    
    # computing nobs
    if getnobs:
        import pandas as pd
        if maskrand is None:
            s = pd.Series(hppix_ran[mainfootprint])
        else:
            s = pd.Series(hppix_ran[(maskrand) & (mainfootprint)])
        d = s.groupby(s).groups
        for i in ['NOBS_G', 'NOBS_R', 'NOBS_Z']:
            hpdict[i] = np.zeros(npix)
            for j in d.keys():
                hpdict[i][j] = np.mean(randoms[i][d[j]])
        print('nobs DONE...')
        
    # north/south/des/decals
    hpdict['issouth'] = np.zeros(npix,dtype=bool)
    tmp               = (hpdict['bgsfracarea']>0) & ((hpdict['galb']<0) | ((hpdict['galb']>0) & (hpdict['dec']<dec_resol_ns)))
    hpdict['issouth'][tmp] = True
    hpdict['isnorth'] = np.zeros(npix,dtype=bool)
    tmp               = (hpdict['bgsfracarea']>0) & (hpdict['dec']>dec_resol_ns) & (hpdict['galb']>0)
    hpdict['isnorth'][tmp] = True
    hpdict['isdes']   = raichoorlib.get_isdes(hpdict['ra'],hpdict['dec'])
    hpdict['isdecals'] = (hpdict['issouth']) & (~hpdict['isdes'])
    hpdict['istest'] = (hpdict['ra'] > 160.) & (hpdict['ra'] < 230.) & (hpdict['dec'] > -2.) & (hpdict['dec'] < 18.)
    print('regions DONE...')

    # areas
    hpdict['area_all']   = hpdict['bgsfracarea'].sum() * pixarea
    for reg in ['south','decals','des','north', 'test']:
        hpdict['bgsarea_'+reg]   = hpdict['bgsfracarea'][hpdict['is'+reg]].sum() * pixarea
    print('areas DONE...')
    
    #target densities
    #namesels = {'any':-1, 'bright':1, 'faint':0, 'wise':2}
    for foot in ['north','south']:
        
        data = cat
        
        if (foot=='north'): keep = (data['DEC']>dec_resol_ns) & (galb>0)
        if (foot=='south'): keep = (data['DEC']<dec_resol_ns) | (galb<0)        
        ##
        if maskcat is None: keep &= catindesi
        else: keep &= (maskcat) & (catindesi)
        
        if survey == 'main': bgstargetname = 'BGS_TARGET'
        elif survey == 'sv1': bgstargetname = 'SV1_BGS_TARGET'
        elif survey == 'bgs': bgstargetname = 'BGSBITS'

        for namesel, bitnum in zip(namesels.keys(), namesels.values()):
            print('computing for ', foot, '/', namesel)
            if (namesel=='any'):             sel = np.ones(len(data),dtype=bool)
            else:                            sel = ((data[bgstargetname] & 2**(bitnum)) != 0)
            
            ind,c = np.unique(hppix_cat[(sel) & (keep)],return_counts=True)
            hpdict[foot+'_n'+namesel]      = np.zeros(npix)
            hpdict[foot+'_n'+namesel][ind] = c
        print('target densities in %s DONE...' %(foot))
            
    # storing mean hpdens
    isdesi = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
    for namesel in namesels.keys():
        ## south + north density
        hpdens = (hpdict['south_n'+namesel] + hpdict['north_n'+namesel] ) / (pixarea * hpdict['bgsfracarea'])
        ## split per region
        for reg in ['all','des','decals','north', 'south', 'test']:
            if (reg=='all'):
                hpdict['meandens_'+namesel+'_'+reg] = np.nanmean(hpdens[isdesi])
            else:
                hpdict['meandens_'+namesel+'_'+reg] = np.nanmean(hpdens[(isdesi) & (hpdict['is'+reg])])
            print('meandens_'+namesel+'_'+reg+' = '+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+' /deg2')
            
    # storing total target density
    isdesi = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
    for namesel in namesels.keys():
        ## split per region
        for reg in ['all','des','decals','north', 'south', 'test']:
            if (reg=='all'):
                hpdict['dens_'+namesel+'_'+reg] = (hpdict['south_n'+namesel][isdesi] + hpdict['north_n'+namesel][isdesi]).sum() / (pixarea * hpdict['bgsfracarea'][isdesi].sum())
            else:
                hpdict['dens_'+namesel+'_'+reg] = (hpdict['south_n'+namesel][(isdesi) & (hpdict['is'+reg])] + hpdict['north_n'+namesel][(isdesi) & (hpdict['is'+reg])]).sum() / (pixarea * hpdict['bgsfracarea'][(isdesi) & (hpdict['is'+reg])].sum())
            print('dens_'+namesel+'_'+reg+' = '+'%.0f'%hpdict['dens_'+namesel+'_'+reg]+' /deg2')
        
    #end = raichoorlib.get_date()
    #print('start:', start)
    #print('end:', end)
    
    return hpdict

# is in desi nominal footprint? (using tile radius of 1.6 degree)
# small test shows that it broadly works to cut on desi footprint 
def get_isdesi(ra,dec, nest=True):
    radius   = 1.6 # degree
    tmpnside = 16
    tmpnpix  = hp.nside2npix(tmpnside)
    # first reading desi tiles, inside desi footprint (~14k deg2)
    hdu  = fits.open('/global/cscratch1/sd/raichoor/desi-tiles-viewer.fits')
    data = hdu[1].data
    keep = (data['in_desi']==1)
    data = data[keep]
    tra,tdec = data['ra'],data['dec']
    # get hppix inside desi tiles
    theta,phi  = hp.pix2ang(tmpnside,np.arange(tmpnpix),nest=nest)
    hpra,hpdec = 180./np.pi*phi,90.-180./np.pi*theta
    hpindesi   = np.zeros(tmpnpix,dtype=bool)
    _,ind,_,_,_= raichoorlib.search_around(tra,tdec,hpra,hpdec,search_radius=1.6*3600)
    hpindesi[np.unique(ind)] = True
    ## small hack to recover few rejected pixels inside desi. Avoid holes if any
    tmp  = np.array([i for i in range(tmpnpix) 
                     if hpindesi[hp.get_all_neighbours(tmpnside,i,nest=nest)].sum()==8])
    hpindesi[tmp] = True
    ##
    pixkeep    = np.where(hpindesi)[0]
    # now compute the hppix for the tested positions
    pix  = hp.ang2pix(tmpnside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
    keep = np.in1d(pix,pixkeep)
    return keep

def get_isbgstile(ra,dec):
    radius   = 1.6 # degree
    tmpnside = 256
    tmpnpix  = hp.nside2npix(tmpnside)
    # first reading desi tiles, inside desi footprint (~14k deg2)
    hdu  = fits.open('/global/cscratch1/sd/qmxp55/BGS_SV_30_3x_superset60_JUL2019.fits')
    data = hdu[1].data
    #keep = (data['in_desi']==1)
    #data = data[keep]
    tra,tdec = data['RA'],data['DEC']
    # get hppix inside desi tiles
    theta,phi  = hp.pix2ang(tmpnside,np.arange(tmpnpix),nest=nest)
    hpra,hpdec = 180./np.pi*phi,90.-180./np.pi*theta
    hpindesi   = np.zeros(tmpnpix,dtype=bool)
    
    idx,ind,_,_,_= raichoorlib.search_around(tra,tdec,hpra,hpdec,search_radius=1.6*3600)
    
    hpindesi[np.unique(ind)] = True
    ## small hack to recover few rejected pixels inside desi. Avoid holes if any
    #tmp  = np.array([i for i in range(tmpnpix) 
    #                 if hpindesi[hp.get_all_neighbours(tmpnside,i,nest=nest)].sum()==8])
    #hpindesi[tmp] = True
    ##
    pixkeep    = np.where(hpindesi)[0]
    # now compute the hppix for the tested positions
    pix  = hp.ang2pix(tmpnside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
    keep = np.in1d(pix,pixkeep)
    
    hptileid = np.ones(tmpnpix)*float('NaN')
    tileid = np.ones_like(pix)*float('NaN')
    for i in range(len(data)):
        mask = ind[idx == i]
        #print(i, len(mask), len(np.unique(mask)))
        hptileid[np.unique(mask)] = data['CENTERID'][i] #hp tile center id
        mask2 = np.in1d(pix,mask)
        tileid[np.where(mask2)] = data['CENTERID'][i] #cat tile center id
    
    return keep, tileid


# mollweide plot setting
# http://balbuceosastropy.blogspot.com/2013/09/the-mollweide-projection.html
def set_mwd(ax,org=0):
    # org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    ax.set_xticklabels(tick_labels)     # we add the scale on the x axis
    ax.set_xlabel('R.A [deg]', size=18)
    ax.xaxis.label.set_fontsize(15)
    ax.set_ylabel('Dec. [deg]', size=18)
    ax.yaxis.label.set_fontsize(15)
    ax.grid(True)
    return True

# convert radec for mollwide
def get_radec_mw(ra,dec,org):
    ra          = np.remainder(ra+360-org,360) # shift ra values
    ra[ra>180] -= 360    # scale conversion to [-180, 180]
    ra          =- ra    # reverse the scale: East to the left
    return np.radians(ra),np.radians(dec)

# plot/xlim settings
def get_systplot(systquant):
    tmparray = np.array([
        'stardens',      [2.4,3.7],  r'log10(Stellar density from Gaia/dr2 [deg$^{-2}$])',
        'ebv',           [0.01,0.11],'Galactic extinction ebv [mag]',
        'psfsize_g',     [1,2.6],  'g-band psfsize [arcsec]',
        'psfsize_r',     [1,2.6],  'r-band psfsize [arcsec]',
        'psfsize_z',     [1,2.6],  'z-band psfsize [arcsec]',
        'galdepth_g',    [23.3,25.5],'g-band 5sig. galdepth [mag]',
        'galdepth_r',    [23.1,25],'r-band 5sig. galdepth [mag]',
        'galdepth_z',    [21.6,23.9],'z-band 5sig. galdepth [mag]',
        'nobs_g',    [3,4],'g-band NOBS',
        'nobs_r',    [3,4],'r-band NOBS',
        'nobs_z',    [3,4],'z-band NOBS',],
        
        
        dtype='object')
    tmparray = tmparray.reshape(int(tmparray.shape[0]/3),3)
    tmpind   = np.where(tmparray[:,0]==systquant.lower())[0][0]
    return tmparray[tmpind,1], tmparray[tmpind,2]

#
def plot_sysdens(hpdicttmp, namesel, regs, syst, mainreg, xlim=None, n=0, nx=20, clip=False, denslims=False, ylab=True, text=None, weights=False, nside=256, fig=None, gs=None, label=False):
    
    pixarea = hp.nside2pixarea(nside,degrees=True)

    hpdens = (hpdicttmp['south_n'+namesel] + hpdicttmp['north_n'+namesel] ) / (pixarea * hpdicttmp['bgsfracarea'])
    # plot/xlim settings
    if xlim is None:
        xlim, xlabel = get_systplot(syst)
    else:
        _, xlabel = get_systplot(syst)
        
    #do we have inf or nans within syst boundaries
    tmpsyst0 = hpdicttmp[syst]
    mask = (tmpsyst0>xlim[0]) & (tmpsyst0<xlim[1])
    tmpsyst = tmpsyst0[mainreg & mask]
    tmpdens   = hpdens[mainreg & mask]
    #print('%i infs and nans found within %s boundaries (%g, %g)' %(np.sum(~np.isfinite(tmpsyst)), syst, xlim[0], xlim[1]))
    #print('%i infs and nans found in target dens. within %s boundaries (%g, %g)' %(np.sum(~np.isfinite(tmpdens)), syst, xlim[0], xlim[1]))
    
    #xlim = tmpsyst[tmpsyst > 0].min(), tmpsyst[tmpsyst > 0].max()
    if clip: xlim = np.percentile(tmpsyst[tmpsyst>0],(1,99))
    xwidth = (xlim[1]-xlim[0])/nx
        
    # initializing plots
    ax = fig.add_subplot(gs[n])
    ## systematics
    ax.plot(xlim,[1.,1.],color='k',linestyle=':')
    ax.set_xlim(xlim)
    ax.set_ylim(0.8,1.2)
    
    delta = (xlim[1] - xlim[0])/15.
    if text is not None: ax.text(xlim[0]+delta, 1.15, text, fontsize=15)
        
    if ylab: ax.set_ylabel(r'n / $\overline{n}$',fontsize=20)
    ax.set_xlabel(xlabel,fontsize=12)
    ax.grid(True)
    #title = []
    #if clip: title.append('clipped')
    #if denslims: title.append('denslims')
    #ax.set_title(r'%s (%s)' %(namesel, ' & '.join(title)))
    ax.set_title(r'%s' %(namesel))
        
    ## histogram
    axh = ax.twinx()
    axh.set_xlim(xlim)
    axh.set_ylim(0,8)
    axh.axes.get_yaxis().set_ticks([])
    
    ## systematics
    cols = ['0.5','b','g','r']
    #regs = ['all','des','decals','north']
    densmin,densmax = 0,2
    for reg,col in zip(regs,cols):
        if (reg=='all'):
            isreg    = (mainreg)
            lw,alpha = 3,0.5
        else:
            isreg    = (mainreg) & (hpdicttmp['is'+reg])
            lw,alpha = 1,1.0
        tmpsyst   = hpdicttmp[syst][isreg]
        #xlim      = tmpsyst[(tmpsyst>0) & (np.isfinite(tmpsyst))].min(), tmpsyst[(tmpsyst>0) & (np.isfinite(tmpsyst))].max()
        #xlim, _ = get_systplot(syst)
        if clip: xlim = np.percentile(tmpsyst[(tmpsyst>0) & (np.isfinite(tmpsyst))],(1,99))
        tmpdens   = hpdens[isreg]
        
        if denslims:
            tmp = ((tmpdens/hpdicttmp['meandens_'+namesel+'_'+reg]>densmin) & 
                     (tmpdens/hpdicttmp['meandens_'+namesel+'_'+reg]<densmax) & 
                     (tmpsyst>xlim[0]) & 
                     (tmpsyst<xlim[1]))
        else:
            tmp       = (tmpsyst>xlim[0]) & (tmpsyst<xlim[1])
            
        systquant = tmpsyst[tmp] #systematics per region
        systdens  = tmpdens[tmp] #target density per region per bit
        
        #systdens /= hpdicttmp['meandens_'+namesel+'_'+reg] #density/mean density per SessionsSessionsbit per region
        systdens /= hpdicttmp['meandens_'+namesel+'_'+'all'] #density/mean density per bit overall desi footprint
        
        # get eta / eta_mean in nx bins
        plotxgrid, systv, systverr, xgrid = pixcorr(x=systquant, y=systdens, nx=nx, xlim=xlim)
        
        if label: lab = reg
        else: lab = None
            
        if len(regs) < 2: newcol, lw, alpha = 'k', 1, 1.0
        else: newcol = col
        
        if weights:
            b0, m0 = findlinmb(plotxgrid, systv, systverr)
            ws = 1./(m0*systquant+b0)
            plotxgrid_w, systv_w, systverr_w, xgrid_w = pixcorr(x=systquant, y=systdens*ws, nx=nx, xlim=xlim)
            
            if label: labw = reg+'_weighted'
            else: labw = None
            
            ax.errorbar(plotxgrid, systv, systverr, color=newcol, ecolor=newcol, zorder=1, lw=2*lw, alpha=alpha, label=lab)
            ax.errorbar(plotxgrid_w, systv_w, systverr_w, color=newcol, ecolor=newcol, zorder=1, lw=2*lw, ls='--',alpha=alpha, label=labw)
            
            ax.text(plotxgrid[2], 1.18, r'b = %2.3g' %(b0))
            ax.text(plotxgrid[2], 1.16, r'm = %2.3g' %(m0))
        else: 
            ax.errorbar(plotxgrid,systv,systverr,color=newcol,ecolor=newcol,zorder=1,lw=2*lw,alpha=alpha, label=lab)
            
        # histogram
        height,_ = np.histogram(systquant,bins=xgrid)
        height   = height.astype(float) / 1.e4
        xcent    = 0.5*(xgrid[1:]+xgrid[:-1])
        if (reg=='all') or (len(regs) < 2):
            axh.bar(xcent,height,align='center',width=xwidth,alpha=0.3,color=newcol)
        elif (len(regs) > 1): axh.step(xcent,height,where='mid',alpha=alpha,lw=lw,color=newcol)
        
        if label: ax.legend()
        
        if reg == 'all': x,yall = plotxgrid,systv
        elif reg == 'north': ynorth = systv
        elif reg == 'decals': ydecals = systv
        elif reg == 'des': ydes = systv
    #return x, yall, ynorth, ydecals, ydes
    
def pixcorr(x=None, y=None, nx=20, xlim=None):
    
    xgrid = xlim[0]+np.arange(nx+1)/float(nx)*(xlim[1]-xlim[0])
    plotxgrid    = (xgrid[0:-1]+xgrid[1:])/2.
    systnobj     = np.ones(nx)*float('NaN')
    systv        = np.ones(nx)*float('NaN')
    systverr     = np.ones(nx)*float('NaN')
    for j in range(nx):
        tmp      = np.where((x >= xgrid[j]) & (x < xgrid[j+1]))[0]
        systnobj[j]= len(tmp)
        if (len(tmp) > 0):
            systv[j]   = np.mean(y[tmp])
            systverr[j]= np.std(y[tmp])/np.sqrt(len(y[tmp]))
    
    return plotxgrid, systv, systverr, xgrid
    
def findlinmb(x, y, yerr):
    #finds linear fit parameters
    lf = linfit(x,y,yerr)
    inl = np.array([1.,0])
    b0,m0 = optimize.fmin(lf.chilin,inl, disp=False)
    return b0,m0

class linfit:
    def __init__(self,xl,yl,el):
        self.xl = xl
        self.yl = yl
        self.el = el
              
    def chilin(self,bml):
        chi = 0
        b = bml[0]
        m = bml[1]
        for i in range(0,len(self.xl)):
            y = b+m*self.xl[i]
            chi += (self.yl[i]-y)**2./self.el[i]**2.
        return chi
        
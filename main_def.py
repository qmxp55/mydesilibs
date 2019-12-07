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
from photometric_def import get_stars, get_galaxies, masking, results
from scipy import optimize
import pygraphviz as pgv
from PIL import Image

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

def get_random(N=3, sweepsize=None, dr='dr8', dirpath='/global/cscratch1/sd/qmxp55/'):
    
    import time
    start = time.time()
        
    if (N < 2):
        log.warning('Number of RANDOMS files must be greater than one')
        raise ValueError
    
    import glob
    #ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.29.0/' #issues with MASKBITS...
    
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


def get_sweep_whole(dr='dr8-south', rlimit=None, maskbitsource=False, bgsbits=False, opt='1', sweepdir='/global/cscratch1/sd/qmxp55/sweep_files/'):
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
                     ranindesi=None, catindesi=None, dec_resol_ns=32.375, namesels=None, galb=None, survey='main', 
                         desifootprint=True, target_outputs=True, log=False, tiledir='/global/cscratch1/sd/raichoor/'):

   # start = raichoorlib.get_date()
    # creating dictionary
    hpdict = {}
    
    if (nside is None) or (npix is None) or (nest is None) or (pixarea is None) & (pixmapfile is not None):
        
        hdr          = fits.getheader(pixmapfile,1)
        nside,nest   = hdr['hpxnside'],hdr['hpxnest']
        npix         = hp.nside2npix(nside)
        pixarea      = hp.nside2pixarea(nside,degrees=True)
    else: raise ValueErro('if not pixel information given, include pixmapfile to compute them.')
    
    if (getnobs) and (randoms is None):
        raise ValueError('random catalogue can not be None when getnobs is set True.')
        
    if (hppix_ran is None) and (randoms is not None):
        hppix_ran = hp.ang2pix(nside,(90.-np.array(randoms['DEC']))*np.pi/180.,np.array(randoms['RA'])*np.pi/180.,nest=nest)
    elif (hppix_ran is None) and (randoms is None):
        raise ValueError('include a random catalogue to compute their hp pixels indexes.')
        
    if (ranindesi is None) and (randoms is not None):
        ranindesi = get_isdesi(randoms['RA'],randoms['DEC'], tiledir=tiledir) # True if is in desi footprint
    elif (ranindesi is None) and (randoms is None):
        raise ValueError('include a random catalogue to compute ranindesi.')
        
        
    theta,phi  = hp.pix2ang(nside,np.arange(npix),nest=nest)
    hpdict['ra'],hpdict['dec'] = 180./np.pi*phi,90.-180./np.pi*theta
    c = SkyCoord(hpdict['ra']*units.degree,hpdict['dec']*units.degree, frame='icrs')
    hpdict['gall'],hpdict['galb'] = c.galactic.l.value,c.galactic.b.value

    # is in desi tile?
    hpdict['isdesi'] = get_isdesi(hpdict['ra'],hpdict['dec'], tiledir=tiledir)
    if log: print('positions and desifotprint DONE...')

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
    if log: print('systematics DONE...')
        
    # computing fracareas
    randdens = 5000*Nranfiles
    if log: print('randdens = ', randdens, ' ; len randoms = ', len(hppix_ran))
    if desifootprint: mainfootprint = ranindesi
    else: mainfootprint = np.ones_like(hppix_ran, dtype='?')
    if maskrand is None:
        ind,c           = np.unique(hppix_ran[mainfootprint],return_counts=True)
    else:
        ind,c           = np.unique(hppix_ran[(maskrand) & (mainfootprint)],return_counts=True)
    hpdict['bgsfracarea']      = np.zeros(npix)
    hpdict['bgsfracarea'][ind] = c / randdens / pixarea
    if log: print('bgsfracarea DONE...')
    
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
        if log: print('nobs DONE...')
        
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
    if log: print('regions DONE...')

    # areas
    hpdict['area_all']   = hpdict['bgsfracarea'].sum() * pixarea
    for reg in ['south','decals','des','north', 'test']:
        hpdict['bgsarea_'+reg]   = hpdict['bgsfracarea'][hpdict['is'+reg]].sum() * pixarea
    if log: print('areas DONE...')
    
    
    #target densities
    if target_outputs:
        
        if (cat is None) or (pixmapfile is None) or (namesels is None) or (Nranfiles is None):
            raise ValueError('cat, pixmapfile, namesels and Nranfiles can not be None.')
        
        if (hppix_cat is None):
            hppix_cat = hp.ang2pix(nside,(90.-cat['DEC'])*np.pi/180.,cat['RA']*np.pi/180.,nest=nest) # catalogue hp pixels array
        
        if (catindesi is None):
            catindesi = get_isdesi(cat['RA'],cat['DEC'], tiledir=tiledir) # True is is in desi footprint
        
        if galb is None:
            c = SkyCoord(cat['RA']*units.degree,cat['DEC']*units.degree, frame='icrs')
            galb = c.galactic.b.value # galb coordinate
        
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
                if log: print('computing for ', foot, '/', namesel)
                if (namesel=='any'):             sel = np.ones(len(data),dtype=bool)
                else:                            sel = ((data[bgstargetname] & 2**(bitnum)) != 0)
            
                ind,c = np.unique(hppix_cat[(sel) & (keep)],return_counts=True)
                hpdict[foot+'_n'+namesel]      = np.zeros(npix)
                hpdict[foot+'_n'+namesel][ind] = c
            if log: print('target densities in %s DONE...' %(foot))
            
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
                if log: print('meandens_'+namesel+'_'+reg+' = '+'%.0f'%hpdict['meandens_'+namesel+'_'+reg]+' /deg2')
            
        # storing total target density
        isdesi = (hpdict['isdesi']) & (hpdict['bgsfracarea']>0)
        for namesel in namesels.keys():
            ## split per region
            for reg in ['all','des','decals','north', 'south', 'test']:
                if (reg=='all'):
                    hpdict['dens_'+namesel+'_'+reg] = (hpdict['south_n'+namesel][isdesi] + hpdict['north_n'+namesel][isdesi]).sum() / (pixarea * hpdict['bgsfracarea'][isdesi].sum())
                else:
                    hpdict['dens_'+namesel+'_'+reg] = (hpdict['south_n'+namesel][(isdesi) & (hpdict['is'+reg])] + hpdict['north_n'+namesel][(isdesi) & (hpdict['is'+reg])]).sum() / (pixarea * hpdict['bgsfracarea'][(isdesi) & (hpdict['is'+reg])].sum())
                if log: print('dens_'+namesel+'_'+reg+' = '+'%.0f'%hpdict['dens_'+namesel+'_'+reg]+' /deg2')
    
    return hpdict

# is in desi nominal footprint? (using tile radius of 1.6 degree)
# small test shows that it broadly works to cut on desi footprint 
def get_isdesi(ra,dec, nest=True, tiledir='/global/cscratch1/sd/raichoor/'):
    radius   = 1.6 # degree
    tmpnside = 16
    tmpnpix  = hp.nside2npix(tmpnside)
    # first reading desi tiles, inside desi footprint (~14k deg2)
    hdu  = fits.open(tiledir+'desi-tiles-viewer.fits')
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
        
        #systdens /= hpdicttmp['meandens_'+namesel+'_'+reg] #density/mean density per bit per region
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
        

def get_reg(reg='decals', hppix=None):
    ''' get an specific LS region i.e., DECaLS/DES/NORTH from catalogue with hppix index info.'''
    
    hpdict = np.load('/global/cscratch1/sd/qmxp55/hppixels_regions.npy')
    isreg_pixlist = hpdict['hpxpixel'][hpdict['is'+reg]]
    regcat = np.in1d(hppix, isreg_pixlist)
    
    return regcat

def getStats(cat=None, hpdict=None, bgsmask=None, rancuts=None, CurrentMask=None, PrevMask=None, 
                 reg='decals', regcat=None, regran=None):
    
    #from astropy.table import Table
    Tab = []
    GMT = np.zeros_like(regcat, dtype='?')
    GMT_ran = np.zeros_like(regran, dtype='?')
    PMT = GMT.copy()
    PMT_ran = GMT_ran.copy()
        
    #if (regcat is not None) & (regran is not None): print('region set...')
    #elif (regcat is None) & (regran is None): regcat, regran = ~GMT.copy(), ~GMT_ran.copy()
    #else: raise ValueError('regcat and regran both have to be None-type or non None-type.')
     
    # area in region
    Areg = hpdict['bgsarea_'+reg]
    # Number of randoms in region
    NRreg = np.sum(regran)
    #
    B, F = cat['RMAG'] < 19.5, np.logical_and(cat['RMAG'] < 20, cat['RMAG'] > 19.5)
        
    if PrevMask is not None:
        PM_lab = '|'.join(PrevMask)
        for i in PrevMask:
            PMT |= (cat['BGSBITS'] & 2**(bgsmask[i])) == 0
            if i in rancuts.keys(): PMT_ran |= ~rancuts[i]   
    else:
        PM_lab = 'None'
    
    for i in CurrentMask:
        
        if i in rancuts.keys(): A_i = (np.sum((~rancuts[i] & (~PMT_ran) & (regran)))/NRreg)*(Areg)
        else: A_i = 0.
        bgscut = (cat['BGSBITS'] & 2**(bgsmask[i])) == 0
        #eta_B_i_in = np.sum((GeoCutsDict[i]) & (B) & (~PMT))/(A_i) #density over the geometric area
        #eta_F_i_in = np.sum((GeoCutsDict[i]) & (F) & (~PMT))/(A_i) #density over the geometric area
        eta_B_i = np.sum((bgscut) & (B) & (~PMT) & (regcat))/(Areg) #density over the total area
        eta_F_i = np.sum((bgscut) & (F) & (~PMT) & (regcat))/(Areg) #density over the total area
        
        Tab.append([i, round(A_i*(100/Areg), 2), round(eta_B_i,2), round(eta_F_i,2)])
            
        GMT |= bgscut
        if i in rancuts.keys(): GMT_ran |= ~rancuts[i]  
    
    lab = '|'.join(CurrentMask)
    lab_in = '(%s)' %(lab)
    lab_out = '~(%s)*' %(lab)
    lab_out2 = '~(%s)' %(lab)
    
    A_GMT_in = (np.sum((GMT_ran) & (~PMT_ran) & (regran))/NRreg)*(Areg)
    eta_B_GMT_in_1 = np.sum((GMT) & (B) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
    eta_F_GMT_in_1 = np.sum((GMT) & (F) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area

    A_GMT_out = (np.sum((~GMT_ran) & (~PMT_ran) & (regran))/NRreg)*(Areg)
    eta_B_GMT_out_1 = np.sum((~GMT) & (B) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
    eta_B_GMT_out_2 = np.sum((~GMT) & (B) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
    eta_F_GMT_out_1 = np.sum((~GMT) & (F) & (~PMT) & (regcat))/(Areg) #Not corrected for mask area
    eta_F_GMT_out_2 = np.sum((~GMT) & (F) & (~PMT) & (regcat))/(A_GMT_out) #Corrected for mask area
    
    if len(CurrentMask) > 1:
        Tab.append([lab_in, round(A_GMT_in*(100/Areg),2), round(eta_B_GMT_in_1,2), round(eta_F_GMT_in_1,2)])
    Tab.append([lab_out, round(A_GMT_out*(100/Areg),2), round(eta_B_GMT_out_1,2), round(eta_F_GMT_out_1,2)])
    Tab.append([lab_out2, round(A_GMT_out*(100/Areg),2), round(eta_B_GMT_out_2,2), round(eta_F_GMT_out_2,2)])
    
    Tab = np.transpose(Tab)
    t = Table([Tab[0], Tab[1], Tab[2], Tab[3]], 
              names=('GM','$f_{A}$ [$\%$]', '$\eta_{B}$ [deg$^2$]', '$\eta_{F}$ [deg$^2$]'),
                    dtype=('S', 'f8', 'f8', 'f8'))
    
    print('Previous Cuts: (%s)' %(PM_lab))
    print('Current Cuts: %s' %(lab_in))
                                    
    return t


def flow(cat=None, hpdict=None, bgsmask=None, rancuts=None, order=None, reg=None, 
             regcat=None, regran=None, file=None):
    
    # add GRAPHVIZ bin files to PATH, otherwise it doesn't find them
    os.environ['PATH'] = '/global/u2/q/qmxp55/bin'
    
    if order is None:
        raise ValueError('define the order of flow chart.')
    
    T = Table()
    Areg = hpdict['bgsarea_'+reg]
    
    B, F = cat['RMAG'] < 19.5, np.logical_and(cat['RMAG'] < 20, cat['RMAG'] > 19.5)
    den0B = np.sum((regcat) & (B))/Areg
    den0F = np.sum((regcat) & (F))/Areg
    
    #T['SU'] = masking(title='START', submasks=None, details=None)
    #T['SG'] = masking(title='GEOMETRICAL', submasks=None, details=None)
    T['I'] = masking(title='%s (%s)' %('LS DR8',reg.upper()), submasks=['rmag < %2.2g' %(20)], details=None)
    T['RI'] = results(a=Areg, b=den0B, f=den0F, stage='ini', per=False)
    
    G=pgv.AGraph(strict=False,directed=True)

    elist = []
    rejLab = []
    #define initial params in flow chart
    #ini = ['SU', 'I', 'RI', 'SG']
    ini = ['I', 'RI']
    for i in range(len(ini) - 1):
        elist.append((list(T[ini[i]]),list(T[ini[i+1]])))
        
    #G.add_edges_from(elist)
    #stages=['SU', 'SG']
    #G.add_nodes_from([list(T[i]) for i in stages], color='green', style='filled')
    nlist=['RI']
    G.add_nodes_from([list(T[i]) for i in nlist], color='lightskyblue', shape='box', style='filled')
    maskings=['I']
    G.add_nodes_from([list(T[i]) for i in maskings], color='lawngreen', style='filled')
        
    #
    for num, sel in enumerate(order):
        
        T['I'+str(num)] = masking(title=' & '.join(sel), submasks=None, details=None)
        
        if num == 0: elist.append((list(T['I']),list(T['I'+str(num)])))
        else: elist.append((list(T['R'+str(num-1)]),list(T['I'+str(num)])))
            
        if num == 0: pm = None
        elif num == 1: pm = order[0]
        else: pm += order[num-1]
            
        if len(sel) > 1: IGMLab_2 = ' | '.join(sel)
        else: IGMLab_2 = sel[0]
        
        t = getStats(cat=cat, hpdict=hpdict, bgsmask=bgsmask, rancuts=rancuts, CurrentMask=sel, PrevMask=pm, 
                 reg=reg, regcat=regcat, regran=regran)
        
        T['R'+str(num)] = results(a=t[-2][1], b=t[-2][2], f=t[-2][3], b2=t[-1][2], f2=t[-1][3], stage='geo', per=True)
        T['REJ'+str(num)] = results(a=t[-3][1], b=t[-3][2], f=t[-3][3], stage='ini', per=True, title='(%s)' %(IGMLab_2))
        
        elist.append((list(T['I'+str(num)]),list(T['REJ'+str(num)])))
        elist.append((list(T['I'+str(num)]),list(T['R'+str(num)])))
        
        if False in [i in rancuts.keys() for i in sel]: icolor = 'plum'
        else: icolor = 'lightgray'
        
        Rlist=['R'+str(num)]
        G.add_nodes_from([list(T[i]) for i in Rlist], color='lightskyblue', shape='box', style='filled')
        REJlist=['REJ'+str(num)]
        G.add_nodes_from([list(T[i]) for i in REJlist], color='lightcoral', shape='box', style='filled')
        Ilist=['I'+str(num)]
        G.add_nodes_from([list(T[i]) for i in Ilist], color=icolor, style='filled')

        if len(sel) > 1:
            for i, j in enumerate(sel):
                T['REJ'+str(num)+str(i)] = results(a=t[i][1], b=t[i][2], f=t[i][3], stage='ini', per=True, title=j)
                elist.append((list(T['REJ'+str(num)]),list(T['REJ'+str(num)+str(i)])))
                
                REJilist=['REJ'+str(num)+str(i)]
                G.add_nodes_from([list(T[i]) for i in REJilist], color='coral', shape='box', style='filled')
        
    #
    if file is None:
        pathdir = os.getcwd()+'/'+'results'+'_'+reg
        if not os.path.isdir(pathdir): os.makedirs(pathdir)
        file = pathdir+'/'+'flow'
        
        
    G.add_edges_from(elist)
    G.write('%s.dot' %(file)) # write to simple.dot
    BB=pgv.AGraph('%s.dot' %(file)) # create a new graph from file
    BB.layout(prog='dot') # layout with default (neato)
    BB.draw('%s.png' %(file)) # draw png
    #os.system('convert ' + file + '.ps ' + file + '.png')
    flow = Image.open('%s.png' %(file))

    return flow, elist, T


import matplotlib.gridspec as gridspec
from geometric_def import circular_mask_radii_func

def overdensity(cat, star, radii_1, nameMag, slitw, density=False, magbins=(8,14,4), radii_2=None, 
                grid=None, SR=[2, 240.], scaling=False, nbins=101, SR_scaling=4, logDenRat=[-3, 3], 
                    radii_bestfit=True, annulus=None, bintype='2', filename=None):
    '''
    Get scatter and density plots of objects of cat1 around objects of cat2 within a search radius in arcsec.

    Inputs
    ------
    cat: (array) catalogue 1;
    star: (array) catalogue 2;
    nameMag: (string) label of magnitude in catalogue 2;
    slitw: (float, integer) slit widht;
    density: (boolean) True to get the density as function of distance (arcsec) within shells;
    magbins: (integers) format to separate the magnitude bins in cat2 (min, max, number bins);

    Output
    ------
    (distance (arcsec), density) if density=True
    '''
    
    # define the slit width for estimating the overdensity off diffraction spikes
    slit_width = slitw
    search_radius = SR[1]

    # Paramater for estimating the overdensities
    annulus_min = SR[0]
    annulus_max = SR[1]

    ra2 = star['RA']
    dec2 = star['DEC']
    
    ra1 = cat['RA']
    dec1 = cat['DEC']

    if density:

        idx2, idx1, d2d, d_ra, d_dec = search_around(ra2, dec2, ra1, dec1,
                                                 search_radius=search_radius)
        density = []
        shells = np.linspace(1, search_radius, search_radius)
        for i in range(len(shells)-1):

            ntot_annulus = np.sum((d2d>shells[i]) & (d2d<shells[i+1]))
            density_annulus = ntot_annulus/(np.pi*(shells[i+1]**2 - shells[i]**2))
            bincenter = (shells[i]+shells[i+1])/2

            density.append([bincenter, density_annulus])

        density = np.array(density).transpose()
        plt.figure(figsize=(12, 8))
        plt.semilogy(density[0], density[1])
        plt.xlabel(r'r(arcsec)')
        plt.ylabel(r'N/($\pi r^2$)')
        plt.grid()
        plt.show()

        return density


    if bintype == '2':
        mag_bins = np.linspace(magbins[0], magbins[1], magbins[2]+1)
        mag_bins_len = len(mag_bins)-1
    elif bintype == '1':
        mag_bins = np.linspace(magbins[0], magbins[1], magbins[2])
        mag_bins_len = len(mag_bins)
    elif bintype == '0':
        mag_bins = np.array(magbins)
        mag_bins_len = len(mag_bins)-1
    else:
        raise ValueError('Invaid bintype. Choose bintype = 0, 1, 2')
    
    
    if grid is not None:
        rows, cols = grid[0], grid[1]
    else:
        rows, cols = len(mag_bins), 1
    figsize = (8*cols, 8*rows)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.3, hspace=0.2)
    fig = plt.figure(num=1, figsize=figsize)
    ax = []
        
    for index in range(mag_bins_len):
        if (bintype == '2') or (bintype == '0'):
            mask_star = (star[nameMag]>mag_bins[index]) & (star[nameMag]<mag_bins[index+1])
            title = '{:.2f} < {} < {:.2f}'.format(mag_bins[index], nameMag, mag_bins[index+1], np.sum(mask_star))    
        elif bintype == '1':
            if index==0:
                mask_star = (star[nameMag]<mag_bins[index])
                title = '{} < {:.2f}'.format(nameMag,mag_bins[0], np.sum(mask_star))
            else:
                mask_star = (star[nameMag]>mag_bins[index-1]) & (star[nameMag]<mag_bins[index])
                title = '{:.2f} < {} < {:.2f}'.format(mag_bins[index-1], nameMag, mag_bins[index], np.sum(mask_star))
        else:
            raise ValueError('Invaid bintype. Choose bintype = 0, 1, 2')

        print(title)
        magminrad = circular_mask_radii_func([mag_bins[index+1]], radii_1, bestfit=radii_bestfit)[0]
        magmaxrad = circular_mask_radii_func([mag_bins[index]], radii_1, bestfit=radii_bestfit)[0]

        if not scaling:
            #get the mask radii from the mean magnitude
            mag_mean = np.mean(star[nameMag][mask_star])
            print('mag_mean', mag_mean)
            mask_radius = circular_mask_radii_func([mag_mean], radii_1, bestfit=radii_bestfit)[0]
            if radii_2:
                mask_radius2 = circular_mask_radii_func([mag_mean], radii_2)[0]

        idx2, idx1, d2d, d_ra, d_dec = search_around(ra2[mask_star], dec2[mask_star], ra1, dec1,
                                                 search_radius=annulus_max)

        Nsources = len(ra2[mask_star])
        perc_sources = 100*len(ra2[mask_star])/len(ra2)
        
        #print('%d sources ~%g %% ' %(Nsources, perc_sources))
        
        mag_radii = circular_mask_radii_func(star[nameMag][mask_star][idx2], radii_1, bestfit=radii_bestfit)
        #print(len(d_ra), len(mag_radii))
        print('mag_radii MAX:',mag_radii.max(), 'mag_radii MIN:',mag_radii.min())
        print('mag MAX:',star[nameMag][mask_star][idx2].max(), 'mag MIN:',star[nameMag][mask_star][idx2].min())

        #markersize = np.max([0.01, np.min([10, 0.3*100000/len(idx2)])])
        #axis = [-search_radius*1.05, search_radius*1.05, -search_radius*1.05, search_radius*1.05]
        #axScatter = scatter_plot(d_ra, d_dec, markersize=markersize, alpha=0.4, figsize=6.5, axis=axis, title=title)
        
        row = (index // cols)
        col = index % cols
        ax.append(fig.add_subplot(gs[row, col]))
        
        if scaling:
            d2d_arcsec = d2d
            d_ra, d_dec, d2d = d_ra/mag_radii, d_dec/mag_radii, d2d_arcsec/mag_radii
            search_radius = SR_scaling #d2d.max() - d2d.max()*0.3
            #ntot_annulus = np.sum((d2d_arcsec>annulus_min) & (d2d<search_radius))
            ntot_annulus = np.sum(d2d<search_radius)
            #density_annulus = ntot_annulus/(np.pi*(search_radius**2 - d2d[d2d_arcsec > 2].min()**2))
            density_annulus = ntot_annulus/(np.pi*(search_radius**2))
            #print('ntot_annulus:', ntot_annulus, 'density_annulus:', density_annulus)
            print('d2d min=%2.3g, d2d max=%2.3g' %(d2d.min(), d2d.max()))
        else:
            d2d_arcsec = None
            ntot_annulus = np.sum((d2d>annulus_min) & (d2d<annulus_max))
            density_annulus = ntot_annulus/(np.pi*(annulus_max**2 - annulus_min**2))
        
        if annulus is not None:
            annMask = np.ones(len(cat), dtype='?')
            d_ra2 = np.zeros(len(cat))
            d_dec2 = np.zeros(len(cat))
            d_ra2[idx1] = d_ra
            d_dec2[idx1] = d_dec
            print(len(cat), len(d_ra2), len(d_dec2))
            #print(len(set(idx1)), len(set(idx2)))
            #print(idx1.max(), idx2.max())
            #angle_array = np.linspace(0, 2*np.pi, 240)
            annMask &= np.logical_and((d_ra2**2 + d_dec2**2) < annulus[1]**2, (d_ra2**2 + d_dec2**2) > annulus[0]**2)
            
            #annMask &= np.logical_and(d_dec < annulus[1] * np.cos(angle_array), d_dec > annulus[0] * np.cos(angle_array))
        
        if scaling:
            mask_radius = None
        
        bins, mesh_d2d, density_ratio = relative_density_plot(d_ra, d_dec, d2d, search_radius,
                        ref_density=density_annulus, return_res=True, show=False, nbins=nbins, 
                            ax=ax[-1], d2d_arcsec=d2d_arcsec, annulus_min=annulus_min, 
                                logDenRat=logDenRat, mask_radius=magmaxrad)
   
        if not scaling:
            angle_array = np.linspace(0, 2*np.pi, 240)
            for i in [magminrad, magmaxrad]:
                x = i * np.sin(angle_array)
                y = i * np.cos(angle_array)
                ax[-1].plot(x, y, 'k', lw=2)
            
            ax[-1].text(-annulus_max+annulus_max*0.02, annulus_max-annulus_max*0.05, '%d sources ~%2.3g %% ' %(Nsources, perc_sources), fontsize=8,color='k')
            ax[-1].text(-annulus_max+annulus_max*0.02, annulus_max-annulus_max*0.11, '%d objects ~%2.3g %% ' %(ntot_annulus, 100*ntot_annulus/len(ra1)), fontsize=8,color='k')
            #ax[-1].text(-annulus_max+annulus_max*0.02, annulus_max-annulus_max*0.17, '$\eta$=%2.3g arcsec$^{-2}$' %(density_annulus), fontsize=8,color='k')

            ax[-1].set_xlabel(r'$\Delta$RA (arcsec)')
            ax[-1].set_ylabel(r'$\Delta$DEC (arcsec)')
        
            if radii_2:
                x2 = mask_radius2 * np.sin(angle_array)
                y2 = mask_radius2 * np.cos(angle_array)
                ax[-1].plot(x2, y2, 'k', lw=1.5, linestyle='--')
        else:
            angle_array = np.linspace(0, 2*np.pi, 100)
            x = 1 * np.sin(angle_array)
            y = 1 * np.cos(angle_array)
            ax[-1].plot(x, y, 'k', lw=4)
            
            ax[-1].text(-SR_scaling+0.1, SR_scaling-0.2, '%d sources ~%2.3g %% ' %(Nsources, perc_sources), fontsize=10,color='k')
            ax[-1].text(-SR_scaling+0.1, -SR_scaling+0.1, '%d objects ~%2.3g %% ' %(ntot_annulus, 100*ntot_annulus/len(ra1)), fontsize=10,color='k')
            #ax[-1].text(-SR_scaling+0.1, SR_scaling-0.9, '$\eta$=%2.3g deg$^{-2}$' %(density_annulus), fontsize=8,color='k')

            ax[-1].set_xlabel(r'$\Delta$RA/radii$_{i}$')
            ax[-1].set_ylabel(r'$\Delta$DEC/radii$_{i}$')
            
        ax[-1].set_title(title)
        ax[-1].axvline(0, ls='--', c='k')
        ax[-1].axhline(0, ls='--', c='k')
        if annulus is not None:
            for i in annulus:
                x = i * np.sin(angle_array)
                y = i * np.cos(angle_array)
                ax[-1].plot(x, y, 'yellow', lw=3, ls='-')
                
    if filename is not None:
            fig.savefig(filename+'.png', bbox_inches = 'tight', pad_inches = 0)
    
    if annulus is not None:
        return d_ra2, d_dec2, annMask
        
        
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

def relative_density_plot(d_ra, d_dec, d2d, search_radius, ref_density, nbins=101, return_res=False, 
                          show=True, ax=plt, d2d_arcsec=None, annulus_min=2, logDenRat=[-3,3], mask_radius=None):

    bins = np.linspace(-search_radius, search_radius, nbins)
    bin_spacing = bins[1] - bins[0]
    bincenter = (bins[1:]+bins[:-1])/2
    mesh_ra, mesh_dec = np.meshgrid(bincenter, bincenter)
    mesh_d2d = np.sqrt(mesh_ra**2 + mesh_dec**2)
    if d2d_arcsec is not None:
        mask = (d2d_arcsec>annulus_min) #to avoid self match with stars
    else:
        mask = (d2d>annulus_min) #to avoid self match with stars
    #taking the 2d histogram and divide by the area of each bin to get the density
    density, _, _ = np.histogram2d(d_ra[mask], d_dec[mask], bins=bins)/(bin_spacing**2)
    #ignoring data outside the circle with radius='search radius'
    #print('Nbins:',len(bins), 'binArea:', bin_spacing**2, 'Nobjects:', len(d_ra[mask]))
    #pix_density = len(d_ra[mask])/((len(bins)**2)*(bin_spacing**2))
    #print('tot_density_pix:', pix_density)
    
    #mean density at search radius
    if search_radius < 10:
        meanmask = np.logical_and(mesh_d2d <= search_radius, mesh_d2d > 1.2)
    else:
        meanmask = np.logical_and(mesh_d2d <= search_radius, mesh_d2d > 100.)
    ref_density = np.mean(density[meanmask])
    
    #density profile
    dist = np.linspace(0., search_radius, nbins)
    dist2 = np.linspace(0.008, search_radius, nbins/2.)
    dist_spacing = dist[1] - dist[0]
    dist_spacing2 = dist2[1] - dist2[0]
    dpx, dpy, dpx2, dpy2 = [], [], [], []
    for i, j in enumerate(dist):
        #for the cumulative radia profile
        dmask = mesh_d2d <= j
        drcumu = np.log2(np.mean(density[dmask]/ref_density))
        if drcumu is np.nan:
            dpy.append(-1)
        else:
            dpy.append(drcumu)
        dpx.append(j)
    for i, j in enumerate(dist2[:-1]):
        #for the no cumulative radia profile
        dmask2 = np.logical_and(mesh_d2d < dist2[i+1], mesh_d2d >= dist2[i])
        drnocumu = np.log2(np.mean(density[dmask2]/ref_density))
        #drnocumu = np.mean(density[dmask2]/ref_density) -1.
        if drnocumu is np.nan:
            dpy2.append(-1)
        else:
            dpy2.append(drnocumu)
        dpx2.append(dist2[i] + dist_spacing2/2.)
    
    if search_radius < 10:
        dpy = np.array(dpy)
        dpy2 = np.array(dpy2)
        dmax = dpy2[np.array(dpx2) > 1].max()
        dmin = dpy2[np.array(dpx2) > 1].min()
        maglimrad = 1
    else:
        dpy20 = np.array(dpy2)
        dpy = np.array(dpy)*search_radius/logDenRat[1]
        dpy2 = np.array(dpy2)*search_radius/logDenRat[1]
        dmax = dpy20[np.array(dpx2) > mask_radius].max()
        dmin = dpy20[np.array(dpx2) > mask_radius].min()
        maglimrad = mask_radius
    
    print('density cumu (min, max): (%2.3g, %2.3g)' %(np.array(dpy).min(), np.array(dpy).max()))
    print('density non-cumu (min, max): (%2.3g, %2.3g)' %(np.array(dpy2).min(), np.array(dpy2).max()))
    
    mask = mesh_d2d >= bins.max()-bin_spacing
    density[mask] = np.nan
    #density_ratio = density/ref_density
    density_ratio = np.log2(density/ref_density)
    
    idxinf = np.where(np.isinf(density_ratio))
    #print('inf values:',density_ratio[idxinf])
    print('%d of inf in density ratio out of a total of %d' %(len(density_ratio[idxinf]), len(density_ratio[~np.isnan(density_ratio)])))
    density_ratio[idxinf] = logDenRat[0]
    #print('inf values AFTER:',density_ratio[idxinf])
    
    den_rat = density_ratio[~np.isnan(density_ratio)]
    denmin = den_rat.min()
    denmax = den_rat.max()
    print('Minimum density ratio = %g, Maximum density ratio = %g' %(denmin, denmax))
    print('----------------')
    fig = plt.figure(1)
    #img = ax.imshow(density_ratio.transpose()-1, origin='lower', aspect='equal',
    img = ax.imshow(density_ratio.transpose(), origin='lower', aspect='equal',
               cmap='seismic', extent=bins.max()*np.array([-1, 1, -1, 1]), vmin=logDenRat[0], vmax=logDenRat[1])
    #ax.colorbar(fraction=0.046, pad=0.04)
    fig.colorbar(img, fraction=0.046, pad=0.04, label=r'$\log_{2}(\eta_{pix}/\eta_{tot})$')
    #ax.plot(np.array(dpx), dpy, lw=2.5, color='green')
    ax.plot(np.array(dpx2), dpy2, lw=2.5, color='red')
    
    # find the max, min of density ratio profile for distances > 1
    ax.text(1*search_radius/10., search_radius - 2*search_radius/30, '$max(\eta(\Delta r)/\eta, r>%i)=%2.3g$' %(maglimrad, 2**(dmax)), fontsize=10,color='k')
    #ax.text(4*search_radius/10., search_radius - 4*search_radius/30, '$min(\eta(\delta r)/\eta)=%2.3g$' %(2**(dmin)), fontsize=10,color='k')
    
    ax.set_ylim(-search_radius, search_radius)
    if show:
        ax.show()

    if return_res:
        return bins, mesh_d2d, density_ratio
    
def limits():
    
    limits = {}
    limits['Grr'] = (-3, 5)
    limits['g-z'] = (-1.8, 6)
    limits['r'] = (15, 20.1)
    limits['rfibmag'] = (16, 26)
    limits['g-r'] = (-0.5, 2.3)
    limits['r-z'] = (-0.7, 2.8)
    
    return limits

def hexbin(coord, catmask, n, C=None, bins=None, title=None, cmap='viridis', ylab=True, xlab=True, vline=None, 
           hline=None, fig=None, gs=None, xlim=None, ylim=None, vmin=None, vmax=None, mincnt=1, fmcline=False, 
               file=None, gridsize=(60,60), comp=False, fracs=False, area=None):
    
    x, y = coord.keys()
    
    ax = fig.add_subplot(gs[n])
    if title is not None: ax.set_title(r'%s' %(title), size=20)
    if xlim is None: xlim = limits()[x]
    if ylim is None: ylim = limits()[y]
    masklims = (coord[x] > xlim[0]) & (coord[x] < xlim[1]) & (coord[y] > ylim[0]) & (coord[y] < ylim[1])
    
    if catmask is None: keep = masklims
    else: keep = (catmask) & (masklims)
        
    Ntot = np.sum((catmask) & (masklims))
        
    if hline is not None:
        maskhigh = (masklims) & (coord[y] > hline) & (catmask)
        masklow = (masklims) & (coord[y] < hline) & (catmask)
        maskgal = (~masklow) & (catmask)
        
    pos = ax.hexbin(coord[x][keep], coord[y][keep], C=C, gridsize=gridsize, cmap=cmap, 
                    vmin=vmin, vmax=vmax, bins=bins, mincnt=mincnt, alpha=0.8)
    
    dx = np.abs(xlim[1] - xlim[0])/15.
    dy = np.abs(ylim[1] - ylim[0])/15.
    if comp: ax.text(xlim[0]+dx, ylim[1]-dy, r'comp. %2.3g %%' %(100 * np.sum(pos.get_array())/np.sum(keep)), size=15)
    if fracs: 
        ax.text(xlim[1]-5*dx, ylim[1]-dy, r'Ntot. %i' %(Ntot), size=15)
        if area is not None: ax.text(xlim[1]-5*dx, ylim[1]-2*dy, r'$\eta$. %.2f/deg$^2$' %(Ntot/area), size=15)
        ax.text(xlim[0]+dx, ylim[1]-dy, r'f.gal. %.2f %%' %(100 * np.sum(maskhigh)/Ntot), size=15)
        ax.text(xlim[0]+dx, ylim[0]+dy, r'f.stars. %.2f %%' %(100 * np.sum(masklow)/Ntot), size=15)
    if ylab: ax.set_ylabel(r'%s' %(y), size=20)
    if xlab: ax.set_xlabel(r'%s' %(x), size=20)
    if hline is not None: ax.axhline(hline, ls='--', lw=2, c='r')
    if vline is not None: ax.axvline(vline, ls='--', lw=2, c='r')
    if fmcline: 
        x_N1 = np.linspace(15.5, 17.1, 4)
        ax.plot(x_N1, 2.9 + 1.2 + x_N1, color='r', ls='--', lw=2)
        x_N2 = np.linspace(17.1, 18.3, 4)
        ax.plot(x_N2, x_N2*0.+21.2, color='r', ls='--', lw=2)
        x_N3 = np.linspace(18.3, 20.1, 4)
        ax.plot(x_N3, 2.9 + x_N3, color='r', ls='--', lw=2)
        
        FMC = np.zeros_like(coord[x], dtype='?')
        FMC |= ((coord[y] < (2.9 + 1.2) + coord[x]) & (coord[x] < 17.1))
        FMC |= ((coord[y] < 21.2) & (coord[x] < 18.3) & (coord[x] > 17.1))
        FMC |= ((coord[y] < 2.9 + coord[x]) & (coord[x] > 18.3))
        
        maskhigh = (~FMC) & (catmask)
        masklow = (FMC) & (catmask)
        ax.text(xlim[1]-8*dx, ylim[1]-dy, r'Ntot. %i' %(Ntot), size=15)
        if area is not None: ax.text(xlim[1]-8*dx, ylim[1]-2*dy, r'$\eta$. %.2f/deg$^2$' %(Ntot/area), size=15)
        ax.text(xlim[1]-5*dx, ylim[0]+dy, r'f.kept. %.2f %%' %(100 * np.sum(masklow)/Ntot), size=15)
        ax.text(xlim[0]+dx, ylim[1]-dy, r'f.rej. %.2f %%' %(100 * np.sum(maskhigh)/Ntot), size=15)
        
    #if bins is not None: clab = r'$\log(N)$'
    clab = r'$N$'
    fig.colorbar(pos, ax=ax, label=clab, orientation="horizontal", pad=0.15)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    if file is not None:
        fig.savefig(file+'.png', bbox_inches = 'tight', pad_inches = 0)
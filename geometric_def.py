import numpy as np
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys, os, warnings
from astropy.table import Table
import matplotlib.pyplot as plt

#from kdcount import KDTree
#from veto import veto, veto_ellip, match


def circular_mask_radii_func(MAG, radii_1, bestfit=True):
    '''
    Define mask radius as a function of the magnitude

    Inputs
    ------
    MAG: Magnitude in (array);
    radii_1: radii as a function of magnitude;
    bestfit: True to get the best-fit of radii instead;

    Output
    ------
    radii: mask radii (array)
    '''
    
    x, y = np.transpose(radii_1)
    
    if not bestfit:
        circular_mask_radii_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    else:
        powr = np.polyfit(x, y, 2)
        Y = powr[0]*x**2 + powr[1]*x + powr[2]
        circular_mask_radii_func = interp1d(x, Y, bounds_error=False, fill_value=(y[0], y[-1]))
        
    # mask radius in arcsec
    return circular_mask_radii_func(MAG)

def ds_masking_func(d_ra, d_dec, d2d, w1_ab, length_radii, widht_radii):
    '''
    Masking function for diffraction spikes
    Inputs
    ------
    d_ra, d_dec: (array) the differences in RA and Dec (arcsec); 
    d2d: (array) angular distances (arcsec);
    w1_ab: (array) W1 magnitude in AB;
    
    Output
    ------
    ds_flag: array of mask value; True if masked (contaminated).
    '''
    
    # Define length for diffraction spikes mask
    x, y = np.transpose(length_radii)
    ds_mask_length_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))

    # Define width for diffraction spikes mask
    x, y = np.transpose(widht_radii)
    ds_mask_width_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))
    
    ds_mask_widths = ds_mask_width_func(w1_ab)
    ds_mask_length = ds_mask_length_func(w1_ab)

    mask1 = d_dec > (d_ra - ds_mask_widths/np.sqrt(2))
    mask1 &= d_dec < (d_ra + ds_mask_widths/np.sqrt(2))
    mask1 &= (d_dec < -d_ra + ds_mask_length/np.sqrt(2)) & (d_dec > -d_ra - ds_mask_length/np.sqrt(2))

    mask2 = d_dec > (-d_ra - ds_mask_widths/np.sqrt(2))
    mask2 &= d_dec < (-d_ra + ds_mask_widths/np.sqrt(2))
    mask2 &= (d_dec < +d_ra + ds_mask_length/np.sqrt(2)) & (d_dec > +d_ra - ds_mask_length/np.sqrt(2))

    ds_flag = (mask1 | mask2)
    
    return ds_flag


def query_catalog_mask(ra, dec, starCat, radii, nameMag='MAG_VT', diff_spikes=True, length_radii=None, widht_radii=None, return_diagnostics=False, bestfit=True):
    '''
    Catalog-based WISE bright star mask.
    Input:
    ra, dec: coordinates;
    diff_spikes: apply diffraction spikes masking if True;
    return_diagnostics: return disgnostic information if True;
    Return:
    cat_flag: array of mask value; the location is masked (contaminated) if True.
    '''

    import time
    start = time.time()
    
    wisecat = starCat

    w1_ab = np.array(wisecat[nameMag])
    raW = np.array(wisecat['RA'])
    decW = np.array(wisecat['DEC'])

    w1_bins = np.arange(0, 22, 0.5)
    
    # only flagged by the circular mask (True if contaminated):
    circ_flag = np.zeros(len(ra), dtype=bool)
    # flagged by the diffraction spike mask but not the circular mask (True if contaminated):
    ds_flag = np.zeros(len(ra), dtype=bool)
    # flagged in the combined masks (True if contaminated):
    cat_flag = np.zeros(len(ra), dtype=bool)

    # record the magnitude of the star that causes the contamination and distance to it
    w1_source = np.zeros(len(ra), dtype=float)
    d2d_source = np.zeros(len(ra), dtype=float)

    ra2, dec2 = map(np.copy, [ra, dec])
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')

    for index in range(len(w1_bins)-1):

        mask_wise = (w1_ab>=w1_bins[index]) & (w1_ab<=w1_bins[index+1])
        print('{:.2f} < {} < {:.2f}   {} TYCHO bright stars'.format(w1_bins[index], nameMag, w1_bins[index+1], np.sum(mask_wise)))

        if np.sum(mask_wise)==0:
            print()
            continue
    
        # find the maximum mask radius for the magnitude bin        
        if not diff_spikes:
            search_radius = np.max(circular_mask_radii_func(w1_ab[mask_wise], radii, bestfit=bestfit))
        else:
            # Define length for diffraction spikes mask
            x, y = np.transpose(length_radii)
            ds_mask_length_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], 0))
            search_radius = np.max([circular_mask_radii_func(w1_ab[mask_wise], radii, bestfit=bestfit), 0.5*ds_mask_length_func(w1_ab[mask_wise])])

        # Find all pairs within the search radius
        ra1, dec1 = map(np.copy, [raW[mask_wise], decW[mask_wise]])
        sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
        idx_wise, idx_decals, d2d, _ = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
        print('%d nearby objects'%len(idx_wise))
        
        # convert distances to numpy array in arcsec
        d2d = np.array(d2d.to(u.arcsec))

        d_ra = (ra2[idx_decals]-ra1[idx_wise])*3600.    # in arcsec
        d_dec = (dec2[idx_decals]-dec1[idx_wise])*3600. # in arcsec
        ##### Convert d_ra to actual arcsecs #####
        mask = d_ra > 180*3600
        d_ra[mask] = d_ra[mask] - 360.*3600
        mask = d_ra < -180*3600
        d_ra[mask] = d_ra[mask] + 360.*3600
        d_ra = d_ra * np.cos(dec1[idx_wise]/180*np.pi)
        ##########################################

        # circular mask
        mask_radii = circular_mask_radii_func(w1_ab[mask_wise][idx_wise], radii, bestfit=bestfit)
        # True means contaminated:
        circ_contam = d2d < mask_radii
        circ_flag[idx_decals[circ_contam]] = True

        w1_source[idx_decals[circ_contam]] = w1_ab[mask_wise][idx_wise[circ_contam]]
        d2d_source[idx_decals[circ_contam]] = d2d[circ_contam]

        if diff_spikes:

            ds_contam = ds_masking_func(d_ra, d_dec, d2d, w1_ab[mask_wise][idx_wise], length_radii, widht_radii)
            ds_flag[idx_decals[ds_contam]] = True

            # combine the two masks
            cat_flag[idx_decals[circ_contam | ds_contam]] = True

            w1_source[idx_decals[ds_contam]] = w1_ab[mask_wise][idx_wise[ds_contam]]
            d2d_source[idx_decals[ds_contam]] = d2d[ds_contam]

            print('{} objects masked by circular mask'.format(np.sum(circ_contam)))
            print('{} additionally objects masked by diffraction spikes mask'.format(np.sum(circ_contam | ds_contam)-np.sum(circ_contam)))
            print('{} objects masked by the combined masks'.format(np.sum(circ_contam | ds_contam)))
            print()

        else:

            print('{} objects masked'.format(np.sum(circ_contam)))
            print()

    if not diff_spikes:
        cat_flag = circ_flag
        
    end = time.time()
    print('Total run time: %f sec' %(end - start))

    if not return_diagnostics:
        return cat_flag
    else:
        # package all the extra info
        more_info = {}
        more_info['w1_source'] = w1_source
        more_info['d2d_source'] = d2d_source
        more_info['circ_flag'] = circ_flag
        more_info['ds_flag'] = ds_flag
        
    
    return cat_flag, more_info

def LSLGA_fit(LSLGA, radii='D25', N=1):
    
    MAG = np.array(LSLGA['MAG'])
    if radii == 'mag-rad':
        major = circular_mask_radii_func(MAG, radii)/3600.#[degrees]
    elif radii == 'D25':
        major = N*LSLGA['D25']/2./60. #[degrees]
    else:
        raise ValueError('User a valid radii:{D25,mag-rad}')
        
    minor = major*LSLGA['BA']#[degrees]
    angle = 90 - LSLGA['PA']
    
    return LSLGA['RA'], LSLGA['DEC'], major, minor, angle

def LSLGA_veto(cat, LSLGA, radii, N):
    
    import time
    start = time.time()
    
    RA, DEC, major, minor, angle = LSLGA_fit(LSLGA, radii, N)
    centers = (RA, DEC)
    mask = veto_ellip((cat['RA'], cat['DEC']), centers, major, minor, angle)
    
    end = time.time()
    print('Total run time: %f sec' %(end - start))
    
    return mask

def get_GM_stats(mask, mask_ran, A, B, F, venn=False, filename=None):
    
    Tab = []
    if len(mask) > 2:
        row_names = ['STARS', 'LG', 'GC', '(STARS | LG | GC)','~(STARS | LG | GC)*', '~(STARS | LG | GC)']
    else:
        row_names = ['STARS', 'LG', '(STARS | LG)','~(STARS | LG)*', '~(STARS | LG)']
    
    for i in range(len(mask)):
        
        A_i = (np.sum(mask_ran[i])/len(mask_ran[i]))*(A)
        eta_B_i_in = np.sum((mask[i] & (B)))/(A_i) #density over the geometric area
        eta_F_i_in = np.sum((mask[i] & (F)))/(A_i) #density over the geometric area
        eta_B_i = np.sum((mask[i] & (B)))/(A) #density over the total area
        eta_F_i = np.sum((mask[i] & (F)))/(A) #density over the total area
        
        Tab.append([row_names[i], A_i*(100/A), eta_B_i, eta_F_i])
    
    if len(mask) > 2:
        GMT = (mask[0]) | (mask[1]) | (mask[2])
        GMT_ran = (mask_ran[0]) | (mask_ran[1]) | (mask_ran[2])
        n = 1
    else:
        GMT = (mask[0]) | (mask[1])
        GMT_ran = (mask_ran[0]) | (mask_ran[1])
        n = 0
    
    A_GMT_in = (np.sum(GMT_ran)/len(GMT_ran))*(A)
    eta_B_GMT_in_1 = np.sum((GMT) & (B))/(A) #Not corrected for mask area
    eta_F_GMT_in_1 = np.sum((GMT) & (F))/(A) #Not corrected for mask area

    A_GMT_out = (np.sum(~GMT_ran)/len(GMT_ran))*(A)
    eta_B_GMT_out_1 = np.sum((~GMT) & (B))/(A) #Not corrected for mask area
    eta_B_GMT_out_2 = np.sum((~GMT) & (B))/(A_GMT_out) #Corrected for mask area
    eta_F_GMT_out_1 = np.sum((~GMT) & (F))/(A) #Not corrected for mask area
    eta_F_GMT_out_2 = np.sum((~GMT) & (F))/(A_GMT_out) #Corrected for mask area
    
    Tab.append([row_names[2+n], A_GMT_in*(100/A), eta_B_GMT_in_1, eta_F_GMT_in_1])
    Tab.append([row_names[3+n], A_GMT_out*(100/A), eta_B_GMT_out_1, eta_F_GMT_out_1])
    Tab.append([row_names[4+n], A_GMT_out*(100/A), eta_B_GMT_out_2, eta_F_GMT_out_2])
    
    Tab = np.transpose(Tab)
    t = Table([Tab[0], Tab[1], Tab[2], Tab[3]], 
              names=('GM','$f_{A}$ [$\%$]', '$\eta_{B}$ [deg$^2$]', '$\eta_{F}$ [deg$^2$]'),
                    dtype=('S', 'f8', 'f8', 'f8'))
        
    if venn:
        from matplotlib_venn import venn2
        
        sf = 2
        M = [B,F]
        for i in range(len(M)):
            fig = plt.figure(figsize=(5, 5))
            a = (mask[0]) & (M[i])
            b = (mask[1]) & (M[i])
            c = (a) & (b)

            if i == 0:
                Mi = 'B'
            else:
                Mi = 'F'
                
            a1 = round((np.sum(a) - np.sum(c))/A, sf)
            b1 = round((np.sum(b) - np.sum(c))/A, sf)
            c1 = round(np.sum(c)/A, sf)

            venn2([a1, b1, c1], set_labels = (row_names[0]+'_%s' %(Mi), row_names[1]+'_%s' %(Mi)))
            if filename is not None:
                plt.savefig(filename + '_' + Mi + '.png')
                                    
    return t

def get_PM_stats(mask, mask_ran, mask2, mask2_ran, A, B, F, venn=False, filename=None):
    
    Tab = []
    row_names = ['ALLMASK', 'NOBS', '(ALLMASK | NOBS)','~(STARS | LG | ALLMASK | NOBS)*', '~(STARS | LG | ALLMASK | NOBS)']
    GMT = mask2
    GMT_ran = mask2_ran
    
    for i in range(len(mask)):
        
        A_i = (np.sum((mask_ran[i]) & (~GMT_ran))/len(mask_ran[i]))*(A)
        eta_B_i_in = np.sum((mask[i]) & (~GMT) & (B))/(A_i) #density over the geometric area
        eta_F_i_in = np.sum((mask[i]) & (~GMT) & (F))/(A_i) #density over the geometric area
        eta_B_i = np.sum((mask[i]) & (~GMT) & (B))/(A) #density over the total area
        eta_F_i = np.sum((mask[i]) & (~GMT) & (F))/(A) #density over the total area
        
        Tab.append([row_names[i], A_i*(100/A), eta_B_i, eta_F_i])
        
    PMT = (mask[0]) | (mask[1])
    PMT_ran = (mask_ran[0]) | (mask_ran[1])
    
    A_PMT_in = (np.sum((PMT_ran) & (~GMT_ran))/len(PMT_ran))*(A)
    eta_B_PMT_in_1 = np.sum((PMT) & (~GMT) & (B))/(A) #density over the total area
    eta_F_PMT_in_1 = np.sum((PMT) & (~GMT) & (F))/(A) #density over the total area
    
    A_PMT_out = (np.sum((~PMT_ran) & (~GMT_ran))/len(GMT_ran))*(A)
    eta_B_PMT_out_1 = np.sum((~PMT) & (~GMT) & (B))/(A) #Not corrected for mask area
    eta_B_PMT_out_2 = np.sum((~PMT) & (~GMT) & (B))/(A_PMT_out) #Corrected for mask area
    eta_F_PMT_out_1 = np.sum((~PMT) & (~GMT) & (F))/(A) #Not corrected for mask area
    eta_F_PMT_out_2 = np.sum((~PMT) & (~GMT) & (F))/(A_PMT_out) #Corrected for mask area
    
    Tab.append([row_names[2], A_PMT_in*(100/A), eta_B_PMT_in_1, eta_F_PMT_in_1])
    Tab.append([row_names[3], A_PMT_out*(100/A), eta_B_PMT_out_1, eta_F_PMT_out_1])
    Tab.append([row_names[4], A_PMT_out*(100/A), eta_B_PMT_out_2, eta_F_PMT_out_2])
    
    Tab = np.transpose(Tab)
    t = Table([Tab[0], Tab[1], Tab[2], Tab[3]], 
              names=('PM','$f_{A}$ [$\%$]', '$\eta_{B}$ [deg$^2$]', '$\eta_{F}$ [deg$^2$]'),
                 dtype=('S', 'f8', 'f8', 'f8'))
        
    if venn:
        from matplotlib_venn import venn2
        
        sf = 2
        M = [B,F, 'ran']
        for i in range(len(M)):
            if i > 1:
                a = (mask_ran[0]) & (~GMT_ran)
                b = (mask_ran[1]) & (~GMT_ran)
            else:
                a = (mask[0]) & (~GMT) & (M[i])
                b = (mask[1]) & (~GMT) & (M[i])
                
            c = (a) & (b)

            if i == 0:
                Mi = 'B'
            if i == 1:
                Mi = 'F'
            if i == 2:
                Mi = M[i]
                
            a1 = round((np.sum(a) - np.sum(c))/A, sf)
            b1 = round((np.sum(b) - np.sum(c))/A, sf)
            c1 = round(np.sum(c)/A, sf)

            plt.figure(figsize=(5, 5))
            venn2([a1, b1, c1], set_labels = (row_names[0]+'_%s' %(Mi), row_names[1]+'_%s' %(Mi)))
            if filename is not None:
                plt.savefig(filename + '_' + Mi + '.png')
            #plt.show()
        
    return t

def sky_PM(coord, coord_ran, mask, mask_ran, mask2, mask2_ran, lim=None, filename=None):
    
    import matplotlib.gridspec as gridspec
    figsize = (18, 8)
    rows, cols = 1, 2
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(num=1, figsize=figsize)
    ax = []
    GMT = mask2
    GMT_ran = mask2_ran
    data = [(mask) & (~GMT), (mask_ran) & (~GMT_ran)]

    for i, j in enumerate(data):
        row = (i // cols)
        col = i % cols
        ax.append(fig.add_subplot(gs[row, col]))
            
        if i == 0:
            PMlab = ['ALLMASK', 'NOBS']
            x, y = coord[0], coord[1]
            if lim is not None:
                s1, s2 = 20, 100
                limMask = (coord[0] > lim[0]) & (coord[0] < lim[1]) & (coord[1] > lim[2]) & (coord[1] < lim[3])
            else:
                s1, s2 = 5, 25
                limMask = np.ones_like(coord[0], dtype='?')
            
            title = 'TRACTOR'
        else:
            PMlab = ['ALLMASK ran', 'NOBS ran']
            x, y = coord_ran[0], coord_ran[1]
            if lim is not None:
                s1, s2 = 20, 100
                limMask = (coord_ran[0] > lim[0]) & (coord_ran[0] < lim[1]) & (coord_ran[1] > lim[2]) & (coord_ran[1] < lim[3])
                res = ~((mask_ran[0]) | (mask_ran[1])) & (~GMT_ran) & (limMask)
                ax[-1].scatter(x[res], y[res], s=s1*0.1, color='gray', alpha=0.3, label='~(allmask | nobs)')
            else:
                s1, s2 = 5, 25
                limMask = np.ones_like(coord_ran[0], dtype='?')
            title = 'RANDOMS'
            
        ax[-1].set_title('%s' %(title))
        ax[-1].scatter(x[(j[0]) & (limMask)], y[(j[0]) & (limMask)], s=s1*3, alpha=0.5, c='r', label=PMlab[0])
        ax[-1].scatter(x[(j[1]) & (limMask)], y[(j[1]) & (limMask)], s=s1, alpha=0.5, c='b', label=PMlab[1])
        print('%s=%i' %(PMlab[0],np.sum((j[0]) & (limMask))))
        print('%s=%i' %(PMlab[1],np.sum((j[1]) & (limMask))))

        plt.xlabel('RA (deg)', size=20)
        plt.xlabel('DEC (deg)', size=20)
        plt.legend()
        plt.grid()
        
        if filename is not None:
            plt.savefig(filename+'.png')
        
        #if lim is not None:
        #    plt.xlim(lim[0], lim[1])
        #    plt.ylim(lim[2], lim[3])

    plt.show()
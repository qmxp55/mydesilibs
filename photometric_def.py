import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
from PIL import Image
import os
from astropy.table import Table

def get_stars(gaiagmag, fluxr):
    
    Grr = gaiagmag - 22.5 + 2.5*np.log10(fluxr)
    GAIA_STAR = np.ones_like(gaiagmag, dtype='?')
    GAIA_STAR &= (Grr  <  0.6) & (gaiagmag != 0)
    
    return GAIA_STAR

def get_galaxies(gaiagmag, fluxr):
    
    Grr = gaiagmag - 22.5 + 2.5*np.log10(fluxr)
    GAIA_GAL = np.ones_like(gaiagmag, dtype='?')
    GAIA_GAL &= (Grr  >  0.6) | (gaiagmag == 0)
    
    return GAIA_GAL

def get_photo_stats(mask, mask2, mask2_ran, A, B, F, row_names):
    
    #totmask = np.ones_like(mask, dtype='?')
    Tab = []
    A_G_out = (np.sum(~mask2_ran)/len(mask2_ran))*(A)
    #print(np.sum(~totmask))
    
    if len(mask) < 2:
        eta_B_star = np.sum((mask) & (~mask2) & (B))/(A_G_out)
        eta_F_star = np.sum((mask) & (~mask2) & (F))/(A_G_out)    
        Tab.append([row_names[0],eta_B_star, eta_F_star])
        totmask = mask[0]
    
    if len(mask) > 1:
        for i in range(len(mask)):
            eta_B_star = np.sum((mask[i]) & (~mask2) & (B))/(A_G_out)
            eta_F_star = np.sum((mask[i]) & (~mask2) & (F))/(A_G_out)    
            Tab.append([row_names[i],eta_B_star, eta_F_star])
            if i == 0:
                totmask = mask[0]
            totmask |= mask[i]
        
    eta_B_gal = np.sum((~totmask) & (~mask2) & (B))/(A_G_out) #Corrected for mask area
    eta_F_gal = np.sum((~totmask) & (~mask2) & (F))/(A_G_out) #Corrected for mask area
    Tab.append([row_names[len(mask)], eta_B_gal, eta_F_gal])
    
    Tab = np.transpose(Tab)
    t = Table([Tab[0], Tab[1], Tab[2]], names=('PHOTOMETRY','$\eta_{B}$ [deg$^2$]', '$\eta_{F}$ [deg$^2$]'),
             dtype=('S', 'f8', 'f8'))
        
    return t

def plot_venn3(mask, mask2, mask2_ran, A, B, F, row_names, filename=None):  
    sf = 2
    M = [B,F]
    A_G_out = (np.sum(~mask2_ran)/len(mask2_ran))*(A)
    for i in range(len(M)):
        A1 = (mask[0]) & (~mask2) & (M[i])
        B1 = (mask[1]) & (~mask2) & (M[i])
        C1 = (mask[2]) & (~mask2) & (M[i])
        AB = (A1) & (B1)
        AC = (A1) & (C1)
        BC = (B1) & (C1)
        ABC = (A1) & (B1) & (C1)
            
        a1 = round((np.sum(A1) - np.sum(AB) - np.sum(AC) + np.sum(ABC))/A_G_out, sf)
        a2 = round((np.sum(B1) - np.sum(AB) - np.sum(BC) + np.sum(ABC))/A_G_out, sf)
        a3 = round((np.sum(AB) - np.sum(ABC))/A_G_out, sf)
        a4 = round((np.sum(C1) - np.sum(AC) - np.sum(BC) + np.sum(ABC))/A_G_out, sf)
        a5 = round((np.sum(AC) - np.sum(ABC))/A_G_out, sf)
        a6 = round((np.sum(BC) - np.sum(ABC))/A_G_out, sf)
        a7 = round(np.sum(ABC)/A_G_out, sf)

        if i == 0:
            Mi = 'B'
        else:
            Mi = 'F'
            
        from matplotlib_venn import venn3
        plt.figure(figsize=(5, 5))
        
        venn3([a1, a2, a3, a4, a5, a6, a7], 
                  set_labels = (row_names[0]+'_%s' %(Mi), row_names[1]+'_%s' %(Mi), row_names[2]+'_%s' %(Mi)))
        if filename is not None:
                plt.savefig(filename + '_' + Mi + '.png')

def flow_1(file, tGM, tPM, area, main_bright, main_faint, patch):

    SU = 'START UP'
    SG = 'GEOMTRICAL'

    I = masking(title='DECaLS DR7', submasks=['rmag < 20', '%i < RA < %i' %(patch[0], patch[1]), 
                                              '%i < DEC < %i' %(patch[2], patch[3])], details=None)
    IGM = masking(title='Geometric Masking', submasks=['Bright Stars (BS)', 'Large Galaxies (LG)'], 
            details=['circular', 'elliptical'])
    IPM = masking(title='Pixel Masking', submasks=['ALLMASK', 'NOBS'], 
            details=['allmask_{G,R,Z} = 0', 'nobs_{G,R,Z} > 0'])

    #RESULTS...
    RI = results(a=area, b=main_bright, f=main_faint, stage='ini', per=False)
    RGM = results(a=tGM[3][1], b=tGM[3][2], f=tGM[3][3], b2=tGM[4][2], f2=tGM[4][3], stage='geo', per=True)
    RPM = results(a=tPM[3][1], b=tPM[3][2], f=tPM[3][3], b2=tPM[4][2], f2=tPM[4][3], stage='geo', per=True)
    #REJECTED by GM...
    RBS = results(a=tGM[0][1], b=tGM[0][2], f=tGM[0][3], stage='ini', per=True, title='Bright Stars')
    RLG = results(a=tGM[1][1], b=tGM[1][2], f=tGM[1][3], stage='ini', per=True, title='Large Galaxies')
    RGMR = results(a=tGM[2][1], b=tGM[2][2], f=tGM[2][3], stage='ini', per=True, title='(BS) or (LG)')
    #REJECTED by PM...
    RAM = results(a=tPM[0][1], b=tPM[0][2], f=tPM[0][3], stage='ini', per=True, title='ALLMASK')
    RNOBS = results(a=tPM[1][1], b=tPM[1][2], f=tPM[1][3], stage='ini', per=True, title='NOBS')
    RPMR = results(a=tPM[2][1], b=tPM[2][2], f=tPM[2][3], stage='ini', per=True, title='(allmask) or (nobs)')

    G=pgv.AGraph(strict=False,directed=True)
    #G.graph_attr['label']="miles_dat"

    elist=[(SU,I), (I,RI),(RI,SG),(SG,IGM),(IGM,RGM),(IGM,RGMR),(RGMR,RBS), (RGMR, RLG),(RGM,IPM), (IPM, RPM),
          (IPM, RPMR), (RPMR, RAM), (RPMR, RNOBS)]
    G.add_edges_from(elist)

    nlist=[RI, RGM, RPM]
    G.add_nodes_from(nlist, color='deepskyblue', shape='box', style='filled')

    rejects=[RBS, RLG, RGMR, RAM, RNOBS, RPMR]
    G.add_nodes_from(rejects, color='coral', shape='box', style='filled')

    stages=[SU, SG]
    G.add_nodes_from(stages, color='green', style='filled')

    maskings=[I, IGM, IPM]
    G.add_nodes_from(maskings, color='gold', style='filled')

    G.write('%s.dot' %(file)) # write to simple.dot
    BB=pgv.AGraph('%s.dot' %(file)) # create a new graph from file
    BB.layout(prog='dot') # layout with default (neato)
    BB.draw('%s.ps' %(file)) # draw png

    os.system('convert ' + file + '.ps ' + file + '.png')

    flow = Image.open('%s.png' %(file))
    
    return flow


def flow_2(file, tPM, tSG, tFMC, tCC, tQC, tQC_all, area):

    SP = 'PHOTOMETRY'

    ISG = masking(title='Star-Galaxy Separation', submasks=['Galaxies'], details=['(G-rr > 0.6) or (G=0)'])
    IFMC = masking(title='Fibre Magnitude Cut (FMC)', submasks=None, details=['fibmag < rmag + 2.9'])
    ICC = masking(title='Colour Cuts', submasks=None, details=['-1 < g-r < 4', '-1 < r-z < 4'])
    IQC = masking(title='Quality Cuts', submasks=['FRACMASKED (FM)', 'FRACIN (FI)', 'FRACFLUX (FF)', 'FLUXIVAR'], 
                  details=['fracmasked_{G,R,Z}<0.4', 'fracin_{G,R,Z}>0.3', 'fracflux_{G,R,Z}<5', 'flux_ivar_{G,R,Z}>0'])

    #RESULTS...
    RI = results(a=tPM[4][1]*(area/100), b=tPM[4][2], f=tPM[4][3], stage='ini', per=False)
    RSG = results(b=tSG[1][1], f=tSG[1][2], stage='photo', per=True)
    RFMC = results(b=tFMC[1][1], f=tFMC[1][2], stage='photo', per=True)
    RCC = results(b=tCC[1][1], f=tCC[1][2], stage='photo', per=True)
    RQC = results(b=tQC[4][1], f=tQC[4][2], stage='photo', per=True)
    #REJECTED by SG...
    RSGR = results(b=tSG[0][1], f=tSG[0][2], stage='photo', per=True, title='STARS')
    #REJECTED by FM...
    RFMCR = results(b=tFMC[0][1], f=tFMC[0][2], stage='photo', per=True, title='FMC')
    #REJECTED by CC...
    RCCR = results(b=tCC[0][1], f=tCC[0][2], stage='photo', per=True, title='Colour Cuts')
    #REJECTED by QC...
    RFM = results(b=tQC[0][1], f=tQC[0][2], stage='photo', per=True, title='FRACMASKED')
    RFI = results(b=tQC[1][1], f=tQC[1][2], stage='photo', per=True, title='FRACIN')
    RFF = results(b=tQC[2][1], f=tQC[2][2], stage='photo', per=True, title='FRACFLUX')
    RIVAR = results(b=tQC[3][1], f=tQC[3][2], stage='photo', per=True, title='FLUXIVAR')
    RQCR = results(b=tQC_all[0][1], f=tQC_all[0][2], stage='photo', per=True, title='(FM) or (FI) or (FF)')

    G=pgv.AGraph(strict=False,directed=True)

    elist=[(RI, SP), (SP, ISG), (ISG, RSG), (ISG, RSGR), (RSG, IFMC), (IFMC, RFMC), (IFMC, RFMCR), (RFMC, ICC),
               (ICC, RCC), (ICC, RCCR), (RCC, IQC), (IQC, RQC), (IQC, RQCR), (RQCR, RFM), (RQCR, RFI),
                   (RQCR, RFF), (IQC, RIVAR)]
    G.add_edges_from(elist)

    nlist=[RI, RSG, RFMC, RCC, RQC]
    G.add_nodes_from(nlist, color='deepskyblue', shape='box', style='filled')

    rejects=[RSGR, RFMCR, RCCR, RFM, RFI, RFF, RIVAR, RQCR]
    G.add_nodes_from(rejects, color='coral', shape='box', style='filled')

    stages=[SP]
    G.add_nodes_from(stages, color='green', style='filled')

    maskings=[ISG, IFMC, ICC, IQC]
    G.add_nodes_from(maskings, color='gold', style='filled')

    G.write('%s.dot' %(file)) # write to simple.dot
    BB=pgv.AGraph('%s.dot' %(file)) # create a new graph from file
    BB.layout(prog='dot') # layout with default (neato)
    BB.draw('%s.ps' %(file)) # draw png

    os.system('convert ' + file + '.ps ' + file + '.png')

    flow = Image.open('%s.png' %(file))
    
    return flow

def results(a=None, b=None, f=None, b2=None, f2=None, stage='geo', per=True, title=None):
    boldblack = ''#'\033[0;30;1m'
    normal = ''#'\033[0;30;0m'
    gray = ''#'\033[0;30;37m'
    
    if per:
        n = '%'
    else:
        n = 'sq.d'
    
    if stage=='geo':
        R1 = '%s Area: %s%.2f (%s) \n' %(boldblack, normal, a, n)
        R2 = '%s Bright*: %s%.2f (1/sq.d) \n' %(boldblack, normal, b)
        R3 = '%s Faint*: %s%.2f (1/sq.d) \n' %(boldblack, normal, f)
    
        R4 = '%s Bright: %.2f (1/sq.d) \n' %(gray, b2)
        R5 = '%s Faint: %.2f (1/sq.d)' %(gray, f2)
        
        return [R1+R2+R3+R4+R5]
    
    if stage=='photo':
        R = ''
        if title is not None:
            R += '%s \n\n' %(title)
        R += '%s Bright: %s%.2f (1/sq.d) \n' %(boldblack, normal, b)
        R += '%s Faint: %s%.2f (1/sq.d)' %(boldblack, normal, f)
        
        return [R]
    
    if stage=='ini':
        R = ''
        if title is not None:
            R += '%s \n\n' %(title)
        R += '%s Area: %s %.2f (%s) \n' %(boldblack, normal, a, n)
        R += '%s Bright: %s %.2f (1/sq.d) \n' %(boldblack, normal, b)
        R += '%s Faint: %s %.2f (1/sq.d)' %(boldblack, normal, f)
        
        return [R]
    
def masking(title, submasks, details):
    
    if details or submasks is not None:
        R = '%s \n\n' %(title)
    else:
        R = '%s' %(title)
    
    if submasks is not None:
        N = len(submasks)
    if details is not None:
        N = len(details)
        
    if details and submasks is not None:
        for i in range(N):
            if i<len(submasks)-1:
                R +='%i) %s \n %s \n' %(i+1, submasks[i], details[i])
            else:
                R +='%i) %s \n %s' %(i+1, submasks[i], details[i])
                
    elif submasks is not None:
        for i in range(N):
            if i<len(submasks)-1:
                R +='%i) %s \n' %(i+1, submasks[i])
            else:
                R +='%i) %s' %(i+1, submasks[i])
                
    elif details is not None:
        for i in range(N):
            if i<len(details)-1:
                R +='%i) %s \n' %(i+1, details[i])
            else:
                R +='%i) %s' %(i+1, details[i])
    
    return [R]
        
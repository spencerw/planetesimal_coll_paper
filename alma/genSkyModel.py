import math as M
import pylab as P
import numpy as NP
from astropy.io import fits as pyfits
import scipy.fftpack as fft
import random as R
import sys
import numpy as np

fname="e3m2_coll.txt"
fileout="skymodel-e3m2.fits"
SHIFT=0

#this stays the same
N=2400 # make multiples of 300
DIST=100
SCALE=10
dr=0.0202020202/DIST*SCALE
dpix=dr/5
MULTIPLIER=15000000
#MULTIPLIER=15000
#MULTIPLIER=150
normflux = 100 # 10 mJy 
RA=165.4663
DEC=-34.70473

nucenter=345e9
beam=M.pi*0.0278*0.0278/4.
abbeam=M.sqrt(beam)
fwhm=0.0278/2

rhodust=2.5
dustsize=1e-1
Mdust=0.020 # Mearth

Mearth=2e30/320.
hplanck=6.626e-27
c=2.989e10
kb=1.3807e-16
au=1.5e13


BSCALE  = 1.
BZERO = 0.
BMAJ = 0.0278/206265*180/np.pi
BMIN = 0.0278/206265*180/np.pi
BPA = 8.402376556396E+01 
BTYPE='Intensity'
OBJECT = 'COLLISION.MODEL'
BUNIT='mJy/pixel'
EQUINOX =  2000 
RADESYS = 'FK5 '
LONPOLE = 180
LATPOLE = -2.295433409167E+01 
CTYPE1  = 'RA---SIN'                                                            
CRVAL1  =    RA
CDELT1  =  -dpix/3600
CRPIX1  =  N/2.+1.
CUNIT1  = 'deg     '                                                            
CTYPE2  = 'DEC--SIN'                                                            
CRVAL2  =  DEC
CDELT2  =   dpix/3600.
CRPIX2  =   N/2.+1.
CUNIT2  = 'deg     '  

CTYPE3  = 'FREQ    '                                                            
CRVAL3  =    3.509375000000E+11                                                 
CDELT3  =     15.625E6                            
CRPIX3  =   1.000000000000E+00                                                  
CUNIT3  = 'Hz      '  
CTYPE4  = 'STOKES  '                                                            
CRVAL4  =   1.000000000000E+00                                                  
CDELT4  =   1.000000000000E+00                                                  
CRPIX4  =   1.000000000000E+00                                                  
CUNIT4  = '        '                                                            
PV2_1   =   0.000000000000E+00                                                  
PV2_2   =   0.000000000000E+00                                                  
RESTFRQ =   3.509375000000E+11 #/Rest Frequency (Hz)                             
SPECSYS = 'LSRK    '           #/Spectral reference frame                        
ALTRVAL =  -0.000000000000E+00 #/Alternate frequency reference value             
ALTRPIX =   1.000000000000E+00 #/Alternate frequency reference pixel             
VELREF  =                  257 #/1 LSR, 2 HEL, 3 OBS, +256 Radio    

OBSRA   =   RA
OBSDEC  =  DEC                                            
OBSGEOX=   2.225142180269E+06                                          
OBSGEOY=  -5.440307370349E+06                                                  
OBSGEOZ=  -2.481029851874E+06  
TELESCOP='ALMA'


def Iring(r,IR,R,sig):
   return IR*np.exp(-(r-R)**2/(2*sig**2))

def Aring(r,IA,R,sigR,theta,TA,sigT):
   return IA*np.exp(-(r-R)**2/(2*sigR**2))*np.exp(-(theta-TA)**2/(2*sigT**2))

        
def main():
        # for conve6nience, define size arrays
        fh = open(fname,"r")
        counts=[]
        line=fh.readline()
        print(line)
        radius=line.split()
        for i in range(len(radius)): radius[i]=float(radius[i])*SCALE/DIST

        line=fh.readline()
        counts=line.split()
        for i in range(len(counts)): counts[i]=(float(counts[i]))

        fh.close()

        val=[]
        norm=[]
        X=[];Y=[]
        for i in range(N):
                val.append([])
                norm.append([])
                X.append( (float(i))*dpix - (N/2.)*dpix+0.5*dpix)
                Y.append( (float(i))*dpix - (N/2.)*dpix+0.5*dpix)
                for j in range(N):
                        val[i].append(0.)


        sum=0.
        RMIN = np.amin(radius)
        RMAX = np.amax(radius)
        NR = len(radius)
        for ix in range(N):
          for iy in range(N):
              rad = np.sqrt(X[ix]**2+Y[iy]**2)
              #print(rad,RMIN,RMAX)
              if rad < RMIN or rad > RMAX:continue
              id = int( (rad-RMIN)/dr  )

              b = counts[id] + (counts[id+1]-counts[id])/dr*(rad-radius[id])
              val[iy][ix] = b
              sum+=b

        val = np.array(val)
        X = np.array(X)
        Y = np.array(Y)

        P.figure()
        P.contourf(X*DIST,Y*DIST,val,255,cmap='gray')
        P.axis('square')
#        P.xlim([-0.2,0.2])
#        P.ylim([-0.2,0.2])
        P.xlabel('au')
        P.ylabel('au')
        P.savefig('skymodel.png')

        fitsarray=NP.zeros((N,N))
        for i in range(N):
                for j in range(N):
                        fitsarray[j][i]=val[j][i]*normflux/sum

        hdu=pyfits.PrimaryHDU(fitsarray)
        hdulist = pyfits.HDUList([hdu])

        hdr = hdulist[0].header

        hdr['BSCALE']=BSCALE
        hdr['BZERO']=BZERO
        hdr['BMAJ']=BMAJ
        hdr['BMIN']=BMIN
        hdr['BPA']=BPA
        hdr['BTYPE']=BTYPE
        hdr['OBJECT']=OBJECT
        hdr['BUNIT']=BUNIT
        hdr['CTYPE1']=CTYPE1
        hdr['CRVAL1']=CRVAL1
        hdr['CDELT1']=CDELT1
        hdr['CRPIX1']=CRPIX1
        hdr['CUNIT1']=CUNIT1
        hdr['CTYPE2']=CTYPE2
        hdr['CRVAL2']=CRVAL2
        hdr['CDELT2']=CDELT2
        hdr['CRPIX2']=CRPIX2
        hdr['CUNIT2']=CUNIT2

        hdr['CTYPE3']=CTYPE3
        hdr['CRVAL3']=CRVAL3
        hdr['CDELT3']=CDELT3
        hdr['CRPIX3']=CRPIX3
        hdr['CUNIT3']=CUNIT3


        hdr['CTYPE4']=CTYPE4
        hdr['CRVAL4']=CRVAL4
        hdr['CDELT4']=CDELT4
        hdr['CRPIX4']=CRPIX4
        hdr['CUNIT4']=CUNIT4
        hdr['PV2_1']=PV2_1
        hdr['PV2_2']=PV2_2
        hdr['RESFRQ']=RESTFRQ
        hdr['SPECSYS']=SPECSYS
        hdr['ALTRVAL']=ALTRVAL
        hdr['ALTRPIX']=ALTRPIX
        hdr['VELREF']=VELREF

        hdr['EQUINOX']=EQUINOX
        hdr['RADESYS']=RADESYS
        hdr['LONPOLE']=LONPOLE
        hdr['LATPOLE']=LATPOLE

        hdr['OBSRA']=OBSRA
        hdr['OBSDEC']=OBSDEC
        hdr['OBSGEO-X']=OBSGEOX
        hdr['OBSGEO-Y']=OBSGEOY
        hdr['OBSGEO-Z']=OBSGEOZ

        hdulist.writeto(fileout)

main()


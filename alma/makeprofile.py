import math as M
import pylab as P
import matplotlib as mplt
import numpy as NP
from astropy.io import fits as pyfits
import scipy.fftpack as fft
import random as R
import sys
import numpy as np


SKYFITS="skymodel-e1m2.fits" # skymodel for simulated observations
SIMFITS="e1m2_robustm1.fits"  # simulated observations, cleaned output file
affix="e1m2"

hdul = pyfits.open(SIMFITS)

NR=160
dr = 0.5/NR

print(hdul.info)

dra=hdul[0].header["CDELT1"]*3600 # stored as deg, want arcsec
dde=hdul[0].header["CDELT2"]*3600

bmaj=hdul[0].header["BMAJ"]*3600
bmin=hdul[0].header["BMIN"]*3600
bpa=hdul[0].header["BPA"]
print(bmaj,bmin,bpa)


data = np.array(hdul[0].data)
data = data[0][0]

hdul.close()

hdul = pyfits.open(SKYFITS)
sky=np.array(hdul[0].data)

dras=hdul[0].header["CDELT1"]*3600
ddes=hdul[0].header["CDELT2"]*3600


NS=int( np.sqrt(sky.size))


XS=np.zeros(NS)
YS=np.zeros(NS)

for i in range(NS):
   XS[i] = -NS/2*dras + dras*i + 0.5*dras
   YS[i] = -NS/2*ddes + ddes*i + 0.5*ddes



hdul.close()


mind = np.amin(data)*0.9*0
maxd = np.amax(data)

levels = np.linspace(0,256)/255*(maxd-mind) + mind

N=int( np.sqrt(data.size))

X=np.zeros(N)
Y=np.zeros(N)

for i in range(N):
   X[i] = -N/2*dra + dra*i + 0.5*dra
   Y[i] = -N/2*dde + dde*i + 0.5*dde


profile=np.zeros(NR)
countr=np.zeros(NR)
for i in range(N):
  for j in range(N):
          x=X[i]
          y=Y[j]
          r = np.sqrt(x**2+y**2)
          ir = int(r/dr)
          if ir > NR-1:continue
          profile[ir] += data[j][i]
          countr[ir]+=1

rad=np.zeros(NR)
for ir in range(NR):
  rad[ir] = dr*ir+0.5*dr
  if countr[ir]>0: profile[ir]/=countr[ir]


#P.rc('text', usetex=True)

fig=P.figure()
ax=fig.add_subplot(111)
cnt=P.contourf(YS,XS,sky,256,cmap=P.get_cmap("gist_gray_r"))
ax.set_aspect('equal')
P.xlabel(r"$\Delta$RA (arcsec)")
P.ylabel(r"$\Delta\delta$ (arcsec)")
P.title("Skymodel, scaled "+affix)
P.xlim([0.5,-0.5])
P.ylim([-0.5,0.5])
P.savefig("skymodel_"+affix+".png")

fig=P.figure()
ax=fig.add_subplot(111)
cnt=P.contourf(Y,X,data,levels=levels,cmap=P.get_cmap("gist_gray_r"))
ax.set_aspect('equal')
P.xlabel(r"$\Delta$RA (arcsec)")
P.ylabel(r"$\Delta\delta$ (arcsec)")
P.title("Cleaned image, scaled "+affix)
P.xlim([0.5,-0.5])
P.ylim([-0.5,0.5])
ax.add_patch(mplt.patches.Ellipse([0.45,-0.45],bmin,bmaj,angle=bpa,edgecolor="black",Fill=False))
P.savefig("xy_"+affix+".png")

fig=P.figure()
P.plot(rad,profile)
P.xlim([0.2,0.4])
P.xlabel("R (arcsec)")
P.ylabel("Jy/beam")
P.title("Averaged radial profile, scaled "+affix)
P.savefig("profile_"+affix+".eps")


P.show()


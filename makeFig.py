#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pynbody as pb
import natsort as ns
import glob as gl
import os

from astropy import units as u
from astropy.constants import G
simT = u.year/(2*np.pi)
simV = u.AU / simT

import sys
sys.path.insert(0, '../OrbitTools/')
import OrbitTools

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

simT = u.year/(2*np.pi)
simV = u.AU/simT
path = '../files/research/planetFormation/jupResonance/data/'
s_c_files = np.array([path + 'hkshiftfullJupCirc/hkshiftfullJupCirc.ic']+ \
	                  ns.natsorted(gl.glob(path + 'hkshiftfullJupCirc/*.[0-9]*[0-9]')))
s_e_files = np.array([path + 'hkshiftfull/hkshiftfull.ic']+ \
	                  ns.natsorted(gl.glob(path + 'hkshiftfull/*.[0-9]*[0-9]')))

a_in, a_out = 2.2, 3.8

# Regenerate existing plots?
clobber = True
fmt = 'png'
s = 0.005

def make_rtheta():
	file_str = 'figures/rtheta.' + fmt
	if not clobber and os.path.exists(file_str):
		return
	s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-26])
	pl_c, pl_e = OrbitTools.orb_params(s_c), OrbitTools.orb_params(s_e)
	x_c, y_c = pl_c['pos'][:,0], pl_c['pos'][:,1]
	r_c, theta_c = np.sqrt(x_c**2 + y_c**2), np.arctan2(y_c, x_c) + np.pi
	x_e, y_e = pl_e['pos'][:,0], pl_e['pos'][:,1]
	r_e, theta_e = np.sqrt(x_e**2 + y_e**2), np.arctan2(y_e, x_e) + np.pi

	fig, (ax1, ax2) = plt.subplots(figsize=(16,6), nrows=1, ncols=2)
	ax1.scatter(theta_c, r_c, s=s)
	ax1.set_ylim(2, 4)
	ax1.set_xlabel('Azimuth')
	ax1.set_ylabel('Cylindrical Distance [AU]')
	ax2.scatter(theta_e, r_e, s=s)
	ax2.set_ylim(2, 4)
	ax2.set_xlabel('Azimuth')
	ax2.set_ylabel('Cylindrical Distance [AU]')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_ae():
	file_str = 'figures/ae.' + fmt
	if not clobber and os.path.exists(file_str):
		return
	s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-26])
	pl_c, pl_e = OrbitTools.orb_params(s_c), OrbitTools.orb_params(s_e)

	fig, (ax1, ax2) = plt.subplots(figsize=(16,6), nrows=1, ncols=2, sharey=True)
	ax1.scatter(pl_c['a'], pl_c['e'], s=s)
	ax1.set_xlim(2, 4)
	ax1.set_xlabel('Semi-Major Axis [AU]')
	ax1.set_ylabel('Eccentricity')
	ax2.scatter(pl_e['a'], pl_e['e'], s=s)
	ax2.set_xlim(2, 4)
	ax2.set_xlabel('Semi-Major Axis [AU]')
	ax2.set_ylabel('Eccentricity')
	# sharey=true hides the tick labels
	ax2.yaxis.set_tick_params(labelleft=True)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_long_ph():
	file_str = 'figures/long_ph.' + fmt
	if not clobber and os.path.exists(file_str):
		return
	s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-26])
	pl_c, pl_e = OrbitTools.orb_params(s_c), OrbitTools.orb_params(s_e)

	fig, ax = plt.subplots(figsize=(8,8))
	ax.scatter((pl_e['asc_node'] + pl_e['omega'] + np.pi)%(2*np.pi), pl_e['a'], s=s)
	ax.set_ylim(2, 4)
	ax.set_xlabel('Longitude of Perihelion')
	ax.set_ylabel('Semi-Major Axis [AU]')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

make_rtheta()
make_ae()
make_long_ph()
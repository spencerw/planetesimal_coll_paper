#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

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

from scipy import stats
import KeplerOrbit as ko
import CollisionTools as coll

simT = u.year/(2*np.pi)
simV = u.AU/simT

path = 'data/'

# Resonances
res_dist = [2.5, 3.27, 3.7]
res_label = ['3:1', '2:1', '5:3']
res_p = [1, 1, 3]
res_q = [2, 1, 2]

# Width of a resonance with jupiter (in AU)
# Assume body has an eccentricity of e_forced
def res_width_jup(p, q):
    m, m_c = 9.54e-4, 1
    a_jup = 5.2
    ecc_jup = 0.048
    j1 = p + q
    j2 = -p
    alpha = (-j2/j1)**(2./3.)
    ecc = ko.lap(2, 3/2, alpha)/ko.lap(1, 3/2, alpha)*ecc_jup
    dist = alpha*a_jup
    rw = ko.res_width(m, m_c, ecc, j1, j2)*dist
    return rw

def plot_res(axis, res=-1, vertical=True, show_widths=False):
	xmin, xmax = axis.get_xlim()
	ymin, ymax = axis.get_ylim()
	if res == -1:
		for idx, dist in enumerate(res_dist):
			if vertical:
				axis.vlines(dist, ymin, ymax, linestyles='-')
				axis.text(dist, ymax, res_label[idx])
				if show_widths:
					width = res_width_jup(res_p[idx], res_q[idx])
					axis.vlines(res_dist[idx] - width, ymin, ymax, linestyle='--')
					axis.vlines(res_dist[idx] + width, ymin, ymax, linestyle='--')
			else:
				axis.hlines(dist, xmin, xmax, linestyles='-')
				axis.text(xmax, dist, res_label[idx])
				if show_widths:
					width = res_width_jup(res_p[idx], res_q[idx])
					axis.hlines(res_dist[idx] - width, xmin, xmax, linestyle='--')
					axis.hlines(res_dist[idx] + width, xmin, xmax, linestyle='--')
# Snapshots
s_c_files = np.array([path + 'hkshiftfullJupCirc/hkshiftfullJupCirc.ic']+ \
	                  ns.natsorted(gl.glob(path + 'hkshiftfullJupCirc/*.[0-9]*[0-9]')))
s0_c = pb.load(s_c_files[0])
s_e_files = np.array([path + 'hkshiftfull/hkshiftfull.ic']+ \
	                  ns.natsorted(gl.glob(path + 'hkshiftfull/*.[0-9]*[0-9]')))
s0_e = pb.load(s_e_files[0])

# Skip the first 2000 years of the collision output. This is about how long it
# takes the resonances to fully show up in the a-e plane
t_skip = 2000
t_max = 5000

a_in, a_out = 2.2, 3.8
nbins = 30

# Regenerate existing plots?
clobber = True
fmt = 'png'
s = 0.005

# Collision log data

# Circular Jupiter case
mc_c = s0_c['mass'][0]
coll_c_log = coll.CollisionLog(path + 'hkshiftfullJupCirc/collisions', mc_c)
coll_c = coll_c_log.coll
# The v2y output in the collision log is messed up. Fortunately, we can recover it from
# v1y and vNewy
coll_c['v2y'] = (coll_c['m1']*coll_c['v1y'] - (coll_c['m1'] + coll_c['m2'])*coll_c['vNewy'])/coll_c['m2']
coll_c = coll_c[np.logical_and(coll_c['time']/(2*np.pi) <= t_max, coll_c['time']/(2*np.pi) >= t_skip)]

# Eccentric Jupiter case
mc_e = s0_e['mass'][0]
coll_e_log = coll.CollisionLog(path + 'hkshiftfull/collisions', mc_e)
coll_e = coll_e_log.coll
# Fix the v2y output again
coll_e['v2y'] = (coll_e['m1']*coll_e['v1y'] - (coll_e['m1'] + coll_e['m2'])*coll_e['vNewy'])/coll_e['m2']
coll_e = coll_e[np.logical_and(coll_e['time']/(2*np.pi) <= t_max, coll_e['time']/(2*np.pi) >= t_skip)]

# Positions of particles in polar coordinates
s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-1])
pl_c, pl_e = ko.orb_params(s_c), ko.orb_params(s_e)
x_c, y_c = pl_c['pos'][:,0], pl_c['pos'][:,1]
r_c, theta_c = np.sqrt(x_c**2 + y_c**2), np.arctan2(y_c, x_c) + np.pi
x_e, y_e = pl_e['pos'][:,0], pl_e['pos'][:,1]
r_e, theta_e = np.sqrt(x_e**2 + y_e**2), np.arctan2(y_e, x_e) + np.pi

def make_rtheta():
	file_str = 'figures/rtheta.' + fmt
	if not clobber and os.path.exists(file_str):
		return

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

	# Libration width varies with eccentricity so it doesn't make much sense
	# to do show_widths=True here
	fig, (ax1, ax2) = plt.subplots(figsize=(8,8), nrows=2, ncols=1, sharex=True)
	ax1.scatter(pl_c['a'], pl_c['e'], s=s)
	plot_res(ax1, show_widths=False)
	ax1.set_ylabel('Eccentricity')
	ax2.scatter(pl_e['a'], pl_e['e'], s=s)
	plot_res(ax2, show_widths=False)
	ax2.set_xlim(2, 4)
	ax2.set_xlabel('Semimajor Axis [AU]')
	ax2.set_ylabel('Eccentricity')
	# sharey=true hides the tick labels
	ax2.yaxis.set_tick_params(labelleft=True)
	plt.tight_layout(h_pad=0)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_long_ph():
	file_str = 'figures/long_ph.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(8,8))
	ax.scatter((pl_e['asc_node'] + pl_e['omega'] + np.pi)%(2*np.pi), pl_e['a'], s=s)
	plot_res(ax, vertical=False, show_widths=True)
	ax.set_ylim(2, 4)
	ax.set_xlabel('Longitude of Perihelion')
	ax.set_ylabel('Semimajor Axis [AU]')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_a():
	file_str = 'figures/coll_hist_a.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	coll_hist_a_c, coll_bins_a_c = np.histogram(coll_c['a1'], bins=np.linspace(a_in, a_out, num=nbins))
	coll_bins_a_c = 0.5*(coll_bins_a_c[1:] + coll_bins_a_c[:-1])

	coll_hist_a_e, coll_bins_a_e = np.histogram(coll_e['a1'], bins=np.linspace(a_in, a_out, num=nbins))
	coll_bins_a_e = 0.5*(coll_bins_a_e[1:] + coll_bins_a_e[:-1])

	fig, (ax1, ax2) = plt.subplots(figsize=(8,8), nrows=2, ncols=1, sharex=True)
	ax1.plot(coll_bins_a_c, coll_hist_a_c, linestyle='steps-mid')
	plot_res(ax1, show_widths=False)
	ax1.set_ylabel('Numer of Collisions')
	ax2.plot(coll_bins_a_e, coll_hist_a_e, linestyle='steps-mid')
	plot_res(ax2, show_widths=False)
	ax2.set_xlabel('Semimajor Axis [AU]')
	ax2.set_ylabel('Number of Collisions')
	# sharey=true hides the tick labels
	ax2.yaxis.set_tick_params(labelleft=True)
	plt.tight_layout(h_pad=0)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_r():
	file_str = 'figures/coll_hist_r.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	coll_dist_c1 = np.sqrt(coll_c['x1x']**2 + coll_c['x1y']**2)
	coll_hist_r_c, coll_bins_r_c = np.histogram(coll_dist_c1, bins=np.linspace(a_in, a_out, num=nbins))
	coll_bins_r_c = 0.5*(coll_bins_r_c[1:] + coll_bins_r_c[:-1])

	coll_dist_e1 = np.sqrt(coll_e['x1x']**2 + coll_e['x1y']**2)
	coll_hist_r_e, coll_bins_r_e = np.histogram(coll_dist_e1, bins=np.linspace(a_in, a_out, num=nbins))
	coll_bins_r_e = 0.5*(coll_bins_r_e[1:] + coll_bins_r_e[:-1])

	p_c = pb.analysis.profile.Profile(pl_c, min=a_in, max=a_out, nbins=nbins)
	surf_den_c = (p_c['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)

	p_e = pb.analysis.profile.Profile(pl_e, min=a_in, max=a_out, nbins=nbins)
	surf_den_e = (p_e['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)

	fig, ax = plt.subplots(figsize=(16,12), nrows=3, ncols=2, sharey=False, sharex=True)
	
	ax[0][0].plot(coll_bins_r_c, coll_hist_r_c, linestyle='steps-mid')
	plot_res(ax[0][0])
	ax[0][0].set_ylabel('Numer of Collisions')
	ax[1][0].plot(p_c['rbins'], surf_den_c)
	ax[1][0].set_ylabel(r'Surface Density [g cm$^{-2}$]')
	ax[2][0].scatter(r_c, pl_c['e'], s=1)
	ax[2][0].set_ylabel('Eccentricity')
	ax[2][0].set_xlabel('Heliocentric Distance [AU]')
	ax[2][0].set_xlim(a_in, a_out)


	ax[0][1].plot(coll_bins_r_e, coll_hist_r_e, linestyle='steps-mid')
	plot_res(ax[0][1])
	ax[0][1].set_ylabel('Numer of Collisions')
	ax[1][1].plot(p_e['rbins'], surf_den_e)
	ax[1][1].set_ylabel(r'Surface Density [g cm$^{-2}$]')
	ax[2][1].scatter(r_e, pl_e['e'], s=1)
	ax[2][1].set_ylabel('Eccentricity')
	ax[2][1].set_xlabel('Heliocentric Distance [AU]')
	ax[2][1].set_xlim(a_in, a_out)

	plt.tight_layout(h_pad=0)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_m_hist():
	file_str = 'figures/m_hist.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	a11, a12 = 3.2, 3.34
	a21, a22 = 2.7, 2.8

	def ang_diff(ang1, ang2):
		return ((ang1 - ang2) + np.pi)%(2*np.pi) - np.pi

	fig, ax = plt.subplots(2, 2, figsize=(16,12))

	mask = np.logical_and(a_c_e1 > a11, a_c_e1 < a12)
	hist, bins = np.histogram(M_c_e1[mask], bins=np.linspace(0, 2*np.pi, num=20))
	bins = 0.5*(bins[1:] + bins[:-1])
	ax[0,0].plot(bins, hist, linestyle='steps-mid')
	ax[0,0].set_xlabel('Mean Anomaly')
	ax[0,0].set_ylabel('Number of Collisions')
	ax[0,0].set_ylim(0, np.max(hist)*1.1)
	ax[0,0].set_title(str(a11) + ' < a < ' + str(a12) + ' (' + str(len(M_c_e1[mask])) + ' coll)')

	mask = np.logical_and(a_c_e1 > a21, a_c_e1 < a22)
	hist, bins = np.histogram(M_c_e1[mask], bins=np.linspace(0, 2*np.pi, num=20))
	bins = 0.5*(bins[1:] + bins[:-1])
	ax[1,0].plot(bins, hist, linestyle='steps-mid')
	ax[1,0].set_xlabel('Mean Anomaly')
	ax[1,0].set_ylabel('Number of Collisions')
	ax[1,0].set_ylim(0, np.max(hist)*1.1)
	ax[1,0].set_title(str(a21) + ' < a < ' + str(a22) + ' (' + str(len(M_c_e1[mask])) + ' coll)')

	mask = np.logical_and(a_c_c1 > a11, a_c_c1 < a12)
	hist, bins = np.histogram(M_c_c1[mask], bins=np.linspace(0, 2*np.pi, num=20))
	bins = 0.5*(bins[1:] + bins[:-1])
	ax[0,1].plot(bins, hist, linestyle='steps-mid')
	ax[0,1].set_xlabel('Mean Anomaly')
	ax[0,1].set_ylabel('Number of Collisions')
	ax[0,1].set_ylim(0, np.max(hist)*1.1)
	ax[0,1].set_title(str(a11) + ' < a < ' + str(a12) + ' (' + str(len(M_c_c1[mask])) + ' coll)')

	mask = np.logical_and(a_c_c1 > a21, a_c_c1 < a22)
	hist, bins = np.histogram(M_c_c1[mask], bins=np.linspace(0, 2*np.pi, num=20))
	bins = 0.5*(bins[1:] + bins[:-1])
	ax[1,1].plot(bins, hist, linestyle='steps-mid')
	ax[1,1].set_xlabel('Mean Anomaly')
	ax[1,1].set_ylabel('Number of Collisions')
	ax[1,1].set_ylim(0, np.max(hist)*1.1)
	ax[1,1].set_title(str(a21) + ' < a < ' + str(a22) + ' (' + str(len(M_c_c1[mask])) + ' coll)')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')


make_rtheta()
make_ae()
make_long_ph()
make_coll_hist_a()
make_coll_hist_r()
#make_m_hist()
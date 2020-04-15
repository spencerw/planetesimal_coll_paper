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

# Build a PDF from a series of data points using a KDE
from sklearn.neighbors import KernelDensity
def kde(qty, bw=0.01):
    bins = np.linspace(np.log10(np.min(qty)), np.log10(np.max(qty)))
    
    def kde_helper(x, x_grid, **kwargs):
        kde_skl = KernelDensity(kernel='tophat', bandwidth=bw, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)
    
    pdf = kde_helper(np.log10(qty), bins)
    cum_df = np.cumsum(pdf[::-1])[::-1]
    
    return 10.**bins, pdf

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
s = 0.0001

# Collision log data

# Circular Jupiter case
mc = s0_c['mass'][0]
coll_c_log = coll.CollisionLog(path + 'hkshiftfullJupCirc/collisions', mc)
coll_c = coll_c_log.coll
# The v2y output in the collision log is messed up. Fortunately, we can recover it from
# v1y and vNewy
coll_c['v2y'] = (coll_c['m1']*coll_c['v1y'] - (coll_c['m1'] + coll_c['m2'])*coll_c['vNewy'])/coll_c['m2']
coll_c = coll_c[np.logical_and(coll_c['time']/(2*np.pi) <= t_max, coll_c['time']/(2*np.pi) >= t_skip)]

# Eccentric Jupiter case
mc_e = s0_e['mass'][0]
coll_e_log = coll.CollisionLog(path + 'hkshiftfull/collisions', mc)
coll_e = coll_e_log.coll
# Fix the v2y output again
coll_e['v2y'] = (coll_e['m1']*coll_e['v1y'] - (coll_e['m1'] + coll_e['m2'])*coll_e['vNewy'])/coll_e['m2']
coll_e = coll_e[np.logical_and(coll_e['time']/(2*np.pi) <= t_max, coll_e['time']/(2*np.pi) >= t_skip)]

# Low eccentricity Jupiter
mc_e = s0_e['mass'][0]
coll_e1_log = coll.CollisionLog(path + 'e1/collisions', mc)
coll_e1 = coll_e1_log.coll
# Fix the v2y output again
coll_e1['v2y'] = (coll_e1['m1']*coll_e1['v1y'] - (coll_e1['m1'] + coll_e1['m2'])*coll_e1['vNewy'])/coll_e1['m2']

# High eccentricity Jupiter
mc_e = s0_e['mass'][0]
coll_e2_log = coll.CollisionLog(path + 'e2/collisions', mc)
coll_e2 = coll_e2_log.coll
# Fix the v2y output again
coll_e2['v2y'] = (coll_e2['m1']*coll_e2['v1y'] - (coll_e2['m1'] + coll_e2['m2'])*coll_e2['vNewy'])/coll_e2['m2']
coll_e2 = coll_e2

# Low mass Jupiter
mc_e = s0_e['mass'][0]
coll_m1_log = coll.CollisionLog(path + 'm1/collisions', mc)
coll_m1 = coll_m1_log.coll
# Fix the v2y output again
coll_m1['v2y'] = (coll_m1['m1']*coll_m1['v1y'] - (coll_m1['m1'] + coll_m1['m2'])*coll_m1['vNewy'])/coll_m1['m2']

# High mass Jupiter
mc_e = s0_e['mass'][0]
coll_m2_log = coll.CollisionLog(path + 'm2/collisions', mc)
coll_m2 = coll_m2_log.coll
# Fix the v2y output again
coll_m2['v2y'] = (coll_e2['m1']*coll_m2['v1y'] - (coll_m2['m1'] + coll_m2['m2'])*coll_m2['vNewy'])/coll_m2['m2']
coll_m2 = coll_m2

# Positions of particles in polar coordinates
s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-1])
pl_c, pl_e = ko.orb_params(s_c), ko.orb_params(s_e)
x_c, y_c = pl_c['pos'][:,0], pl_c['pos'][:,1]
r_c, theta_c = np.sqrt(x_c**2 + y_c**2), np.arctan2(y_c, x_c) + np.pi
x_e, y_e = pl_e['pos'][:,0], pl_e['pos'][:,1]
r_e, theta_e = np.sqrt(x_e**2 + y_e**2), np.arctan2(y_e, x_e) + np.pi

def make_xy():
	file_str = 'figures/xy.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	# Rotate so that jupiter is at theta =0
	theta_jup_c = np.arctan2(pl_c['pos'][0][1], pl_c['pos'][0][0])
	x_c_rot = x_c*np.cos(-theta_jup_c) - y_c*np.sin(-theta_jup_c)
	y_c_rot = x_c*np.sin(-theta_jup_c) + y_c*np.cos(-theta_jup_c)
	theta_jup_e = np.arctan2(pl_e['pos'][0][1], pl_e['pos'][0][0])
	x_e_rot = x_e*np.cos(-theta_jup_e) - y_e*np.sin(-theta_jup_e)
	y_e_rot = x_e*np.sin(-theta_jup_e) + y_e*np.cos(-theta_jup_e)

	fig, (ax1, ax2) = plt.subplots(figsize=(16,6), nrows=1, ncols=2, sharex=True, sharey=True)
	ax1.scatter(x_c_rot, y_c_rot, s=s)
	ax1.scatter(x_c_rot[0], y_c_rot[0], color='r')
	ax1.set_xlabel('X [AU]')
	ax1.set_ylabel('Y [AU]')
	ax1.set_title('Circular Jupiter')
	ax1.set_xlim(-5.5, 5.5)
	ax1.set_ylim(-5.5, 5.5)
	ax2.scatter(x_e_rot, y_e_rot, s=s)
	ax2.scatter(x_e_rot[0], y_e_rot[0], color='r')
	ax2.set_xlabel('X [AU]')
	ax2.set_ylabel('Y [AU]')
	ax2.set_title('Eccentric Jupiter')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

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
	fig, (ax1, ax2) = plt.subplots(figsize=(8,8), nrows=2, ncols=1, sharex=True, sharey=True)
	ax1.scatter(pl_c['a'], pl_c['e'], s=s)
	plot_res(ax1, show_widths=False)
	ax1.set_ylabel('Eccentricity')
	ax1.set_title('Circular Jupiter')
	ax2.scatter(pl_e['a'], pl_e['e'], s=s)
	plot_res(ax2, show_widths=False)
	ax2.set_xlim(2, 4)
	ax2.set_xlabel('Semimajor Axis [AU]')
	ax2.set_ylabel('Eccentricity')
	ax2.set_title('Eccentric Jupiter')
	# sharey=true hides the tick labels
	ax2.yaxis.set_tick_params(labelleft=True)
	#plt.tight_layout(h_pad=0)
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

	coll_bins_a_c, coll_pdf_a_c = kde(coll_c['a1'])
	coll_bins_a_e, coll_pdf_a_e = kde(coll_e['a1'])

	fig, (ax1, ax2) = plt.subplots(figsize=(8,8), nrows=2, ncols=1, sharex=True)
	ax1.plot(coll_bins_a_c, coll_pdf_a_c, linestyle='steps-mid')
	plot_res(ax1, show_widths=False)
	ax1.set_ylabel('dN/da')
	ax1.set_title('Circular Jupiter')
	ax2.plot(coll_bins_a_e, coll_pdf_a_e, linestyle='steps-mid')
	plot_res(ax2, show_widths=False)
	ax2.set_xlabel('Semimajor Axis [AU]')
	ax2.set_ylabel('dN/da')
	ax2.set_title('Eccentric Jupiter')
	# sharey=true hides the tick labels
	ax2.yaxis.set_tick_params(labelleft=True)
	#plt.tight_layout(h_pad=0)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_r():
	file_str = 'figures/coll_hist_r.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	coll_dist_c1 = np.sqrt(coll_c['x1x']**2 + coll_c['x1y']**2)
	coll_bins_r_c, coll_pdf_r_c = kde(coll_dist_c1)

	coll_dist_e1 = np.sqrt(coll_e['x1x']**2 + coll_e['x1y']**2)
	coll_bins_r_e, coll_pdf_r_e = kde(coll_dist_e1)

	p_c = pb.analysis.profile.Profile(pl_c, min=a_in, max=a_out, nbins=nbins)
	surf_den_c = (p_c['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)

	p_e = pb.analysis.profile.Profile(pl_e, min=a_in, max=a_out, nbins=nbins)
	surf_den_e = (p_e['density']*u.M_sun/u.AU**2).to(u.g/u.cm**2)

	fig, ax = plt.subplots(figsize=(16,12), nrows=3, ncols=2, sharey='row', sharex='col')
	
	ax[0][0].plot(coll_bins_r_c, coll_pdf_r_c, linestyle='steps-mid')
	plot_res(ax[0][0])
	ax[0][0].set_ylabel('dN/dr')
	ax[0][0].set_title('Circular Jupiter')
	ax[1][0].plot(p_c['rbins'], surf_den_c)
	ax[1][0].set_ylabel(r'Surface Density [g cm$^{-2}$]')
	ax[1][0].set_ylim(0.7, 3)
	ax[2][0].scatter(r_c, pl_c['e'], s=1)
	ax[2][0].set_ylabel('Eccentricity')
	ax[2][0].set_xlabel('Heliocentric Distance [AU]')
	ax[2][0].set_xlim(a_in, a_out)
	ax[2][0].set_ylim(-0.02, 0.35)

	ax[0][1].plot(coll_bins_r_e, coll_pdf_r_e, linestyle='steps-mid')
	plot_res(ax[0][1])
	ax[0][1].set_ylabel('dN/dr')
	ax[0][1].set_title('Eccentric Jupiter')
	ax[1][1].plot(p_e['rbins'], surf_den_e)
	ax[1][1].set_ylabel(r'Surface Density [g cm$^{-2}$]')
	ax[2][1].scatter(r_e, pl_e['e'], s=1)
	ax[2][1].set_ylabel('Eccentricity')
	ax[2][1].set_xlabel('Heliocentric Distance [AU]')
	ax[2][1].set_xlim(a_in, a_out)

	plt.tight_layout(h_pad=0)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_e_and_m():
	file_str = 'figures/coll_hist_e_and_m.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(16,12), nrows=2, ncols=2, sharex=True, sharey=True)

	coll_bins_a_e1, coll_pdf_a_e1 = kde(coll_e1['a1'])
	coll_bins_a_e, coll_pdf_a_e = kde(coll_e['a1'])
	coll_bins_a_e2, coll_pdf_a_e2 = kde(coll_e2['a1'])
	coll_bins_a_m1, coll_pdf_a_m1 = kde(coll_m1['a1'])
	coll_bins_a_m2, coll_pdf_a_m2 = kde(coll_m2['a1'])

	coll_dist_e1 = np.sqrt(coll_e1['x1x']**2 + coll_e1['x1y']**2)
	coll_bins_r_e1, coll_pdf_r_e1 = kde(coll_dist_e1)
	coll_dist_e = np.sqrt(coll_e['x1x']**2 + coll_e['x1y']**2)
	coll_bins_r_e, coll_pdf_r_e = kde(coll_dist_e)
	coll_dist_e2 = np.sqrt(coll_e2['x1x']**2 + coll_e2['x1y']**2)
	coll_bins_r_e2, coll_pdf_r_e2 = kde(coll_dist_e2)
	coll_dist_m1 = np.sqrt(coll_m1['x1x']**2 + coll_m1['x1y']**2)
	coll_bins_r_m1, coll_pdf_r_m1 = kde(coll_dist_m1)
	coll_dist_m2 = np.sqrt(coll_m2['x1x']**2 + coll_m2['x1y']**2)
	coll_bins_r_m2, coll_pdf_r_m2 = kde(coll_dist_m2)

	ax[0][0].plot(coll_bins_a_e1, coll_pdf_a_e1, label=r'e$_{pl}$ = 1/2 e$_{jup}$', linestyle='steps-mid')
	ax[0][0].plot(coll_bins_a_e, coll_pdf_a_e, label=r'e$_{pl}$ = e$_{jup}$', linestyle='steps-mid')
	ax[0][0].plot(coll_bins_a_e2, coll_pdf_a_e2, label=r'e$_{pl}$ = 2 e$_{jup}$', linestyle='steps-mid')

	ax[1][0].plot(coll_bins_a_m1, coll_pdf_a_m1, label=r'e$_{pl}$ = 1/2 e$_{jup}$', linestyle='steps-mid')
	ax[1][0].plot(coll_bins_a_e, coll_pdf_a_e, label=r'e$_{pl}$ = e$_{jup}$', linestyle='steps-mid')
	ax[1][0].plot(coll_bins_a_m2, coll_pdf_a_m2, label=r'e$_{pl}$ = 2 e$_{jup}$', linestyle='steps-mid')

	ax[0][1].plot(coll_bins_r_e1, coll_pdf_r_e1, label=r'e$_{pl}$ = 1/2 e$_{jup}$', linestyle='steps-mid')
	ax[0][1].plot(coll_bins_r_e, coll_pdf_r_e, label=r'e$_{pl}$ = e$_{jup}$', linestyle='steps-mid')
	ax[0][1].plot(coll_bins_r_e2, coll_pdf_r_e2, label=r'e$_{pl}$ = 2 e$_{jup}$', linestyle='steps-mid')

	ax[1][1].plot(coll_bins_r_m1, coll_pdf_r_m1, label=r'm$_{pl}$ = 1/2 m$_{jup}$', linestyle='steps-mid')
	ax[1][1].plot(coll_bins_r_e, coll_pdf_r_e, label=r'm$_{pl}$ = m$_{jup}$', linestyle='steps-mid')
	ax[1][1].plot(coll_bins_r_m2, coll_pdf_r_m2, label=r'm$_{pl}$ = 2 m$_{jup}$', linestyle='steps-mid')

	ax[0][0].set_ylabel('dn/dr')
	ax[1][0].set_xlabel('Semimajor Axis [AU]')
	ax[1][1].set_xlabel('Heliocentric Distance [AU]')
	ax[1][0].set_ylabel('dn/dr')

	ax[0][1].legend()
	ax[1][1].legend()

	plt.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

#make_xy()
#make_ae()
#make_long_ph()
#make_coll_hist_a()
#make_coll_hist_r()
make_coll_hist_e_and_m()
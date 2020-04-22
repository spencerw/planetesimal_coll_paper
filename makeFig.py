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

mc = 1
m_jup = 9.54e-4
a_jup = 5.2
ecc_jup = 0.048

# Build a PDF from a series of data points using a KDE
from sklearn.neighbors import KernelDensity
def kde(qty, bw=0.05):
    bins = np.linspace(np.min(qty), np.max(qty))
    
    def kde_helper(x, x_grid, **kwargs):
        kde_skl = KernelDensity(kernel='tophat', bandwidth=bw, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)
    
    pdf = kde_helper(qty, bins)
    
    return bins, pdf

def e_forced(a, ecc_jup):
	return ko.lap(2, 3/2, a/a_jup)/ko.lap(1, 3/2, a/a_jup)*ecc_jup

# Width of a resonance with jupiter (in AU)
# Assume body has an eccentricity of e_forced
def res_width_jup(p, q, ecc_jup=ecc_jup, m_jup=m_jup):
	j1 = p + q
	j2 = -p
	alpha = (-j2/j1)**(2./3.)
	ecc = ko.lap(2, 3/2, alpha)/ko.lap(1, 3/2, alpha)*ecc_jup
	dist = alpha*a_jup
	rw = ko.res_width(m_jup, mc, ecc, j1, j2)*dist
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
s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-1])
pl_c, pl_e = ko.orb_params(s_c), ko.orb_params(s_e)

s_m1_files = np.array([path + 'm1/m1.ic']+ \
	                  ns.natsorted(gl.glob(path + 'm1/m1_coll.[0-9]*[0-9]')))
s0_m1 = pb.load(s_m1_files[0])
s_m2_files = np.array([path + 'm2/m2.ic']+ \
	                  ns.natsorted(gl.glob(path + 'm2/m2_coll.[0-9]*[0-9]')))
s0_m2 = pb.load(s_m2_files[0])
s_m1, s_m2 = pb.load(s_m1_files[-1]), pb.load(s_m2_files[-1])
pl_m1, pl_m2 = ko.orb_params(s_m1), ko.orb_params(s_m2)

s_e1_files = np.array([path + 'e1/e1.ic']+ \
	                  ns.natsorted(gl.glob(path + 'e1/e1.[0-9]*[0-9]')))
s0_e1 = pb.load(s_e1_files[0])
s_e2_files = np.array([path + 'e2/e2.ic']+ \
	                  ns.natsorted(gl.glob(path + 'e2/e2.[0-9]*[0-9]')))
s0_e2 = pb.load(s_e2_files[0])
s_e1, s_e2 = pb.load(s_e1_files[-1]), pb.load(s_e2_files[-1])
pl_e1, pl_e2 = ko.orb_params(s_e1), ko.orb_params(s_e2)

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
mj = s0_c['mass'][1]
coll_c_log = coll.CollisionLog(path + 'hkshiftfullJupCirc/collisions', mc, fix_v2y=True)
coll_c = coll_c_log.coll
coll_c = coll_c[np.logical_and(coll_c['time']/(2*np.pi) <= t_max, coll_c['time']/(2*np.pi) >= t_skip)]
coll_c['dist2'] = np.sqrt(coll_c['x2x']**2 + coll_c['x2y']**2)

# Eccentric Jupiter case
mc_e = s0_e['mass'][0]
coll_e_log = coll.CollisionLog(path + 'hkshiftfull/collisions', mc, fix_v2y=True)
coll_e = coll_e_log.coll
coll_e = coll_e[np.logical_and(coll_e['time']/(2*np.pi) <= t_max, coll_e['time']/(2*np.pi) >= t_skip)]
coll_e['dist2'] = np.sqrt(coll_e['x2x']**2 + coll_e['x2y']**2)

# Low eccentricity Jupiter
mc_e = s0_e['mass'][0]
coll_e1_log = coll.CollisionLog(path + 'e1/collisions', mc, fix_v2y=True)
coll_e1 = coll_e1_log.coll
coll_e1['dist2'] = np.sqrt(coll_e1['x2x']**2 + coll_e1['x2y']**2)

# High eccentricity Jupiter
mc_e = s0_e['mass'][0]
coll_e2_log = coll.CollisionLog(path + 'e2/collisions', mc, fix_v2y=True)
coll_e2 = coll_e2_log.coll
coll_e2['dist2'] = np.sqrt(coll_e2['x2x']**2 + coll_e2['x2y']**2)

# Low mass Jupiter
mc_e = s0_e['mass'][0]
coll_m1_log = coll.CollisionLog(path + 'm1/collisions', mc, fix_v2y=True)
coll_m1 = coll_m1_log.coll
coll_m1['dist2'] = np.sqrt(coll_m1['x2x']**2 + coll_m1['x2y']**2)

# High mass Jupiter
mc_e = s0_e['mass'][0]
coll_m2_log = coll.CollisionLog(path + 'm2/collisions', mc, fix_v2y=True)
coll_m2 = coll_m2_log.coll
coll_m2['dist2'] = np.sqrt(coll_m2['x2x']**2 + coll_m2['x2y']**2)

# Positions of particles in polar coordinates
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
	ax1.plot(coll_bins_a_c, coll_pdf_a_c)
	plot_res(ax1, show_widths=False)
	ax1.set_ylabel('dN/da')
	ax1.set_title('Circular Jupiter')
	ax2.plot(coll_bins_a_e, coll_pdf_a_e)
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
	
	ax[0][0].plot(coll_bins_r_c, coll_pdf_r_c)
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

	ax[0][1].plot(coll_bins_r_e, coll_pdf_r_e)
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

	coll_bins_a_e, coll_pdf_a_e = kde(coll_e['a1'])

	coll_bins_a_e1, coll_pdf_a_e1 = kde(coll_e1['a1'])
	coll_pdf_a_e1 *= len(coll_e1)/len(coll_e)
	coll_bins_a_e2, coll_pdf_a_e2 = kde(coll_e2['a1'])
	coll_pdf_a_e2 *= len(coll_e2)/len(coll_e)
	coll_bins_a_m1, coll_pdf_a_m1 = kde(coll_m1['a1'])
	coll_pdf_a_m1 *= len(coll_m1)/len(coll_e)
	coll_bins_a_m2, coll_pdf_a_m2 = kde(coll_m2['a1'])
	coll_pdf_a_m2 *= len(coll_m2)/len(coll_e)

	coll_dist_e = np.sqrt(coll_e['x1x']**2 + coll_e['x1y']**2)
	coll_bins_r_e, coll_pdf_r_e = kde(coll_dist_e)

	coll_dist_e1 = np.sqrt(coll_e1['x1x']**2 + coll_e1['x1y']**2)
	coll_bins_r_e1, coll_pdf_r_e1 = kde(coll_dist_e1)
	coll_pdf_r_e1 *= len(coll_e1)/len(coll_e)
	coll_dist_e2 = np.sqrt(coll_e2['x1x']**2 + coll_e2['x1y']**2)
	coll_bins_r_e2, coll_pdf_r_e2 = kde(coll_dist_e2)
	coll_pdf_r_e2 *= len(coll_e2)/len(coll_e)
	coll_dist_m1 = np.sqrt(coll_m1['x1x']**2 + coll_m1['x1y']**2)
	coll_bins_r_m1, coll_pdf_r_m1 = kde(coll_dist_m1)
	coll_pdf_r_m1 *= len(coll_m1)/len(coll_e)
	coll_dist_m2 = np.sqrt(coll_m2['x1x']**2 + coll_m2['x1y']**2)
	coll_bins_r_m2, coll_pdf_r_m2 = kde(coll_dist_m2)
	coll_pdf_r_m2 *= len(coll_m2)/len(coll_e)

	ax[0][0].plot(coll_bins_a_e1, coll_pdf_a_e1, label=r'e$_{pl}$ = 1/2 e$_{jup}$')
	ax[0][0].plot(coll_bins_a_e, coll_pdf_a_e, label=r'e$_{pl}$ = e$_{jup}$')
	ax[0][0].plot(coll_bins_a_e2, coll_pdf_a_e2, label=r'e$_{pl}$ = 2 e$_{jup}$')

	ax[1][0].plot(coll_bins_a_m1, coll_pdf_a_m1, label=r'e$_{pl}$ = 1/2 e$_{jup}$')
	ax[1][0].plot(coll_bins_a_e, coll_pdf_a_e, label=r'e$_{pl}$ = e$_{jup}$')
	ax[1][0].plot(coll_bins_a_m2, coll_pdf_a_m2, label=r'e$_{pl}$ = 2 e$_{jup}$')

	ax[0][1].plot(coll_bins_r_e1, coll_pdf_r_e1, label=r'e$_{pl}$ = 1/2 e$_{jup}$')
	ax[0][1].plot(coll_bins_r_e, coll_pdf_r_e, label=r'e$_{pl}$ = e$_{jup}$')
	ax[0][1].plot(coll_bins_r_e2, coll_pdf_r_e2, label=r'e$_{pl}$ = 2 e$_{jup}$')

	ax[1][1].plot(coll_bins_r_m1, coll_pdf_r_m1, label=r'm$_{pl}$ = 1/2 m$_{jup}$')
	ax[1][1].plot(coll_bins_r_e, coll_pdf_r_e, label=r'm$_{pl}$ = m$_{jup}$')
	ax[1][1].plot(coll_bins_r_m2, coll_pdf_r_m2, label=r'm$_{pl}$ = 2 m$_{jup}$')

	ax[0][0].set_ylabel('dn/dr')
	ax[1][0].set_xlabel('Semimajor Axis [AU]')
	ax[1][1].set_xlabel('Heliocentric Distance [AU]')
	ax[1][0].set_ylabel('dn/dr')

	ax[0][1].legend()
	ax[1][1].legend()

	plt.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_hot_cold_a():
	file_str = 'figures/coll_hist_hot_cold_a.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	def coll_hot_cold(coll):
		xv_rel = coll['v2x'] - coll['v1x']
		yv_rel = coll['v2y'] - coll['v1y']
		zv_rel = coll['v2z'] - coll['v1z']
		coll_speed = np.sqrt(xv_rel**2 + yv_rel**2 + zv_rel**2)
		mut_escape = np.sqrt(2*np.min(coll['m1'])/np.min(coll['r1']))
		coll_hot = coll[coll_speed/mut_escape > 1]
		coll_cold = coll[coll_speed/mut_escape < 1]
		return coll_hot, coll_cold

	coll_e1_hot, coll_e1_cold = coll_hot_cold(coll_e1)
	coll_e2_hot, coll_e2_cold = coll_hot_cold(coll_e2)

	coll_bins_a_e1_cold, coll_pdf_a_e1_cold = kde(coll_e1_cold['a1'])
	coll_bins_a_e1_hot, coll_pdf_a_e1_hot = kde(coll_e1_hot['a1'])
	coll_pdf_a_e1_hot *= len(coll_e1_hot)/len(coll_e1_cold)
	coll_bins_a_e2_cold, coll_pdf_a_e2_cold = kde(coll_e2_cold['a1'])
	coll_bins_a_e2_hot, coll_pdf_a_e2_hot = kde(coll_e2_hot['a1'])
	coll_pdf_a_e2_hot *= len(coll_e2_hot)/len(coll_e2_cold)

	fig, ax = plt.subplots(figsize=(16,6), nrows=1, ncols=2, sharex=True, sharey=True)

	ax[0].plot(coll_bins_a_e1_cold, coll_pdf_a_e1_cold, label=r'Cold Collisions (v < v$_{esc}$)')
	ax[0].plot(coll_bins_a_e1_hot, coll_pdf_a_e1_hot, label=r'Hot Collisions (v > v$_{esc}$)')
	ax[1].plot(coll_bins_a_e2_cold, coll_pdf_a_e2_cold, label=r'Cold Collisions (v < v$_{esc}$)')
	ax[1].plot(coll_bins_a_e2_hot, coll_pdf_a_e2_hot, label=r'Hot Collisions (v > v$_{esc}$)')
	ax[0].set_title(r'e$_{pl}$ = 1/2 e$_{jup}$')
	ax[0].legend()
	ax[0].set_ylabel('dn/da')
	ax[0].set_xlabel('Semimajor Axis [AU]')
	ax[1].set_xlabel('Semimajor Axis [AU]')
	ax[1].set_title(r'e$_{pl}$ = 2 e$_{jup}$')

	plt.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_gf_vary():
	file_str = 'figures/coll_gf_vary.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	v_mut_esc = np.sqrt(2*np.min(pl_e['mass'])/np.min(pl_e['eps'][1:]*2))
	ecc_vals = np.logspace(-3, -0.8)
	vrel_vals_31 = ecc_vals*np.sqrt(1/res_dist[0])
	vrel_vals_21 = ecc_vals*np.sqrt(1/res_dist[2])
	cross_31 = (1+v_mut_esc**2/vrel_vals_31**2)*vrel_vals_31
	cross_21 = (1+v_mut_esc**2/vrel_vals_21**2)*vrel_vals_21

	fig, axes = plt.subplots(figsize=(8,8))
	axes.plot(ecc_vals, cross_31/cross_31[0], label='3:1 MMR')
	axes.plot(ecc_vals, cross_21/cross_21[0], label='2:1 MMR')

	axes.set_xscale('log')
	axes.legend()
	axes.set_xlabel(r'$\left< e^{2} \right>^{1/2}$')
	axes.set_ylabel(r'$\left< \sigma v \right>$/$\left< \sigma v \right>_{0}$')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_r_inout_res():
	file_str = 'figures/coll_hist_r_inout_res.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	a21_in, a21_out = 3.2, 3.35

	def coll_in_out_21(coll):
		mask = np.logical_and(coll['a1'] > a21_in, coll['a1'] < a21_out)
		coll_in = coll[mask]
		coll_out = coll[~mask]
		return coll_in, coll_out

	coll_in_21_e1, coll_out_21_e1 = coll_in_out_21(coll_e1)
	coll_in_21_e2, coll_out_21_e2 = coll_in_out_21(coll_e2)

	coll_bins_in_21_e1, coll_pdf_in_21_e1 = kde(coll_in_21_e1['dist2'])
	coll_bins_in_21_e2, coll_pdf_in_21_e2 = kde(coll_in_21_e2['dist2'])
	coll_pdf_in_21_e2 *= len(coll_in_21_e1)/len(coll_in_21_e2)
	coll_bins_out_21_e1, coll_pdf_out_21_e1 = kde(coll_out_21_e1['dist2'])
	coll_bins_out_21_e2, coll_pdf_out_21_e2 = kde(coll_out_21_e2['dist2'])
	coll_pdf_out_21_e2 *= len(coll_out_21_e1)/len(coll_out_21_e2)

	fig, ax = plt.subplots(figsize=(16,6), nrows=1, ncols=2, sharex=True)
	ax[0].plot(coll_bins_in_21_e1, coll_pdf_in_21_e1, label=r'e$_{pl}$ = 1/2 e$_{jup}$')
	ax[0].plot(coll_bins_in_21_e2, coll_pdf_in_21_e2, label=r'e$_{pl}$ = 2 e$_{jup}$')
	ax[1].plot(coll_bins_out_21_e1, coll_pdf_out_21_e1)
	ax[1].plot(coll_bins_out_21_e2, coll_pdf_out_21_e2)
	ax[0].legend()
	ax[0].set_xlabel('Heliocentric Distance [AU]')
	ax[0].set_title(str(a21_in) + ' < a < ' + str(a21_out))
	ax[1].set_xlabel('Heliocentric Distance [AU]')
	ax[0].set_ylabel('dn/dr')
	ax[1].set_title(str(a21_in) + ' > a > ' + str(a21_out))

	plt.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_wander_res_scale():
	file_str = 'figures/wander_res_scale.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	jup_ecc_fac = [0.5, 1, 2]
	labels = ['e$_{pl}$ = 1/2 e$_{jup}$', 'e$_{pl}$ = e$_{jup}$', 'e$_{pl}$ = 2 e$_{jup}$']

	fig, ax = plt.subplots(figsize=(16,6), nrows=1, ncols=2)

	res_idx = 0
	for idx, jef in enumerate(jup_ecc_fac):
		rw = res_width_jup(res_p[res_idx], res_q[res_idx], ecc_jup*jef)

		avals = np.linspace(2,3)
		evals = np.zeros_like(avals)
		for idx1, val in enumerate(avals):
			evals[idx1] = e_forced(val, ecc_jup*jef)

		color = next(ax[0]._get_lines.prop_cycler)['color']
		ax[0].axhline(rw*2, color=color, linestyle='--')
		ax[0].axvline(res_dist[0], linestyle='--', color='k')
		ax[0].plot(avals, avals*(1+evals) - avals, color=color)
		ax[0].set_xlim(2, 3)
		ax[0].set_xlabel('Semimajor Axis [AU]')
		ax[0].set_ylabel('Size Scale [AU]')
		ax[0].set_title('3:1 MMR')

	res_idx = 2
	for idx, jef in enumerate(jup_ecc_fac):
		rw = res_width_jup(res_p[res_idx], res_q[res_idx], ecc_jup*jef)

		avals = np.linspace(2.87,3.67)
		evals = np.zeros_like(avals)
		for idx1, val in enumerate(avals):
			evals[idx1] = e_forced(val, ecc_jup*jef)

		color = next(ax[1]._get_lines.prop_cycler)['color']
		ax[1].axhline(rw*2, color=color, linestyle='--')
		ax[1].axvline(res_dist[1], linestyle='--', color='k')
		ax[1].plot(avals, avals*(1+evals) - avals, color=color, label=labels[idx])
		ax[1].set_xlim(2.87, 3.67)
		ax[1].set_xlabel('Semimajor Axis [AU]')
		ax[1].set_title('2:1 MMR')
		ax[1].legend()

	plt.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

#make_xy()
#make_ae()
#make_long_ph()
#make_coll_hist_a()
#make_coll_hist_r()
#make_coll_hist_e_and_m()
#make_coll_hist_hot_cold_a()
#make_coll_gf_vary()
make_coll_hist_r_inout_res()
make_wander_res_scale()
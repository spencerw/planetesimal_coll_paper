#/bin/bash
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors

mpl.rcParams.update({'font.size': 16, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis', 'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})

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
res_label = ['7:2', '10:3', '3:1', '8:3', '5:2', '2:1', '5:3']
res_p = [2, 3, 1, 3, 2, 1, 3]
res_q = [5, 7, 2, 4, 3, 1, 2]

mc = 1
m_jup = 9.54e-4
a_jup = 5.2
ecc_jup = 0.048

res_dist = []
for idx in range(len(res_p)):
    j1 = res_p[idx] + res_q[idx]
    j2 = -res_p[idx]
    alpha = (-j2/j1)**(2./3.)
    res_dist.append(a_jup*alpha)

# Convert a simulation distance into an angle on the sky for the
# simulated ALMA images
def dist_to_ang(dist):
	# Disk size expanded to increase angular size of observed disk
	jup_pos_arcsec = ((52*u.AU).to(u.pc)/(100*u.pc)*u.rad).to(u.arcsec).value
	return dist/a_jup*jup_pos_arcsec

# Build a PDF from a series of data points using a KDE
from sklearn.neighbors import KernelDensity
def kde(qty, bw=0.02):
    bins = np.linspace(2, 4, (4-2)/bw)
    
    def kde_helper(x, x_grid, **kwargs):
        kde_skl = KernelDensity(kernel='gaussian', bandwidth=bw, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)
    
    pdf = kde_helper(qty, bins)
    
    return bins, pdf

def coll_hist(coll, key, nbins=60):
	bins = np.linspace(2, 4, nbins)

	hist, bins = np.histogram(coll[key], bins=bins)
	bins = 0.5*(bins[1:] + bins[:-1])

	bins, hist = kde(coll[key])

	return bins, hist

def e_forced(a, ecc_jup):
	return ko.lap(2, 3/2, a/a_jup)/ko.lap(1, 3/2, a/a_jup)*ecc_jup

# Width of a resonance with jupiter (in AU)
# Assume body has an eccentricity of e_forced
def res_width_jup(res_idx, ecc_jup=ecc_jup, m_jup=m_jup):
	j1 = res_p[res_idx] + res_q[res_idx]
	j2 = -res_p[res_idx]
	alpha = (-j2/j1)**(2./3.)
	ecc = ko.lap(2, 3/2, alpha)/ko.lap(1, 3/2, alpha)*ecc_jup
	dist = alpha*a_jup
	rw = ko.res_width(m_jup, mc, ecc, j1, j2)*dist
	return rw

def plot_res(axis, res=-1, vertical=True, show_widths=False, all=False):
	xmin, xmax = axis.get_xlim()
	ymin, ymax = axis.get_ylim()
	res_ind = range(len(res_dist))
	if all == False:
		res_ind = [2, 5]
	if res == -1:
		for idx, val in enumerate(res_ind):
			if vertical:
				axis.vlines(res_dist[val], ymin, ymax, linestyles='--', lw=0.5)
				axis.text(res_dist[val], ymax*0.99, res_label[val], verticalalignment='top', fontsize=16)
				if show_widths:
					width = res_width_jup(val)
					axis.vlines(res_dist[val] - width, ymin, ymax, linestyle='-', lw=0.5)
					axis.vlines(res_dist[val] + width, ymin, ymax, linestyle='-', lw=0.5)
			else:
				axis.hlines(res_dist[val], xmin, xmax, linestyles='--', lw=0.5)
				axis.text(xmax*0.99, res_dist[val]*1.01, res_label[val], horizontalalignment='right', fontsize=16)
				if show_widths:
					width = res_width_jup(val)
					axis.hlines(res_dist[val] - width, xmin, xmax, linestyle='-', lw=0.5)
					axis.hlines(res_dist[val] + width, xmin, xmax, linestyle='-', lw=0.5)
# Snapshots
s_c_files = np.array([path + 'hkshiftfullJupCirc/hkshiftfullJupCirc.ic']+ \
	                  ns.natsorted(gl.glob(path + 'hkshiftfullJupCirc/*.[0-9]*[0-9]')))
s0_c = pb.load(s_c_files[0])
s_e_files = np.array([path + 'hkshiftfull/hkshiftfull.ic']+ \
	                  ns.natsorted(gl.glob(path + 'hkshiftfull/*.[0-9]*[0-9]')))
s0_e = pb.load(s_e_files[0])
s_c, s_e = pb.load(s_c_files[-1]), pb.load(s_e_files[-2])
print(s_e_files[-2])
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
coll_c['dist1'] = np.sqrt(coll_c['x2x']**2 + coll_c['x2y']**2)

# Eccentric Jupiter case
mc_e = s0_e['mass'][0]
coll_e_log = coll.CollisionLog(path + 'hkshiftfull/collisions', mc, fix_v2y=True)
coll_e = coll_e_log.coll
coll_e = coll_e[np.logical_and(coll_e['time']/(2*np.pi) <= t_max, coll_e['time']/(2*np.pi) >= t_skip)]
coll_e['dist1'] = np.sqrt(coll_e['x2x']**2 + coll_e['x2y']**2)

# Low eccentricity Jupiter
mc_e = s0_e['mass'][0]
coll_e1_log = coll.CollisionLog(path + 'e1/collisions', mc, fix_v2y=True)
coll_e1 = coll_e1_log.coll
coll_e1['dist1'] = np.sqrt(coll_e1['x2x']**2 + coll_e1['x2y']**2)

# High eccentricity Jupiter
mc_e = s0_e['mass'][0]
coll_e2_log = coll.CollisionLog(path + 'e2/collisions', mc, fix_v2y=True)
coll_e2 = coll_e2_log.coll
coll_e2['dist1'] = np.sqrt(coll_e2['x2x']**2 + coll_e2['x2y']**2)

# Low mass Jupiter
mc_e = s0_e['mass'][0]
coll_m1_log = coll.CollisionLog(path + 'm1/collisions', mc, fix_v2y=True)
coll_m1 = coll_m1_log.coll
coll_m1['dist1'] = np.sqrt(coll_m1['x2x']**2 + coll_m1['x2y']**2)

# High mass Jupiter
mc_e = s0_e['mass'][0]
coll_m2_log = coll.CollisionLog(path + 'm2/collisions', mc, fix_v2y=True)
coll_m2 = coll_m2_log.coll
coll_m2['dist1'] = np.sqrt(coll_m2['x2x']**2 + coll_m2['x2y']**2)

# Positions of particles in polar coordinates
x_e1, y_e1 = pl_e1['pos'][:,0], pl_e1['pos'][:,1]
r_e1, theta_e1 = np.sqrt(x_e1**2 + y_e1**2), np.arctan2(y_e1, x_e1)
x_e, y_e = pl_e['pos'][:,0], pl_e['pos'][:,1]
r_e, theta_e = np.sqrt(x_e**2 + y_e**2), np.arctan2(y_e, x_e)
x_e2, y_e2 = pl_e2['pos'][:,0], pl_e2['pos'][:,1]
r_e2, theta_e2 = np.sqrt(x_e2**2 + y_e2**2), np.arctan2(y_e2, x_e2)

x_m1, y_m1 = pl_m1['pos'][:,0], pl_m1['pos'][:,1]
r_m1, theta_m1 = np.sqrt(x_m1**2 + y_m1**2), np.arctan2(y_m1, x_m1)
x_m2, y_m2 = pl_m2['pos'][:,0], pl_m2['pos'][:,1]
r_m2, theta_m2 = np.sqrt(x_m2**2 + y_m2**2), np.arctan2(y_m2, x_m2)

# Radial collision histograms
coll_bins_e1, coll_hist_e1 = coll_hist(coll_e1, 'dist1')
coll_bins_e, coll_hist_e = coll_hist(coll_e, 'dist1')
coll_bins_e2, coll_hist_e2 = coll_hist(coll_e2, 'dist1')

coll_bins_m1, coll_hist_m1 = coll_hist(coll_m1, 'dist1')
coll_bins_m2, coll_hist_m2 = coll_hist(coll_m2, 'dist1')

def make_coll_polar_e():
	file_str = 'figures/coll_polar_e.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	rmin, rmax = 2, 4

	fig = plt.figure(figsize=(12,14), constrained_layout=True)
	widths = [1, 4]
	heights = [1, 1, 1]
	gs = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)

	# Low eccentricity case
	ecc = ecc_jup/2
	ridx = 2
	rw_31 = res_width_jup(ridx, ecc_jup=ecc)*2
	fecc_31 = e_forced(res_dist[ridx], ecc)
	outer_peri_31 = (res_dist[ridx]+rw_31)*(1+fecc_31)
	outer_apo_31 = (res_dist[ridx]+rw_31)*(1-fecc_31)
	inner_peri_31 = (res_dist[ridx]-rw_31)*(1+fecc_31)
	inner_apo_31 = (res_dist[ridx]-rw_31)*(1-fecc_31)
	ridx = 5
	rw_21 = res_width_jup(ridx, ecc_jup=ecc)*2
	fecc_21 = e_forced(res_dist[ridx], ecc)
	outer_peri_21 = (res_dist[ridx]+rw_21)*(1+fecc_21)
	outer_apo_21 = (res_dist[ridx]+rw_21)*(1-fecc_21)
	inner_peri_21 = (res_dist[ridx]-rw_21)*(1+fecc_21)
	inner_apo_21 = (res_dist[ridx]-rw_21)*(1-fecc_21)

	ax00 = fig.add_subplot(gs[0, 0])
	ax01 = fig.add_subplot(gs[0, 1])
	ax01.set_yticks([])
	ax00.invert_xaxis()

	ax00.plot(coll_hist_e1, coll_bins_e1, drawstyle='steps')
	theta_jup = np.arctan2(pl_e1['pos'][0][1], pl_e1['pos'][0][0])
	ax01.axvline(theta_jup, color='k')
	ax01.scatter(theta_e1, r_e1, s=s*10)

	ax00.axhline(outer_peri_31)
	ax00.axhline(outer_apo_31, linestyle='--')
	ax00.axhline(inner_peri_31)
	ax00.axhline(inner_apo_31, linestyle='--')
	ax00.axhline(outer_peri_21)
	ax00.axhline(outer_apo_21, linestyle='--')
	ax00.axhline(inner_peri_21)
	ax00.axhline(inner_apo_21, linestyle='--')
	ax01.axhline(outer_peri_31)
	ax01.axhline(outer_apo_31, linestyle='--')
	ax01.axhline(inner_peri_31)
	ax01.axhline(inner_apo_31, linestyle='--')
	ax01.axhline(outer_peri_21)
	ax01.axhline(outer_apo_21, linestyle='--')
	ax01.axhline(inner_peri_21)
	ax01.axhline(inner_apo_21, linestyle='--')

	ax00.set_ylim(2, 4)
	ax00.set_xlim(2, 0)
	ax00.set_ylim(rmin, rmax)
	ax01.set_xlim(-np.pi, np.pi)
	ax01.set_ylim(rmin, rmax)
	ax00.set_xlabel('Rel. # of Collisions')
	ax01.set_xlabel('Azimuth')
	ax00.set_ylabel('Cylindrical Distance [AU]')
	ax01.set_title(r'e$_{g}$ = 1/2 e$_{jup}$ (e1m2)')

	# Moderate eccentricity case
	ecc = ecc_jup
	ridx = 2
	rw_31 = res_width_jup(ridx, ecc_jup=ecc)*2
	fecc_31 = e_forced(res_dist[ridx], ecc)
	outer_peri_31 = (res_dist[ridx]+rw_31)*(1+fecc_31)
	outer_apo_31 = (res_dist[ridx]+rw_31)*(1-fecc_31)
	inner_peri_31 = (res_dist[ridx]-rw_31)*(1+fecc_31)
	inner_apo_31 = (res_dist[ridx]-rw_31)*(1-fecc_31)
	ridx = 5
	rw_21 = res_width_jup(ridx, ecc_jup=ecc)*2
	fecc_21 = e_forced(res_dist[ridx], ecc)
	outer_peri_21 = (res_dist[ridx]+rw_21)*(1+fecc_21)
	outer_apo_21 = (res_dist[ridx]+rw_21)*(1-fecc_21)
	inner_peri_21 = (res_dist[ridx]-rw_21)*(1+fecc_21)
	inner_apo_21 = (res_dist[ridx]-rw_21)*(1-fecc_21)

	ax10 = fig.add_subplot(gs[1, 0])
	ax11 = fig.add_subplot(gs[1, 1])
	ax11.set_yticks([])
	ax10.invert_xaxis()

	ax10.plot(coll_hist_e, coll_bins_e, drawstyle='steps')
	theta_jup = np.arctan2(pl_e['pos'][0][1], pl_e['pos'][0][0])
	ax11.axvline(theta_jup, color='k')
	ax11.scatter(theta_e, r_e, s=s*10)

	ax10.axhline(outer_peri_31)
	ax10.axhline(outer_apo_31, linestyle='--')
	ax10.axhline(inner_peri_31)
	ax10.axhline(inner_apo_31, linestyle='--')
	ax10.axhline(outer_peri_21)
	ax10.axhline(outer_apo_21, linestyle='--')
	ax10.axhline(inner_peri_21)
	ax10.axhline(inner_apo_21, linestyle='--')
	ax11.axhline(outer_peri_31)
	ax11.axhline(outer_apo_31, linestyle='--')
	ax11.axhline(inner_peri_31)
	ax11.axhline(inner_apo_31, linestyle='--')
	ax11.axhline(outer_peri_21)
	ax11.axhline(outer_apo_21, linestyle='--')
	ax11.axhline(inner_peri_21)
	ax11.axhline(inner_apo_21, linestyle='--')

	ax10.set_ylim(2, 4)
	ax10.set_xlim(2, 0)
	ax11.set_ylim(rmin, rmax)
	ax11.set_xlim(-np.pi, np.pi)
	ax11.set_ylim(rmin, rmax)
	ax10.set_xlabel('Rel. # of Collisions')
	ax11.set_xlabel('Azimuth')
	ax10.set_ylabel('Cylindrical Distance [AU]')
	ax11.set_title(r'e$_{g}$ = e$_{jup}$ (e2m2)')

	# High eccentricity case
	ecc = ecc_jup*2
	ridx = 2
	rw_31 = res_width_jup(ridx, ecc_jup=ecc)*2
	fecc_31 = e_forced(res_dist[ridx], ecc)
	outer_peri_31 = (res_dist[ridx]+rw_31)*(1+fecc_31)
	outer_apo_31 = (res_dist[ridx]+rw_31)*(1-fecc_31)
	inner_peri_31 = (res_dist[ridx]-rw_31)*(1+fecc_31)
	inner_apo_31 = (res_dist[ridx]-rw_31)*(1-fecc_31)
	ridx = 5
	rw_21 = res_width_jup(ridx, ecc_jup=ecc)*2
	fecc_21 = e_forced(res_dist[ridx], ecc)
	outer_peri_21 = (res_dist[ridx]+rw_21)*(1+fecc_21)
	outer_apo_21 = (res_dist[ridx]+rw_21)*(1-fecc_21)
	inner_peri_21 = (res_dist[ridx]-rw_21)*(1+fecc_21)
	inner_apo_21 = (res_dist[ridx]-rw_21)*(1-fecc_21)

	ax20 = fig.add_subplot(gs[2, 0])
	ax21 = fig.add_subplot(gs[2, 1])
	ax21.set_yticks([])
	ax20.invert_xaxis()

	ax20.plot(coll_hist_e2, coll_bins_e2, drawstyle='steps')
	theta_jup = np.arctan2(pl_e2['pos'][0][1], pl_e2['pos'][0][0])
	ax21.axvline(theta_jup, color='k')
	ax21.scatter(theta_e2, r_e2, s=s*10)

	ax20.axhline(outer_peri_31)
	ax20.axhline(outer_apo_31, linestyle='--')
	ax20.axhline(inner_peri_31)
	ax20.axhline(inner_apo_31, linestyle='--')
	ax20.axhline(outer_peri_21)
	ax20.axhline(outer_apo_21, linestyle='--')
	ax20.axhline(inner_peri_21)
	ax20.axhline(inner_apo_21, linestyle='--')
	ax21.axhline(outer_peri_31)
	ax21.axhline(outer_apo_31, linestyle='--')
	ax21.axhline(inner_peri_31)
	ax21.axhline(inner_apo_31, linestyle='--')
	ax21.axhline(outer_peri_21)
	ax21.axhline(outer_apo_21, linestyle='--')
	ax21.axhline(inner_peri_21)
	ax21.axhline(inner_apo_21, linestyle='--')

	ax20.set_ylim(2, 4)
	ax20.set_xlim(2, 0)
	ax21.set_ylim(rmin, rmax)
	ax21.set_xlim(-np.pi, np.pi)
	ax21.set_ylim(rmin, rmax)
	ax20.set_xlabel('Rel. # of Collisions')
	ax21.set_xlabel('Azimuth')
	ax20.set_ylabel('Cylindrical Distance [AU]')
	ax21.set_title(r'$e_{g}$ = 2 e$_{jup}$ (e3m2)')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_polar_m():
	file_str = 'figures/coll_polar_m.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	rmin, rmax = 2, 4

	fig = plt.figure(figsize=(12,14), constrained_layout=True)
	widths = [1, 4]
	heights = [1, 1, 1]
	gs = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)

	# Low mass case
	mass = 0.5*m_jup
	ridx = 2
	rw_31 = res_width_jup(ridx, m_jup=mass)*2
	fecc_31 = e_forced(res_dist[ridx], ecc_jup)
	outer_peri_31 = (res_dist[ridx]+rw_31)*(1+fecc_31)
	outer_apo_31 = (res_dist[ridx]+rw_31)*(1-fecc_31)
	inner_peri_31 = (res_dist[ridx]-rw_31)*(1+fecc_31)
	inner_apo_31 = (res_dist[ridx]-rw_31)*(1-fecc_31)
	ridx = 5
	rw_21 = res_width_jup(ridx, m_jup=mass)*2
	fecc_21 = e_forced(res_dist[ridx], ecc_jup)
	outer_peri_21 = (res_dist[ridx]+rw_21)*(1+fecc_21)
	outer_apo_21 = (res_dist[ridx]+rw_21)*(1-fecc_21)
	inner_peri_21 = (res_dist[ridx]-rw_21)*(1+fecc_21)
	inner_apo_21 = (res_dist[ridx]-rw_21)*(1-fecc_21)

	ax00 = fig.add_subplot(gs[0, 0])
	ax01 = fig.add_subplot(gs[0, 1])
	ax01.set_yticks([])
	ax00.invert_xaxis()

	ax00.plot(coll_hist_m1, coll_bins_m1, drawstyle='steps')
	theta_jup = np.arctan2(pl_m1['pos'][0][1], pl_m1['pos'][0][0])
	ax01.axvline(theta_jup, color='k')
	ax01.scatter(theta_m1, r_m1, s=s*10)

	ax00.axhline(outer_peri_31)
	ax00.axhline(outer_apo_31, linestyle='--')
	ax00.axhline(inner_peri_31)
	ax00.axhline(inner_apo_31, linestyle='--')
	ax00.axhline(outer_peri_21)
	ax00.axhline(outer_apo_21, linestyle='--')
	ax00.axhline(inner_peri_21)
	ax00.axhline(inner_apo_21, linestyle='--')
	ax01.axhline(outer_peri_31)
	ax01.axhline(outer_apo_31, linestyle='--')
	ax01.axhline(inner_peri_31)
	ax01.axhline(inner_apo_31, linestyle='--')
	ax01.axhline(outer_peri_21)
	ax01.axhline(outer_apo_21, linestyle='--')
	ax01.axhline(inner_peri_21)
	ax01.axhline(inner_apo_21, linestyle='--')

	ax00.set_ylim(2, 4)
	ax00.set_xlim(2, 0)
	ax00.set_ylim(rmin, rmax)
	ax01.set_xlim(-np.pi, np.pi)
	ax01.set_ylim(rmin, rmax)
	ax00.set_xlabel('Rel. # of Collisions')
	ax01.set_xlabel('Azimuth')
	ax00.set_ylabel('Cylindrical Distance [AU]')
	ax01.set_title(r'M$_{g}$ = 1/2 M$_{jup}$ (e2m1)')

	# Moderate mass case
	mass = m_jup
	ridx = 2
	rw_31 = res_width_jup(ridx, m_jup=mass)*2
	fecc_31 = e_forced(res_dist[ridx], ecc_jup)
	outer_peri_31 = (res_dist[ridx]+rw_31)*(1+fecc_31)
	outer_apo_31 = (res_dist[ridx]+rw_31)*(1-fecc_31)
	inner_peri_31 = (res_dist[ridx]-rw_31)*(1+fecc_31)
	inner_apo_31 = (res_dist[ridx]-rw_31)*(1-fecc_31)
	ridx = 5
	rw_21 = res_width_jup(ridx, m_jup=mass)*2
	fecc_21 = e_forced(res_dist[ridx], ecc_jup)
	outer_peri_21 = (res_dist[ridx]+rw_21)*(1+fecc_21)
	outer_apo_21 = (res_dist[ridx]+rw_21)*(1-fecc_21)
	inner_peri_21 = (res_dist[ridx]-rw_21)*(1+fecc_21)
	inner_apo_21 = (res_dist[ridx]-rw_21)*(1-fecc_21)

	ax10 = fig.add_subplot(gs[1, 0])
	ax11 = fig.add_subplot(gs[1, 1])
	ax11.set_yticks([])
	ax10.invert_xaxis()

	ax10.plot(coll_hist_e, coll_bins_e, drawstyle='steps')
	theta_jup = np.arctan2(pl_e['pos'][0][1], pl_e['pos'][0][0])
	ax11.axvline(theta_jup, color='k')
	ax11.scatter(theta_e, r_e, s=s*10)

	ax10.axhline(outer_peri_31)
	ax10.axhline(outer_apo_31, linestyle='--')
	ax10.axhline(inner_peri_31)
	ax10.axhline(inner_apo_31, linestyle='--')
	ax10.axhline(outer_peri_21)
	ax10.axhline(outer_apo_21, linestyle='--')
	ax10.axhline(inner_peri_21)
	ax10.axhline(inner_apo_21, linestyle='--')
	ax11.axhline(outer_peri_31)
	ax11.axhline(outer_apo_31, linestyle='--')
	ax11.axhline(inner_peri_31)
	ax11.axhline(inner_apo_31, linestyle='--')
	ax11.axhline(outer_peri_21)
	ax11.axhline(outer_apo_21, linestyle='--')
	ax11.axhline(inner_peri_21)
	ax11.axhline(inner_apo_21, linestyle='--')

	ax10.set_ylim(2, 4)
	ax10.set_xlim(2, 0)
	ax11.set_ylim(rmin, rmax)
	ax11.set_xlim(-np.pi, np.pi)
	ax11.set_ylim(rmin, rmax)
	ax10.set_xlabel('Rel. # of Collisions')
	ax11.set_xlabel('Azimuth')
	ax10.set_ylabel('Cylindrical Distance [AU]')
	ax11.set_title(r'M$_{g}$ = M$_{jup}$ (e2m2)')

	# High mass case
	mass = m_jup*2
	ridx = 2
	rw_31 = res_width_jup(ridx, m_jup=mass)*2
	fecc_31 = e_forced(res_dist[ridx], ecc_jup)
	outer_peri_31 = (res_dist[ridx]+rw_31)*(1+fecc_31)
	outer_apo_31 = (res_dist[ridx]+rw_31)*(1-fecc_31)
	inner_peri_31 = (res_dist[ridx]-rw_31)*(1+fecc_31)
	inner_apo_31 = (res_dist[ridx]-rw_31)*(1-fecc_31)
	ridx = 5
	rw_21 = res_width_jup(ridx, m_jup=mass)*2
	fecc_21 = e_forced(res_dist[ridx], ecc_jup)
	outer_peri_21 = (res_dist[ridx]+rw_21)*(1+fecc_21)
	outer_apo_21 = (res_dist[ridx]+rw_21)*(1-fecc_21)
	inner_peri_21 = (res_dist[ridx]-rw_21)*(1+fecc_21)
	inner_apo_21 = (res_dist[ridx]-rw_21)*(1-fecc_21)

	ax20 = fig.add_subplot(gs[2, 0])
	ax21 = fig.add_subplot(gs[2, 1])
	ax21.set_yticks([])
	ax20.invert_xaxis()

	ax20.plot(coll_hist_m2, coll_bins_m2, drawstyle='steps')
	theta_jup = np.arctan2(pl_m2['pos'][0][1], pl_m2['pos'][0][0])
	ax21.axvline(theta_jup, color='k')
	ax21.scatter(theta_m2, r_m2, s=s*10)

	ax20.axhline(outer_peri_31)
	ax20.axhline(outer_apo_31, linestyle='--')
	ax20.axhline(inner_peri_31)
	ax20.axhline(inner_apo_31, linestyle='--')
	ax20.axhline(outer_peri_21)
	ax20.axhline(outer_apo_21, linestyle='--')
	ax20.axhline(inner_peri_21)
	ax20.axhline(inner_apo_21, linestyle='--')
	ax21.axhline(outer_peri_31)
	ax21.axhline(outer_apo_31, linestyle='--')
	ax21.axhline(inner_peri_31)
	ax21.axhline(inner_apo_31, linestyle='--')
	ax21.axhline(outer_peri_21)
	ax21.axhline(outer_apo_21, linestyle='--')
	ax21.axhline(inner_peri_21)
	ax21.axhline(inner_apo_21, linestyle='--')

	ax20.set_ylim(2, 4)
	ax20.set_xlim(2, 0)
	ax21.set_ylim(rmin, rmax)
	ax21.set_xlim(-np.pi, np.pi)
	ax21.set_ylim(rmin, rmax)
	ax20.set_xlabel('Rel. # of Collisions')
	ax21.set_xlabel('Azimuth')
	ax20.set_ylabel('Cylindrical Distance [AU]')
	ax21.set_title(r'M$_{g}$ = 2 M$_{jup}$ (e2m3)')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def bump_dip_diag():
	file_str = 'figures/bump_dip_diag.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	res_idx = 2
	ngrid = 50
	e_jup_vals = np.logspace(-3.5, np.log10(0.5), ngrid)
	m_jup_vals = np.logspace(-1, 1, ngrid)

	is_bump_21 = np.zeros((ngrid, ngrid))
	for idx in range(ngrid):
	    for idx1 in range(ngrid):
	        a_res = res_dist[res_idx]
	        rw = res_width_jup(res_idx, ecc_jup=e_jup_vals[idx], \
	                              m_jup=(m_jup_vals[idx1]*u.M_jup).to(u.M_sun).value)*2
	        fecc = e_forced(a_res, e_jup_vals[idx])
	        apo_inner = (a_res - rw)*(1 + fecc)
	        peri_outer = (a_res + rw)*(1 - fecc)
	        
	        if (apo_inner - peri_outer)/rw > 0:
	            is_bump_21[idx][idx1] = 1.0
	            
	res_idx = 5
	e_jup_vals = np.logspace(-3.5, np.log10(0.5), ngrid)
	m_jup_vals = np.logspace(-1, 1, ngrid)

	is_bump_31 = np.zeros((ngrid, ngrid))
	for idx in range(ngrid):
	    for idx1 in range(ngrid):
	        a_res = res_dist[res_idx]
	        rw = res_width_jup(res_idx, ecc_jup=e_jup_vals[idx], \
	                              m_jup=(m_jup_vals[idx1]*u.M_jup).to(u.M_sun).value)*2
	        fecc = e_forced(a_res, e_jup_vals[idx])
	        apo_inner = (a_res - rw)*(1 + fecc)
	        peri_outer = (a_res + rw)*(1 - fecc)
	        
	        if (apo_inner - peri_outer)/rw > 0:
	            is_bump_31[idx][idx1] = 1.0

	fig, axes = plt.subplots(figsize=(5, 5))
	axes.contour(e_jup_vals, m_jup_vals, np.transpose(is_bump_21), 0, colors='k')
	axes.contourf(e_jup_vals, m_jup_vals, np.transpose(is_bump_21), 1, colors='none', \
	            hatches=[' ', '/'])

	axes.contour(e_jup_vals, m_jup_vals, np.transpose(is_bump_31), 0, colors='k')
	axes.contourf(e_jup_vals, m_jup_vals, np.transpose(is_bump_31), 1, colors='none', \
	            hatches=[' ', '\\'])
	axes.set_xlabel('Eccentricity of Perturber')
	axes.set_ylabel(r'Mass of Perturber [M$_{jup}$]')
	axes.set_xscale('log')
	axes.set_yscale('log')

	mvals = [0.5, 1, 2, 1 , 1]
	evals = [0.048, 0.048, 0.048, 0.024, 0.096]
	axes.scatter(evals, mvals, color='r', s=100, zorder=3)

	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
	axes.text(0.05, 0.95, 'Dip at 2:1 and 3:1', transform=axes.transAxes, fontsize=14,
	        verticalalignment='top', bbox=props)

	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
	axes.text(0.25, 0.15, 'Dip at 2:1, \nBump at 3:1', transform=axes.transAxes, fontsize=14,
	        verticalalignment='top', bbox=props)

	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
	axes.text(0.75, 0.25, 'Bump at 2:1, \nBump at 3:1', transform=axes.transAxes, fontsize=14,
	        verticalalignment='top', bbox=props)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def bump_dip_diag2():
	file_str = 'figures/bump_dip_diag.' + fmt
	if not clobber and os.path.exists(file_str):
		return
	            
	res_idx = 5
	ngrid = 50
	e_jup_vals = np.logspace(-3.5, np.log10(0.5), ngrid)
	m_jup_vals = np.logspace(-1, 1, ngrid)

	bump_21 = np.zeros((ngrid, ngrid))
	strength_21 = np.zeros((ngrid, ngrid))
	for idx in range(ngrid):
	    for idx1 in range(ngrid):
	        a_res = res_dist[res_idx]
	        rw = res_width_jup(res_idx, ecc_jup=e_jup_vals[idx], \
	                              m_jup=(m_jup_vals[idx1]*u.M_jup).to(u.M_sun).value)*2
	        fecc = e_forced(a_res, e_jup_vals[idx])
	        apo_inner = (a_res - rw)*(1 + fecc)
	        peri_outer = (a_res + rw)*(1 - fecc)
	        
	        bump_21[idx][idx1] = (apo_inner - peri_outer)/rw
        	strength_21[idx][idx1] = m_jup_vals[idx1]*e_jup_vals[idx]**2

	strength_21 /= strength_21[-1][-1]

	fig, axes = plt.subplots(figsize=(8,8))
	cmap = cm.get_cmap('viridis', 8)
	axes.contour(e_jup_vals, m_jup_vals, np.transpose(bump_21), levels=[-1, 1], colors='k', linestyles='--')
	cax = axes.pcolormesh(e_jup_vals, m_jup_vals, np.flipud(np.rot90(strength_21)), \
	                         norm=colors.LogNorm(vmin=1e-8, vmax=1), cmap=cmap)
	cbar = plt.colorbar(cax)
	cbar.set_label('Relative Strength of Resonance')
	axes.set_xscale('log')
	axes.set_yscale('log')
	axes.set_xlabel('Eccentricity of Perturber')
	axes.set_ylabel(r'Mass of Perturber [M$_{jup}$]')
	axes.set_title('2:1 MMR')

	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
	axes.text(0.1, 0.85, 'Central Dip', transform=axes.transAxes, fontsize=14,
	        verticalalignment='top', bbox=props)
	props = dict(boxstyle='round', facecolor='white', alpha=1.0)
	axes.text(0.65, 0.1, 'Central Bump', transform=axes.transAxes, fontsize=14,
	        verticalalignment='top', bbox=props)

	mvals = [0.5, 1, 2, 1 , 1]
	evals = [0.048, 0.048, 0.048, 0.024, 0.096]
	axes.scatter(evals, mvals, color='k', s=100, zorder=3)

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_xy():
	file_str = 'figures/xy.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	# Rotate so that jupiter is at theta =0
	rad_peri = 100
	theta_jup_e1 = np.arctan2(pl_e1['pos'][0][1], pl_e1['pos'][0][0])
	x_e1_rot = x_e1*np.cos(-theta_jup_e1) - y_e1*np.sin(-theta_jup_e1)
	y_e1_rot = x_e1*np.sin(-theta_jup_e1) + y_e1*np.cos(-theta_jup_e1)
	x_peri_e1, y_peri_e1 = rad_peri*np.cos(theta_jup_e1), rad_peri*np.sin(theta_jup_e1)
	theta_jup_e = np.arctan2(pl_e['pos'][0][1], pl_e['pos'][0][0])
	x_e_rot = x_e*np.cos(-theta_jup_e) - y_e*np.sin(-theta_jup_e)
	y_e_rot = x_e*np.sin(-theta_jup_e) + y_e*np.cos(-theta_jup_e)
	x_peri_e, y_peri_e = rad_peri*np.cos(theta_jup_e), rad_peri*np.sin(theta_jup_e)
	theta_jup_e2 = np.arctan2(pl_e2['pos'][0][1], pl_e2['pos'][0][0])
	x_e2_rot = x_e2*np.cos(-theta_jup_e2) - y_e2*np.sin(-theta_jup_e2)
	y_e2_rot = x_e2*np.sin(-theta_jup_e2) + y_e2*np.cos(-theta_jup_e2)
	x_peri_e2, y_peri_e2 = rad_peri*np.cos(theta_jup_e2), rad_peri*np.sin(theta_jup_e2)

	fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16,5), nrows=1, ncols=3, sharex=True, sharey=True)
	ax1.set_aspect('equal')
	ax1.scatter(x_e1_rot, y_e1_rot, s=s)
	ax1.scatter(x_e1_rot[0], y_e1_rot[0], color='r')
	ax1.plot([0, x_peri_e1], [0, y_peri_e1], linestyle='--')
	ax1.set_xlabel('X [AU]')
	ax1.set_ylabel('Y [AU]')
	ax1.set_title(r'e$_{pl}$ = 1/2 e$_{jup}$ (e1m2)')
	ax1.set_xlim(-5.5, 5.5)
	ax1.set_ylim(-5.5, 5.5)
	ax2.set_aspect('equal')
	ax2.scatter(x_e_rot, y_e_rot, s=s)
	ax2.scatter(x_e_rot[0], y_e_rot[0], color='r')
	ax2.plot([0, x_peri_e], [0, y_peri_e], linestyle='--')
	ax2.set_xlabel('X [AU]')
	ax2.set_ylabel('Y [AU]')
	ax2.set_title(r'e$_{pl}$ = e$_{jup}$ (e2m2)')
	ax3.set_aspect('equal')
	ax3.scatter(x_e2_rot, y_e2_rot, s=s)
	ax3.scatter(x_e2_rot[0], y_e2_rot[0], color='r')
	ax3.plot([0, x_peri_e2], [0, y_peri_e2], linestyle='--')
	ax3.set_xlabel('X [AU]')
	ax3.set_ylabel('Y [AU]')
	ax3.set_title(r'e$_{pl}$ = 2 e$_{jup}$ (e3m2)')
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_ae():
	file_str = 'figures/ae.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	# Libration width varies with eccentricity so it doesn't make much sense
	# to do show_widths=True here
	fig, ax = plt.subplots(figsize=(8,12), nrows=3, ncols=1, sharex=True, sharey=True)
	ax[0].scatter(pl_e1['a'], pl_e1['e'], s=s)
	ax[0].set_ylim(-0.001, 0.2)
	plot_res(ax[0], all=True)
	ax[0].set_ylabel('Eccentricity')
	ax[0].set_title(r'e$_{g}$ = 1/2 e$_{jup}$ (e1m2)')

	ax[1].scatter(pl_e['a'], pl_e['e'], s=s)
	plot_res(ax[1], all=True)
	ax[1].set_ylabel('Eccentricity')
	ax[1].set_title(r'e$_{g}$ = e$_{jup}$ (e2m2)')

	ax[2].scatter(pl_e2['a'], pl_e2['e'], s=s)
	plot_res(ax[2], all=True)
	ax[2].set_ylabel('Eccentricity')
	ax[2].set_title(r'e$_{g}$ = 2 e$_{jup}$ (e3m2)')

	ax[2].set_xlabel('Semimajor Axis [AU')

	# sharey=true hides the tick labels
	ax[2].yaxis.set_tick_params(labelleft=True)
	ax[2].set_xlim(2, 4)
	#plt.tight_layout(h_pad=0)
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_long_ph():
	file_str = 'figures/long_ph.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	fig, ax = plt.subplots(figsize=(16,6), nrows=1, ncols=3, sharex=True, sharey=True)
	ax[0].set_ylim(2, 4)
	ax[0].set_xlim(-np.pi, np.pi)
	ax[0].scatter((pl_e1['asc_node'] + pl_e1['omega'] + np.pi)%(2*np.pi) - np.pi, pl_e1['a'], s=s)
	plot_res(ax[0], vertical=False, all=True)
	ax[0].set_title(label=r'e$_{g}$ = 1/2 e$_{jup}$ (e1m2)')
	ax[1].set_xlabel('Longitude of Perihelion')
	ax[0].set_ylabel('Semimajor Axis [AU]')

	ax[1].scatter((pl_e['asc_node'] + pl_e['omega'] + np.pi)%(2*np.pi) - np.pi, pl_e['a'], s=s)
	plot_res(ax[1], vertical=False, all=True)
	ax[1].set_title(label=r'e$_{g}$ = e$_{jup}$ (e2m2)')

	ax[2].scatter((pl_e2['asc_node'] + pl_e2['omega']+ np.pi)%(2*np.pi) - np.pi, pl_e2['a'], s=s)
	plot_res(ax[2], vertical=False, all=True)
	ax[2].set_title(label=r'e$_{g}$ = 2 e$_{jup}$ (e3m2)')
	fig.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_a():
	file_str = 'figures/coll_hist_a.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	coll_bins_a_e1, coll_pdf_a_e1 = kde(coll_e1['a1'])
	coll_bins_a_e, coll_pdf_a_e = kde(coll_e['a1'])
	coll_bins_a_e2, coll_pdf_a_e2 = kde(coll_e2['a1'])

	fig, axes = plt.subplots(figsize=(8,12), nrows=3, ncols=1, sharex=True)
	axes[0].plot(coll_bins_a_e1, coll_pdf_a_e1, drawstyle='steps')
	plot_res(axes[0])
	axes[0].set_ylabel('Rel. # of Collisions')
	axes[0].set_title(label=r'e$_{g}$ = 1/2 e$_{jup}$ (e1m2)')
	axes[1].plot(coll_bins_a_e, coll_pdf_a_e, drawstyle='steps')
	plot_res(axes[1])
	axes[1].set_ylabel('Rel. # of Collisions')
	axes[1].set_title(label=r'e$_{g}$ = e$_{jup}$ (e2m2)')
	axes[2].plot(coll_bins_a_e2, coll_pdf_a_e2, drawstyle='steps')
	plot_res(axes[2])
	axes[2].set_ylabel('Rel. # of Collisions')
	axes[2].set_title(label=r'e$_{g}$ = 2 e$_{jup}$ (e3m2)')
	axes[2].set_xlabel('Semimajor Axis [AU]')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_r():
	file_str = 'figures/coll_hist_r.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	coll_bins_a_e1, coll_pdf_a_e1 = kde(coll_e1['dist1'])
	coll_bins_a_e, coll_pdf_a_e = kde(coll_e['dist1'])
	coll_bins_a_e2, coll_pdf_a_e2 = kde(coll_e2['dist1'])

	# Also plot curves with bodies in resonance excluded
	a311, a312 = 2.49, 2.51
	a211, a212 = 3.2, 3.35

	c = coll_e1
	res_mask = np.logical_and(np.logical_and(~np.logical_and(c['a1'] > a311, c['a1'] < a312), \
                          ~np.logical_and(c['a1'] > a211, c['a1'] < a212)),
							np.logical_and(~np.logical_and(c['a2'] > a311, c['a2'] < a312), \
                          ~np.logical_and(c['a2'] > a211, c['a2'] < a212)))
	coll_bins_a_e1_ex, coll_pdf_a_e1_ex = kde(c['dist1'][res_mask])
	e1_norm = len(c[res_mask])/len(coll_e1)

	c = coll_e
	res_mask = np.logical_and(np.logical_and(~np.logical_and(c['a1'] > a311, c['a1'] < a312), \
                          ~np.logical_and(c['a1'] > a211, c['a1'] < a212)),
							np.logical_and(~np.logical_and(c['a2'] > a311, c['a2'] < a312), \
                          ~np.logical_and(c['a2'] > a211, c['a2'] < a212)))
	coll_bins_a_e_ex, coll_pdf_a_e_ex = kde(c['dist1'][res_mask])
	e_norm = len(c[res_mask])/len(coll_e)

	c = coll_e2
	res_mask = np.logical_and(np.logical_and(~np.logical_and(c['a1'] > a311, c['a1'] < a312), \
                          ~np.logical_and(c['a1'] > a211, c['a1'] < a212)),
							np.logical_and(~np.logical_and(c['a2'] > a311, c['a2'] < a312), \
                          ~np.logical_and(c['a2'] > a211, c['a2'] < a212)))
	coll_bins_a_e2_ex, coll_pdf_a_e2_ex = kde(c['dist1'][res_mask])
	e2_norm = len(c[res_mask])/len(coll_e2)

	fig, axes = plt.subplots(figsize=(8,12), nrows=3, ncols=1, sharex=True)
	axes[0].plot(coll_bins_a_e1, coll_pdf_a_e1, drawstyle='steps')
	axes[0].plot(coll_bins_a_e1_ex, coll_pdf_a_e1_ex*e1_norm, drawstyle='steps', linestyle='--')
	plot_res(axes[0])
	axes[0].set_ylabel('Rel. # of Collisions')
	axes[0].set_title(label=r'e$_{g}$ = 1/2 e$_{jup}$ (e1m2)')
	axes[1].plot(coll_bins_a_e, coll_pdf_a_e, drawstyle='steps')
	axes[1].plot(coll_bins_a_e_ex, coll_pdf_a_e_ex*e_norm, drawstyle='steps', linestyle='--')
	plot_res(axes[1])
	axes[1].set_ylabel('Rel. # of Collisions')
	axes[1].set_title(label=r'e$_{g}$ = e$_{jup}$ (e2m2)')
	axes[2].plot(coll_bins_a_e2, coll_pdf_a_e2, drawstyle='steps')
	axes[2].plot(coll_bins_a_e2_ex, coll_pdf_a_e2_ex*e2_norm, drawstyle='steps', linestyle='--')
	plot_res(axes[2])
	axes[2].set_ylabel('Rel. # of Collisions')
	axes[2].set_title(label=r'e$_{g}$ = 2 e$_{jup}$ (e3m2)')
	axes[2].set_xlabel('Cylindrical Distance [AU]')

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_polar_hk_plots():
	file_str = 'figures/polar_hk.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	a211, a212 = 3.16, 3.2
	a211a, a212a = 3.38, 3.43

	fig = plt.figure(figsize=(8, 8), constrained_layout=True)
	gs = fig.add_gridspec(2, 2)

	ax1 = fig.add_subplot(gs[0, :])
	theta1, theta2 = np.pi-0.1, np.pi+0.1
	theta1a, theta2a = 3*np.pi/2-0.1, 3*np.pi/2+0.1
	pl = pl_e
	x, y = pl['pos'][:,0], pl['pos'][:,1]
	theta_jup = np.arctan2(y[0], x[0])
	rad, theta = np.sqrt(x**2 + y**2), (np.arctan2(y, x) - theta_jup)%(2*np.pi)
	mask_21 = np.logical_and(np.logical_and(pl['a'] > a211, pl['a'] < a212), \
	                         np.logical_and(theta > theta1, theta < theta2))
	mask_21a = np.logical_and(np.logical_and(pl['a'] > a211a, pl['a'] < a212a), \
	                          np.logical_and(theta > theta1, theta < theta2))
	mask_a21 = np.logical_and(np.logical_and(pl['a'] > a211, pl['a'] < a212), \
	                         np.logical_and(theta > theta1a, theta < theta2a))
	mask_a21a = np.logical_and(np.logical_and(pl['a'] > a211a, pl['a'] < a212a), \
	                          np.logical_and(theta > theta1a, theta < theta2a))
	ax1.scatter(theta[~mask_21], rad[~mask_21], s=0.01)
	ax1.scatter(theta[mask_21], rad[mask_21], s=0.5, color='orange')
	ax1.scatter(theta[mask_21a], rad[mask_21a], s=0.5, color='orange')
	ax1.scatter(theta[mask_a21], rad[mask_a21], s=0.5, color='red')
	ax1.scatter(theta[mask_a21a], rad[mask_a21a], s=0.5, color='red')
	ax1.set_xlim(0, 2*np.pi)
	ax1.set_ylim(2.8, 3.8)
	ax1.set_xlabel('Azimuth')
	ax1.set_ylabel('Cylindrical Distance [AU]')

	curlypi = (pl['omega'] + pl['asc_node'])%(2*np.pi)
	hvals = pl['e']*np.cos(curlypi)
	kvals = pl['e']*np.sin(curlypi)

	hmin, hmax = -0.05, 0.05
	kmin, kmax = -0.05, 0.05
	hshift = 0.03
	ax2 = fig.add_subplot(gs[1, :-1])
	ax2.scatter(hvals, kvals, s=0.5)
	ax2.scatter(hvals[mask_21], kvals[mask_21], s=0.5, color='orange')
	ax2.scatter(hvals[mask_a21], kvals[mask_a21], s=0.5, color='red')
	ax2.set_xlabel('h')
	ax2.set_ylabel('k')
	ax2.set_title('Interior to Resonance')
	ax2.set_xlim(hmin + hshift, hmax + hshift)
	ax2.set_ylim(kmin, kmax)

	ax3 = fig.add_subplot(gs[1, -1])
	ax3.scatter(hvals, kvals, s=0.5)
	ax3.scatter(hvals[mask_21a], kvals[mask_21a], s=0.5, color='orange')
	ax3.scatter(hvals[mask_a21a], kvals[mask_a21a], s=0.5, color='red')
	ax3.set_xlabel('h')
	ax3.set_ylabel('k')
	ax3.set_title('Exterior to Resonance')
	ax3.set_xlim(hmin + hshift, hmax + hshift)
	ax3.set_ylim(kmin, kmax)

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

	# Cut out secularly aligned collisions from out_21
	cut_angle = 0.5

	curlypi_diff_e11 = np.fabs((coll_out_21_e1['omega1'] + coll_out_21_e1['Omega1'] + np.pi)%(2*np.pi) - np.pi)
	curlypi_diff_e12 = np.fabs((coll_out_21_e1['omega2'] + coll_out_21_e1['Omega2'] + np.pi)%(2*np.pi) - np.pi)
	curlypi_mask_e1 = np.logical_or(curlypi_diff_e11 > cut_angle, curlypi_diff_e12 > cut_angle)
	coll_bins_out_spread_21_e1, coll_pdf_out_spread_21_e1 = kde(coll_out_21_e1[curlypi_mask_e1]['dist2'])
	coll_pdf_out_spread_21_e1 *= len(coll_out_21_e1[curlypi_mask_e1])/len(coll_out_21_e1)

	curlypi_diff_e21 = np.fabs((coll_out_21_e2['omega1'] + coll_out_21_e2['Omega1'] + np.pi)%(2*np.pi) - np.pi)
	curlypi_diff_e22 = np.fabs((coll_out_21_e2['omega2'] + coll_out_21_e2['Omega2'] + np.pi)%(2*np.pi) - np.pi)
	curlypi_mask_e2 = np.logical_or(curlypi_diff_e21 > cut_angle, curlypi_diff_e22 > cut_angle)
	coll_bins_out_spread_21_e2, coll_pdf_out_spread_21_e2 = kde(coll_out_21_e2[curlypi_mask_e2]['dist2'])
	coll_pdf_out_spread_21_e2 *= len(coll_out_21_e2[curlypi_mask_e2])/len(coll_out_21_e2)

	fig, ax = plt.subplots(figsize=(16,6), nrows=1, ncols=2, sharex=True)
	ax[0].plot(coll_bins_in_21_e1, coll_pdf_in_21_e1, label=r'e$_{pl}$ = 1/2 e$_{jup}$')
	ax[0].plot(coll_bins_in_21_e2, coll_pdf_in_21_e2, label=r'e$_{pl}$ = 2 e$_{jup}$')
	c = next(ax[1]._get_lines.prop_cycler)['color']
	ax[1].plot(coll_bins_out_21_e1, coll_pdf_out_21_e1, color=c)
	ax[1].plot(coll_bins_out_spread_21_e1, coll_pdf_out_spread_21_e1, color=c, linestyle='--')
	c = next(ax[1]._get_lines.prop_cycler)['color']
	ax[1].plot(coll_bins_out_21_e2, coll_pdf_out_21_e2, color=c)
	ax[1].plot(coll_bins_out_spread_21_e2, coll_pdf_out_spread_21_e2, color=c, linestyle='--')
	ax[0].legend()
	ax[0].set_xlabel('Heliocentric Distance [AU]')
	ax[0].set_title(str(a21_in) + ' < a < ' + str(a21_out))
	ax[1].set_xlabel('Heliocentric Distance [AU]')
	ax[0].set_ylabel('dn/dr')
	ax[1].set_title(str(a21_in) + ' > a > ' + str(a21_out))

	plt.tight_layout()
	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_coll_hist_r_inout_res31():
	file_str = 'figures/coll_hist_r_inout_res31.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	a21_in, a21_out = 2.49, 2.51

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

	res_idx = 2
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

	res_idx = 5
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

def make_boley_rtheta():
	file_str = 'figures/boley_rtheta.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	data = np.genfromtxt('snapshot-100')
	xpos, ypos = data[:,1], data[:,2]
	r, theta = np.sqrt(xpos**2 + ypos**2), np.arctan2(ypos, xpos)

	fig, axes = plt.subplots(figsize=(12,6))
	axes.scatter(theta, r, s=0.1)
	axes.set_xlabel('Azimuth')
	axes.set_ylabel('Cylindrical Distance [AU]')
	axes.set_ylim(20, 100)
	axes.set_xlim(-np.pi, np.pi)

	# Planet 1
	#axes.axhline(32.3)
	#axes.axhline(39.1, linestyle='--', lw=1) # 4:3
	#axes.axhline(42.3, linestyle='--', lw=1) # 3:2
	#axes.axhline(51.3, linestyle='--', lw=1) # 2:1

	# Planet 2
	lw = 2
	alpha = 0.5
	axes.axhline(68.8, color='black', lw=lw)
	axes.axhline(56.8, color='black', linestyle='--', lw=lw, alpha=alpha) # 4:3
	axes.axhline(52.5, color='black', linestyle='--', lw=lw, alpha=alpha) # 3:2
	axes.axhline(43.3, color='black', linestyle='--', lw=lw, alpha=alpha) # 2:1
	axes.axhline(83.3, color='black', linestyle='--', lw=lw, alpha=alpha) # 3:4
	axes.axhline(90.1, color='black', linestyle='--', lw=lw, alpha=alpha) # 2:3

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def make_alma_profile():
	file_str = 'figures/alma_profiles.' + fmt
	if not clobber and os.path.exists(file_str):
		return

	r1, p1 = np.loadtxt('alma/e2m1.txt')
	r2, p2 = np.loadtxt('alma/e1m2.txt')
	r3, p3 = np.loadtxt('alma/e2m2.txt')
	r4, p4 = np.loadtxt('alma/e3m2.txt')
	r5, p5 = np.loadtxt('alma/e2m3.txt')

	r = [r1, r2, r3, r4, r5]
	p = [p1, p2, p3, p4, p5]

	fig = plt.figure(constrained_layout=True, figsize=(12,10))
	spec = fig.add_gridspec(ncols=3, nrows=3)

	spec_idx = [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]]
	sim_title = ['e2m3', 'e1m2', 'e2m2', 'e2m3', 'e2m1']
	for idx in range(len(spec_idx)):
	    i, j = spec_idx[idx][0], spec_idx[idx][1]
	    ax = fig.add_subplot(spec[i, j])
	    ax.plot(r[idx], p[idx])
	    # Mark locations of MMRs
	    ax.axvline(dist_to_ang(res_dist[2]), linestyle='--')
	    ax.axvline(dist_to_ang(res_dist[5]), linestyle='--')
	    ax.set_title(sim_title[idx])
	    if idx != 0 and idx != 2:
	    	ax.set_xlabel('R [arcsec]')
	    if idx != 2 and idx != 3:
	    	ax.set_ylabel('Brightness [Jy/beam]')
	    ax.set_xlim(0.2, 0.4)
	    ax.set_ylim(-0.001, 0.145)
	    
	plt.tight_layout()

	plt.savefig(file_str, format=fmt, bbox_inches='tight')

def gen_coll_PDF():
	c = coll_e1
	coll_bins_r_e1, coll_pdf_r_e1 = kde(c['dist1'])

	c = coll_e
	coll_bins_r_e, coll_pdf_r_e = kde(c['dist1'])

	c = coll_e2
	coll_bins_r_e2, coll_pdf_r_e2 = kde(c['dist1'])

	c = coll_m1
	coll_bins_r_m1, coll_pdf_r_m1 = kde(c['dist1'])

	c = coll_m2
	coll_bins_r_m2, coll_pdf_r_m2 = kde(c['dist1'])

	np.savetxt('e1m2_coll.txt', [coll_bins_r_e1, coll_pdf_r_e1])
	np.savetxt('e2m2_coll.txt', [coll_bins_r_e, coll_pdf_r_e])
	np.savetxt('e3m2_coll.txt', [coll_bins_r_e2, coll_pdf_r_e2])
	np.savetxt('e2m1_coll.txt', [coll_bins_r_m1, coll_pdf_r_m1])
	np.savetxt('e2m3_coll.txt', [coll_bins_r_m2, coll_pdf_r_m2])

#gen_coll_PDF()
#make_boley_rtheta()
make_alma_profile()
#make_coll_polar_e()
#make_coll_polar_m()
#bump_dip_diag2()
#make_xy()
#make_ae()
#make_long_ph()
#make_coll_hist_a()
#make_coll_hist_r()
#make_polar_hk_plots()
#make_coll_hist_e_and_m()
#make_coll_hist_hot_cold_a()
#make_coll_gf_vary()
#make_coll_hist_r_inout_res()
#make_coll_hist_r_inout_res31()
#make_wander_res_scale()
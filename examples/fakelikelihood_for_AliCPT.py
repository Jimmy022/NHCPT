#/usr/bin/env python
# --
# quicklens/examples/plot_lens_reconstruction_noise_levels.py
# --
# calculates and plots the lensing reconstruction noise levels
# for an idealized experiment with symmetric beam and white
# pixel noise, then plots. calculation is done in both the
# full-sky and flat-sky limits for comparison.

import numpy as np
import pylab as pl

import quicklens as ql

# calculation parameters.
lmax       = 1000 # maximum multipole for T, E, B and \phi.
nx         = 512  # number of pixels for flat-sky calc.
dx         = 1./60./180.*np.pi # pixel width in radians.

nlev_p1     = 40.    # polarization noise level, in uK.arcmin.
nlev_p2     = 40.

#l_knee1     = 60      # Parameters for computing 1/f noise
#l_knee2     = 60
#alpha_knee1  = 2.5
#alpha_knee2  = 3.0

ATT       = 354.73 * 0   # Parameters for computing foreground residual,here we igore the foreground residual
AEE       = 4.547901 * 10**(-4) * 0
ABB       = 2.3831 * 10**(-4) * 0

beam_fwhm1  = 19.0  # Gaussian beam full-width-at-half-maximum.
beam_fwhm2  = 11.0

fsky       = 0.4


data=np.loadtxt('data1000.txt')

bl1         = ql.spec.bl(beam_fwhm1, lmax) # transfer function.
bl2         = ql.spec.bl(beam_fwhm2, lmax)

pix         = ql.maps.pix(nx,dx)

# signal spectra
sltt       = data[:,1]
slee       = data[:,2]
slbb       = data[:,3]
slte       = data[:,4]
sldd       = data[:,5]
sldt       = data[:,6]
f1         = data[:,7]  #95 GHZ 1/f noise parameter for EE, l_knee=60
f2         = data[:,8]  #150 GHZ 1/f noise parameter for EE, l_knee=60
f1_2         = data[:,9]  #95 GHZ 1/f noise parameter for EE, l_knee=50
f2_2         = data[:,10]  #150 GHZ 1/f noise parameter for EE, l_knee=50
f1_3         = data[:,11]  #95 GHZ 1/f noise parameter for EE, l_knee=40
f2_3         = data[:,12]  #150 GHZ 1/f noise parameter for EE, l_knee=40
f1_4         = data[:,13]  #95 GHZ 1/f noise parameter for EE, l_knee=30
f2_4         = data[:,14]  #150 GHZ 1/f noise parameter for EE, l_knee=30
fg1         = data[:,15]  #foreground dust power law for TT
fg2         = data[:,16]  #foreground dust power law for EE
fg3         = data[:,17]  #foreground dust power law for BB
zero       = np.zeros(lmax+1)


# noise spectra
ls         = np.arange(0,lmax+1)

nlee1      = (np.pi/180./60.*nlev_p1)**2 / bl1**2
nltt1       = 0.5 * nlee1
nlee2      = (np.pi/180./60.*nlev_p2)**2 / bl2**2
nltt2       = 0.5 * nlee2

#nlee1      = nlee1 * (1.+f1)
#nlee2      = nlee2 * (1.+f2)

nlee       = 1./nlee1 + 1./nlee2
nlee       = 1./nlee

nlbb       = nlee

nltt       = 1./nltt1 + 1./nltt2
nltt       = 1./nltt

fltt       = ATT * fg1
flee       = AEE * fg2
flbb       = ABB * fg3

nltt       = nltt + fltt
nlee       = nlee + flee
nlbb       = nlbb + flbb


# signal spectra
sltt       = data[:,1]
slee       = data[:,2]
slbb       = data[:,3]
slte       = data[:,4]
sldd       = data[:,5]
sldt       = data[:,6]
zero       = np.zeros(lmax+1)

# signal+noise spectra
cltt       = sltt + nltt
clee       = slee + nlee
clte       = slte
clbb       = slbb + nlbb


# filter functions
flt        = np.zeros( lmax+1 ); flt[2:] = 1./cltt[2:]
fle        = np.zeros( lmax+1 ); fle[2:] = 1./clee[2:]
flb        = np.zeros( lmax+1 ); flb[2:] = 1./clbb[2:]

# intialize quadratic estimators
qest_TT    = ql.qest.lens.phi_TT(sltt)
qest_EE    = ql.qest.lens.phi_EE(slee)
qest_TE    = ql.qest.lens.phi_TE(slte)
qest_TB    = ql.qest.lens.phi_TB(slte)
qest_EB    = ql.qest.lens.phi_EB(slee)

# calculate the noise spectra.
watch = ql.util.stopwatch()
def calc_nlqq(qest, clXX, clXY, clYY, flX, flY):
    errs = np.geterr(); np.seterr(divide='ignore', invalid='ignore')
    
    print "[%s]"%watch.elapsed(), "calculating flat-sky noise level for estimator of type", type(qest)
    clqq_flatsky = qest.fill_clqq(ql.maps.cfft(nx,dx), clXX*flX*flX, clXY*flX*flY, clYY*flY*flY)
    resp_flatsky = qest.fill_resp(qest, ql.maps.cfft(nx, dx), flX, flY)
    nlqq_flatsky = clqq_flatsky / resp_flatsky**2
    
    print "[%s]"%watch.elapsed(), "calculating full-sky noise level for estimator of type", type(qest)
    clqq_fullsky = qest.fill_clqq(np.zeros(lmax+1, dtype=np.complex), clXX*flX*flX, clXY*flX*flY, clYY*flY*flY)
    resp_fullsky = qest.fill_resp(qest, np.zeros(lmax+1, dtype=np.complex), flX, flY)
    nlqq_fullsky = clqq_fullsky / resp_fullsky**2

    np.seterr(**errs)
    return nlqq_flatsky, nlqq_fullsky


nlpp_TT_flatsky, nlpp_TT_fullsky = calc_nlqq( qest_TT, cltt, cltt, cltt, flt, flt )
nlpp_EE_flatsky, nlpp_EE_fullsky = calc_nlqq( qest_EE, clee, clee, clee, fle, fle )
nlpp_TE_flatsky, nlpp_TE_fullsky = calc_nlqq( qest_TE, cltt, slte, clee, flt, fle )
nlpp_TB_flatsky, nlpp_TB_fullsky = calc_nlqq( qest_TB, cltt, zero, clbb, flt, flb )
nlpp_EB_flatsky, nlpp_EB_fullsky = calc_nlqq( qest_EB, clee, zero, clbb, fle, flb )

Rev=1/nlpp_TT_fullsky+1/nlpp_EE_fullsky+1/nlpp_TE_fullsky+1/nlpp_TB_fullsky+1/nlpp_EB_fullsky
MV_noise_fullsky=1/Rev

# make plot
t  = lambda l: (l*(l+1.))**2/(2.*np.pi)
fsk= (ls+1.)*fsky/(ls+1.)

sh=np.shape(ls)
d1=np.empty((sh[0],10))
d1[:,0]=ls
d1[:,1]=np.real(nltt)
d1[:,2]=np.real(nlee)
d1[:,3]=np.real(t(ls) * MV_noise_fullsky)
d1[:,4]=np.real(cltt)
d1[:,5]=np.real(clte)
d1[:,6]=np.real(clee)
d1[:,7]=np.real(ls*(ls+1)*MV_noise_fullsky)
d1[:,8]=np.real(sldt)
d1[:,9]=np.real(fsk)

np.savetxt("noise_file.txt",d1)


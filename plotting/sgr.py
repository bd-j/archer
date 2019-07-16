from astropy.io import fits
rcat = fits.getdata("rcat_V1.3_MSG.fits")
vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
zgal = np.clip(rcat["Z_gal"], -100, 100)
ygal = np.clip(rcat["Y_gal"], -60, 100)
feh = np.clip(rcat["feh"], -3.0, 0.1)

# --- Basic selections ---
good = ((rcat["logg"] < 3.5) & np.isfinite(rcat["Z_gal"]) &
        (rcat["FLAG"] == 0) & np.isfinite(vtot) & (vtot < 3000))
sgr = np.abs(rcat["Sgr_B"]) < 40
etot = rcat["E_tot_pot1"]
sel = good & (etot > -130000) & (etot < -50000)
sel = good & sgr

scatter(rcat[sel]["Sgr_l"], rcat[sel]["dist_MS"], c=rcat[sel]["Vrad"], alpha=0.3)
ylim(5, 140)
xlim(0, 360)
sel = good & (np.abs(rcat["Sgr_B"]) < 5)
scatter(rcat[sel]["Sgr_l"], rcat[sel]["dist_MS"], c=rcat[sel]["Vrad"], alpha=0.3)
ylim(5, 140)
xlim(0, 360)
scatter(rcat[sel]["Sgr_l"], rcat[sel]["dist_MS"], c=rcat[sel]["Sgr_b"], cmap="viridis", alpha=0.3, vmin=-13, vmax=13)
colorbar()
ylim(5, 140)
xlim(0, 360)


figure()
plot(rcat[sel]["R_gal"], rcat[sel]["Vr_gal"], "o", alpha=0.5)

figure()
scatter(rcat[sel]["Vr_gal"], rcat["Vtheta_gal"][sel], c=rcat["R_gal"][sel], marker="o", alpha=0.5, vmin=0, vmax=100, cmap="viridis")
xlim(-300, 300)
ylim(-400, 400)
colorbar()

cat = rcat[sel]

import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc
ceq = coord.ICRS(ra=cat['GaiaDR2_ra']*u.deg, dec=cat['GaiaDR2_dec']*u.deg,
                 distance=cat["dist_MS"] * u.kpc,
                 pm_ra_cosdec=cat['GaiaDR2_pmra']*u.mas/u.yr, pm_dec=cat['GaiaDR2_pmdec']*u.mas/u.yr,
                 radial_velocity=cat['Vrad']*u.km/u.s)
sgr = ceq.transform_to(gc.Sagittarius)

figure()
scatter(sgr.Lambda, sgr.distance, c=cat["Vr_gal"], marker='o', alpha=0.5, cmap="viridis", vmin=-300, vmax=0)
colorbar()

# !!!!!!
figure()
scatter(sgr.Lambda, cat["V_gsr"], c=cat["Vr_gal"], marker='o', alpha=0.5, cmap="viridis", vmin=-300, vmax=0)
xlabel("$\Lambda$")
ylabel("$v_{gsr}$")
ylim(-300, 350)

# !!!!!!!!
figure()
scatter(sgr.Lambda, cat["V_gsr"], c=sgr.distance, 
        marker='o', alpha=0.5, cmap="viridis", vmin=0, vmax=50)
xlabel("$\Lambda$")
ylabel("$v_{gsr}$")


figure()
scatter(sgr.Lambda, cat["Vrad"], c=cat["Vr_gal"], marker='o', alpha=0.5, cmap="viridis", vmin=-300, vmax=0)

figure()
scatter(sgr.Lambda, cat["V_gsr"], c=cat["FeH"], marker='o', alpha=0.5, cmap="viridis", vmin=-2, vmax=0.)
colorbar()

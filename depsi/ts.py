import os
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import LAMBDA
from ds import CRB, extract_blocks_idx, is_symetric_positive_definite
from scipy.linalg import block_diag
from scipy.optimize import differential_evolution, fmin, minimize
from scipy.stats import mode
from scipy.stats.distributions import chi2
from spams import spams_model
from utils import wrap, wrap_with_ambiguity


def ts_est_prep(arc_stm_list, ts_est_dir):
    """Preparation for time series estimation.

    Parameters
    ----------
    arc_stm_list : list of xr.Dataset
        List of arc stm datasets.
    ts_est_dir : str
        Directory for storing time series estimates datasets.

    Returns
    -------
    list of xr.Dataset
        List of prepared time series estimates datasets.
    """
    ts_stm_list = []
    for i in range(len(arc_stm_list)):
        stack_id = arc_stm_list[i].attrs["stack_id"]
        ts_filepath = os.path.join(ts_est_dir, "ts_stm_" + stack_id + ".zarr")
        if os.path.isdir(ts_filepath):
            print(f"Loading existing ts_stm for stack {stack_id} ...")
            ts_stm = xr.open_zarr(ts_filepath, consolidated=False,)
        else:
            print(f"Preparing ts_stm for stack {stack_id} ...")
            arc_stm = arc_stm_list[i]
            ts_stm = arc_stm.copy()
            nspace = ts_stm.sizes["space"]
            ntime = ts_stm.sizes["time"]
            ts_stm = ts_stm.assign(
                xP_init=(["space"], np.full((nspace,), np.nan)),
                xE_init=(["space"], np.full((nspace,), np.nan)),
                xI_init=(["space"], np.full((nspace,), np.nan)),
                tau_init=(["space"], np.full((nspace,), np.nan)),
                T_init=(["space"], np.full((nspace,), np.nan)),
                nobs_init=(["space"], np.full((nspace,), np.nan)),
                gamma_t_init=(["space"], np.full((nspace,), np.nan)),
                rmse_init=(["space"], np.full((nspace,), np.nan)),
                ahat_init=(["space", "time"], np.zeros((nspace, ntime))),
                ts_phi_abs_init=(["space", "time"], np.full((nspace, ntime), np.nan)),
                ts_pov_init=(["space", "time"], np.full((nspace, ntime), np.nan)),
                ehat_rad_init=(["space", "time"], np.full((nspace, ntime), np.nan)),
                ehat_mm_init=(["space", "time"], np.full((nspace, ntime), np.nan)),

                xP_group=(["space"], np.full((nspace,), np.nan)),
                xE_group=(["space"], np.full((nspace,), np.nan)),
                xI_group=(["space"], np.full((nspace,), np.nan)),
                tau_group=(["space"], np.full((nspace,), np.nan)),
                rmse_group=(["space"], np.full((nspace,), np.nan)),

                xP_final=(["space"], np.full((nspace,), np.nan)),
                xE_final=(["space"], np.full((nspace,), np.nan)),
                xI_final=(["space"], np.full((nspace,), np.nan)),
                tau_final=(["space"], np.full((nspace,), np.nan)),
                rmse_final=(["space"], np.full((nspace,), np.nan)),
                T_final=(["space"], np.full((nspace,), np.nan)),
                nobs_final=(["space"], np.full((nspace,), np.nan)),
                ahat_final=(["space", "time"], np.zeros((nspace, ntime))),
                ts_phi_abs_final=(["space", "time"], np.full((nspace, ntime), np.nan)),
                ts_pov_final=(["space", "time"], np.full((nspace, ntime), np.nan)),
                ehat_rad_final=(["space", "time"], np.full((nspace, ntime), np.nan)),
                ehat_mm_final=(["space", "time"], np.full((nspace, ntime), np.nan)),
                var_xP_final=(["space"], np.full((nspace,), np.nan)),
                var_xE_final=(["space"], np.full((nspace,), np.nan)),
                var_xI_final=(["space"], np.full((nspace,), np.nan)),
                var_xPxE_final=(["space"], np.full((nspace,), np.nan)),
                var_xPxI_final=(["space"], np.full((nspace,), np.nan)),
                var_xExI_final=(["space"], np.full((nspace,), np.nan)),
            )
            ts_stm.to_zarr(ts_filepath, mode="w", consolidated=False)
        ts_stm_list.append(ts_stm)

    return ts_stm_list


def ts_est_init(
    ts_stm_list,
    ts_est_dir,
    ds_ids,
    meteo_data,
    start_date,
    end_date,
    init_params,
    param_bounds=None,
    tau_range=None,
    use_global=True,
    unwrapping_method="lsa",
    pe_dir=None,
    fitting_flag=1,
):
    """Initial time series estimation for each point.

    Parameters
    ----------
    ts_stm_list : list of xr.Dataset
        List of time series datasets.
    ts_est_dir : str
        Directory for storing time series estimates.
    ds_ids : list of str
        List of DS IDs.
    meteo_data : pd.DataFrame
        Meteorological data.
    start_date : datetime
        Start date for the analysis period.
    end_date : datetime
        End date for the analysis period.
    init_params : dict
        Initial parameters for the estimation.
    param_bounds : dict, optional
        Bounds for the parameters, by default None.
    tau_range : tuple, optional
        Range for the tau parameter, by default None.
    use_global : bool, optional
        Whether to use global parameters, by default True.
    unwrapping_method : str, optional
        Unwrapping method, options are "implicit" and "lsa", by default "lsa".
    pe_dir : str, optional
        Directory for the full coherence matrix, if using LSA unwrapping, by default None.
    fitting_flag : int boolean, optional
        Flag for choosing the fitting method, options are 0 for temporal coherence and 1 for normalized temporal coherence, by default 1.


    Returns
    -------
    list of xr.Dataset
        List of datasets containing the initial estimated parameters and time series for each point.
    """
    ntracks = len(ts_stm_list)
    sdate = datetime.strptime(str(start_date), "%Y%m%d").strftime("%Y-%m-%d")
    edate = datetime.strptime(str(end_date), "%Y%m%d").strftime("%Y-%m-%d")
    daily_dates = pd.date_range(pd.to_datetime(sdate), pd.to_datetime(edate))

    for ds_id in ds_ids: #[64068]
        meteo_id = ts_stm_list[0]["meteo_id"].values[0]
        meteo_df = meteo_data[meteo_data["meteo_id"] == meteo_id].reset_index(drop=True)

        insar_epoch_idx_list = [None] * ntracks
        z2ph_list = [None] * ntracks
        arc_phi_dc_list = [None] * ntracks
        arc_phi_var_dc_list = [None] * ntracks
        seg_block_idx_list = [None] * ntracks

        for i in range(ntracks):
            ts_stm = ts_stm_list[i]
            t = ts_stm["time"].values
            mask = (ts_stm["pnt_id"] == ds_id).compute()
            idx = mask.where(mask, drop=True)["space"].values[0]

            insar_epoch_idx_list[i] = np.where(daily_dates.isin(t))[0]
            z2ph_list[i] = ts_stm["z2ph"].values[idx]
            arc_phi_dc_list[i] = ts_stm["arc_phi_dc"].values[idx]
            arc_phi_var_dc_list[i] = ts_stm["arc_phi_var_dc"].values[idx]
            seg_block_idx_list[i] = extract_blocks_idx(ts_stm["ds_segments"].values[idx])

        # - Find optimal parameters
        # - If one of the stacks has no valid observations, skip the estimation for this point
        if any(np.isnan(a).all() for a in arc_phi_dc_list):
            # print("No observations for this arc. Continue to the next arc.")
            continue
        else:
            print(f"DS #{ds_id}: estimating initial parameters ...")
            results = insar_derived_spams_x_init(
                init_params,
                meteo_df,
                pd.to_datetime(sdate),
                pd.to_datetime(edate),
                arc_phi_dc_list,
                insar_epoch_idx_list,
                seg_block_idx_list,
                z2ph_list,
                bounds=param_bounds,
                tau_range=tau_range,
                use_global=use_global,
                # fitting_flag=fitting_flag,
            )
            print(f"Optimal parameters: cost={results['cost']:.4f}, xP={results['xP']:.4f}, xE={results['xE']:.4f}, xI={results['xI']:.4f}, tau={results['tau']} days")

            # - Calculate absolute phase
            meteo_subset = meteo_df.loc[
                (
                    meteo_df["datum"]
                    > (pd.to_datetime(sdate) - timedelta(days=results["tau"]))
                )
                & (meteo_df["datum"] <= pd.to_datetime(edate))
            ].reset_index(drop=True)

            _, _, height = spams_model(
                results["xP"],
                results["xE"],
                results["xI"],
                results["tau"],
                meteo_subset,
            )

            ehat_m_list = [None] * len(z2ph_list)
            for i in range(ntracks):
                ts_stm = ts_stm_list[i]
                stack_id = ts_stm.attrs["stack_id"]
                t = ts_stm["time"].values
                mask = (ts_stm["pnt_id"] == ds_id).compute()
                idx = mask.where(mask, drop=True)["space"].values[0]

                ts_phi_abs_init, ahat, ehat_rad, T, nobs = ts_unwrap_init(
                    height,
                    arc_phi_dc_list[i],
                    arc_phi_var_dc_list[i],
                    insar_epoch_idx_list[i],
                    seg_block_idx_list[i],
                    z2ph_list[i],
                    method=unwrapping_method,
                    cm_fpath=os.path.join(pe_dir, stack_id, f"ds_{ds_id}_{stack_id}.h5") if unwrapping_method == "lsa" else None,
                    enl=ts_stm["ds_enl"].values[idx] if unwrapping_method == "lsa" else None,
                    ref_phi_var=ts_stm["ref_phi_var"].values[idx] if unwrapping_method == "lsa" else None,
                )

                gamma_t = np.abs(np.nansum(np.exp(1j * ehat_rad))) / np.sum(~np.isnan(ehat_rad))
                rmse = np.sqrt(np.nanmean((ehat_rad/z2ph_list[i]) ** 2))

                ts_stm["ts_phi_abs_init"].values[idx] = ts_phi_abs_init
                ts_stm["ts_pov_init"].values[idx] = ts_phi_abs_init / z2ph_list[i]
                ts_stm["ahat_init"].values[idx] = ahat
                ts_stm["xP_init"].values[idx] = results["xP"]
                ts_stm["xE_init"].values[idx] = results["xE"]
                ts_stm["xI_init"].values[idx] = results["xI"]
                ts_stm["tau_init"].values[idx] = results["tau"]
                ts_stm["ehat_rad_init"].values[idx] = ehat_rad
                ts_stm["ehat_m_init"].values[idx] = ehat_rad / z2ph_list[i]
                ts_stm["T_init"].values[idx] = T
                ts_stm["nobs_init"].values[idx] = nobs
                ts_stm["gamma_t_init"].values[idx] = gamma_t
                ts_stm["rmse_init"].values[idx] = rmse
                ts_stm_list[i] = ts_stm

                ehat_m_list[i] = ehat_rad / z2ph_list[i]
                K = chi2.ppf(1 - 0.10, nobs - len(init_params))
                omt90 = 1 if T > K else 0
                K = chi2.ppf(1 - 0.05, nobs - len(init_params))
                omt95 = 1 if T > K else 0
                K = chi2.ppf(1 - 0.01, nobs - len(init_params))
                omt99 = 1 if T > K else 0

                print(
                    f"Stack {i + 1}/{len(z2ph_list)} ({stack_id}): rmse={rmse:.2f} mm, gamma_t={gamma_t:.2f}, omt90_reject={omt90}, omt95_reject={omt95}, omt99_reject={omt99}, sigma2_test={T / (nobs - 4):.2f}"
                )

            # - Summary statistics
            ehat_mm = np.concatenate(ehat_m_list)
            rmse = np.sqrt(np.nanmean(ehat_mm**2))
            print(
                f"DS #{ds_id}: RMSE={rmse:.2f} mm, xP={results['xP']:.4f}, xE={results['xE']:.4f}, xI={results['xI']:.4f}, tau={results['tau']} days"
            )
    
    for ts_stm in ts_stm_list:
        stack_id = ts_stm.attrs["stack_id"]
        ts_filepath = os.path.join(ts_est_dir, "ts_stm_" + stack_id + ".zarr")
        ts_stm.to_zarr(ts_filepath, mode="a", consolidated=False,)

    return ts_stm_list


def insar_derived_spams_x_init_v0(
    init_params,
    meteo_df,
    date1,
    date2,
    arc_phis_dc,
    epoch_idxs,
    seg_block_idxs,
    z2phs,
    fitting_flag=1,
):
    """Find optimal InSAR-derived SPAMS parameters by minimizing the objective function.

    Parameters
    ----------
    init_params : list or np.array
        Initial parameter values for optimization [xP, xE, xI, tau].
    meteo_df : pd.DataFrame
        Meteorological data from a nearby station containing at least "datum", "pr", and "pet" columns.
    date1 : datetime
        Start date of observation period.
    date2 : datetime
        End date of observation period.
    arc_phis_dc : list of np.array
        List of InSAR daisy chain observations (multistack).
        Each stack have different shapes.
    epoch_idxs: list of np.array
        List of indices of InSAR epochs corresponding to the daily estimates (multistack).
        Each stack have different shapes.
    seg_block_idxs: list of list of tuple
        List of list of tuples indicating the start and end indices of each segment in the daisy chain (multistack).
        Each stack have different shapes.
    z2phs : list of float
        Conversion factor from height to phase (multistack).
        Each stack have one constant.
    fitting_flag : int boolean, optional
        Flag for choosing the fitting method, options are 0 for temporal coherence and 1 for normalized temporal coherence, by default 1.

    Returns
    -------
    dict
        A dictionary containing the optimum objective value and the corresponding parameters.
    """

    best = {
        "cost": np.inf,
        "xP": None,
        "xE": None,
        "xI": None,
        "tau": None,
    }

    # - Constrains for fminsearch
    xP = init_params[0]
    xP_min = 0.05
    xP_max = 0.15
    xP_mid = (xP_max-xP_min)/2+xP_min
    xP_scale = 100/(xP_max-xP_mid)**20
    xP_factor = xP_scale*(xP-xP_mid)**20 + 1

    xE = init_params[1]
    xE_min = 0.04
    xE_max = 0.35
    xE_mid = (xE_max-xE_min)/2+xE_min
    xE_scale = 100/(xE_max-xE_mid)**20
    xE_factor = xE_scale*(xE-xE_mid)**20 + 1

    xI = init_params[2]
    xI_min = -0.1
    xI_max = 1e-8
    xI_mid = (xI_max-xI_min)/2+xI_min
    xI_scale = 100/(xI_max-xI_mid)**20
    xI_factor = xI_scale*(xI-xI_mid)**20 + 1

    tau = 70
    meteo_subset = meteo_df.loc[
        (meteo_df["datum"] > (date1 - timedelta(days=tau)))
        & (meteo_df["datum"] <= date2)
    ].reset_index(drop=True)
        
    # - Define objective for current tau
    def objective(params):
        params[3] = 70
        _, _, height = spams_model(*params, meteo_subset)

        # - Model residue
        ehats_rad = [None] * len(z2phs)
        ehats_mm = [None] * len(z2phs)

        for i in range(len(z2phs)):
            model_height = height[epoch_idxs[i]]
            model_phi = model_height * z2phs[i]
            ehat_rad = np.full((epoch_idxs[i].size,), np.nan)
            for s in range(len(seg_block_idxs[i])):
                rmin, rmax = seg_block_idxs[i][s][0], seg_block_idxs[i][s][1]
                model_phi_seg = (
                    model_phi[rmin : rmax + 1] - model_phi[rmin : rmax + 1][0]
                )  # -each segment start at zero

                # - Use daisy chain data
                model_phi_dc = wrap(np.diff(model_phi_seg))
                arc_phi_dc = arc_phis_dc[i][rmin + 1 : rmax + 1]
                ehat_rad[rmin + 1 : rmax + 1] = arc_phi_dc - model_phi_dc

            ehats_rad[i] = ehat_rad
            ehats_mm[i] = ehats_rad[i] / z2phs[i]

        ehat_rad = np.concatenate(ehats_rad)

        # - Temporal coherence
        gamma_t = np.abs(np.nansum(np.exp(1j * ehat_rad))) / np.sum(~np.isnan(ehat_rad))

        if fitting_flag == 1:
            cost = 1 / gamma_t * xP_factor * xE_factor * xI_factor
        else:
            cost = 1 / gamma_t

        return cost

    # - Local refinement with Nelder-Mead (fmin)
    xopt, fopt, _, _, _ = fmin(
        objective,
        x0=init_params,
        xtol=1e-3,
        ftol=1e-3,
        full_output=True,
        disp=True,
    )
    if fopt < best["cost"]:
        best.update(
            {
                "cost": fopt,
                "xP": xopt[0],
                "xE": xopt[1],
                "xI": xopt[2],
                "tau": tau,
            }
        )

    return best


def insar_derived_spams_x_init(
    init_params,
    meteo_df,
    date1,
    date2,
    obs_phi_list,
    obs_epoch_idx_list,
    seg_block_idx_list,
    z2ph_list,
    bounds=None,
    tau_range=None,
    use_global=True,
):
    """Find optimal InSAR-derived SPAMS parameters by minimizing the objective function.

    Parameters
    ----------
    init_params : list or np.array
        Initial parameter values for optimization [xP, xE, xI, tau].
    meteo_df : pd.DataFrame
        Meteorological data from a nearby station containing at least "datum", "pr", and "pet" columns.
    date1 : datetime
        Start date of observation period.
    date2 : datetime
        End date of observation period.
    obs_phi_list : list of np.array
        List of InSAR observations (multistack).
        Each stack have different shapes.
    obs_epoch_idx_list: list of np.array
        List of indices of InSAR epochs corresponding to the daily estimates (multistack).
        Each stack have different shapes.
    seg_block_idx_list: list of list or tuple
        List of list of tuples indicating the start and end indices of each segment in the daisy chain (multistack).
        Each stack have different shapes.
    z2ph_list : list of float
        Conversion factor from height to phase (multistack).
        Each stack have one constant.
    bounds : list of tuple, optional
        Bounds for each parameter in the form [(min1, max1), (min2, max2), ...], by default None.
    tau_range : tuple, optional
        Bounds for tau in the form (min_tau, max_tau), by default None.
    use_global : bool, optional (True/False/None)
        Whether to use global optimization (differential evolution) before local refinement, by default True.

    Returns
    -------
    dict
        A dictionary containing the optimum objective value and the corresponding parameters.
    """

    best = {
        "cost": np.inf,
        "xP": None,
        "xE": None,
        "xI": None,
        "tau": None,
    }

    if bounds is None:
        bounds = [(0.0, 0.9), (0.0, 0.9), (-0.03, 1e-6)]
    
    if tau_range is None:
        tau_range = (60,80)
    
    for tau in range(tau_range[0], tau_range[1]+1):
        meteo_subset = meteo_df.loc[
            (meteo_df["datum"] > (date1 - timedelta(days=tau)))
            & (meteo_df["datum"] <= date2)
        ].reset_index(drop=True)
        
        try:
            # - Define objective for current tau
            def objective(params):
                # params[3] = round(params[3])
                # meteo_subset = meteo_df.loc[
                #     (meteo_df["datum"] > (date1 - timedelta(days=params[3])))
                #     & (meteo_df["datum"] <= date2)
                # ].reset_index(drop=True)

                _, _, height = spams_model(*params, tau, meteo_subset)

                # - Model residue
                ehats_rad = [None] * len(z2ph_list)
                ehats_mm = [None] * len(z2ph_list)

                for i in range(len(z2ph_list)):
                    model_height = height[obs_epoch_idx_list[i]]
                    model_phi = model_height * z2ph_list[i]
                    ehat_rad = np.full((obs_epoch_idx_list[i].size,), np.nan)
                    for s in range(len(seg_block_idx_list[i])):
                        rmin, rmax = seg_block_idx_list[i][s][0], seg_block_idx_list[i][s][1]
                        model_phi_seg = (
                            # model_phi[rmin : rmax + 1]
                            model_phi[rmin : rmax + 1] - model_phi[rmin : rmax + 1][0]
                        )  # -each segment start at zero (?)

                        # - Use single mother data
                        # _, ahat = wrap_with_ambiguity(model_phi_seg)
                        # arc_phi_seg_abs = wrap(np.cumsum(obs_phi_list[i][rmin:rmax+1])) + (2*np.pi*ahat)
                        # ehat_rad[rmin+1:rmax+1] = arc_phi_seg_abs[1:] - model_phi_seg[1:]

                        # - Use daisy chain data
                        model_phi_dc = wrap(np.diff(model_phi_seg))
                        arc_phi_dc = wrap(np.diff(obs_phi_list[i][rmin : rmax + 1]))
                        ehat_rad[rmin + 1 : rmax + 1] = arc_phi_dc - model_phi_dc

                    ehats_rad[i] = ehat_rad
                    ehats_mm[i] = ehats_rad[i] / z2ph_list[i]

                ehat_rad = np.concatenate(ehats_rad)

                # - Temporal coherence
                gamma_t = np.abs(np.nansum(np.exp(1j * ehat_rad))) / np.sum(~np.isnan(ehat_rad))
                cost = 1 / gamma_t

                # - RMSE
                # ehat_mm = np.concatenate(ehats_mm)
                # rmse = np.sqrt(np.nanmean(ehat_mm**2))
                # cost = rmse

                # - Cost
                # cost = 1 / gamma_t + rmse

                return cost

            # - Optimize parameters for current tau
            if use_global:
                print("Global search with differential evolution ...")
                global_result = differential_evolution(
                    objective,
                    bounds=bounds,
                    # tol=0.01,
                    # atol=1e-3,
                    popsize=30,
                    mutation=(0.3, 1.0),
                    recombination=0.4,
                    polish=False,
                    # disp=True,
                )
                x0_refined = global_result.x

            else:
                x0_refined = init_params[:3]
            print(f"Initial guess for local optimization at tau={tau}: {x0_refined}")

            print("Local refinement with Nelder-Mead (minimize) ...")
            result = minimize(objective, x0=x0_refined,
                              bounds=bounds,
                              method="Nelder-Mead",
                              options={"xatol": 1e-6, "fatol": 1e-6, "disp": True},
                              )
            print(f"Parameters from local optimization: {result.x}")
            if result.success and result.fun < best["cost"]:
                best.update({
                    "cost": result.fun,
                    "xP": result.x[0],
                    "xE": result.x[1],
                    "xI": result.x[2],
                    "tau": round(tau),
                })

            # print("Local refinement with Nelder-Mead (fmin) ...")
            # xopt, fopt, _, _, _ = fmin(
            #     objective,
            #     x0=x0_refined,
            #     xtol=1e-4,
            #     ftol=1e-3,
            #     full_output=True,
            #     disp=True,
            # )
            # print(f"Parameters from local optimization: {xopt}")
            # if fopt < best["cost"]:
            #     best.update(
            #         {
            #             "cost": fopt,
            #             "xP": xopt[0],
            #             "xE": xopt[1],
            #             "xI": xopt[2],
            #             "tau": tau,
            #         }
            #     )
        
        except Exception as e:
            print(f"tau={tau} failed: {str(e)}")
            continue

    return best


def ts_unwrap_init(
    h_model,
    arc_phi_dc,
    arc_phi_var_dc,
    epoch_idx,
    seg_block_idx,
    z2ph,
    method="lsa",
    cm_fpath=None,
    enl=None,
    ref_phi_var=None,
):
    """Initial phase unwrapping.

    Parameters
    ----------
    h_model : _type_
        Modeled height time series daily.
    arc_phi_dc : _type_
        Daisy chain phase observations for a single point.
    arc_phi_var_dc : _type_
        Variance of the daisy chain phase observations.
    epoch_idx : _type_
        Index of the InSAR observations epochs in daily time series.
    seg_block_idx : _type_
        Indices of the segment blocks.
    z2ph : _type_
        Conversion factor from p.o.v. height to l.o.s. phase.
    method : str, optional
        Unwrapping method, options are "implicit" and "lsa", by default "lsa".
    cm_fpath : str, optional
        File path for the full coherence matrix, if using LSA unwrapping.
    enl : float, optional
        Equivalent number of looks for the coherence matrix, if using LSA unwrapping.
    ref_phi_var : float, optional
        Variance of the reference phase, if using LSA unwrapping.

    Returns
    -------
    ts_phi_abs_shifted : np.array
        Initial absolute phase estimates after unwrapping and mean shift correction.
    ahat : np.array
        Estimated integer ambiguities for each segment.
    ehat_rad : np.array
        Phase residuals (observed - modeled) in radians after unwrapping.
    T : float
        Test statistic for model fit (e.g., weighted sum of squared residuals).
    nobs : int
        Number of valid observations used in the test statistic.
    """
    model_height = h_model[epoch_idx]
    model_phi = model_height * z2ph
    ts_phi_abs = np.full((epoch_idx.size,), np.nan)
    ts_phi_abs_shifted = np.full((epoch_idx.size,), np.nan)
    ehat_rad = np.full((epoch_idx.size,), np.nan)
    ahat = np.zeros((epoch_idx.size,), dtype=int)

    # - Full coherence matrix for LSA unwrapping
    Qy_full = None
    if os.path.exists(cm_fpath) and False: # - Disable for now since the coherence matrix is not positive definite.
        with h5py.File(cm_fpath, "r") as f:
            ds_cpx_coh = f["cpx_coh"][()]
        ds_phi_var_full = CRB(ds_cpx_coh, enl, mode="phi")
        Qy_full = (np.sqrt(ds_phi_var_full) + np.sqrt(ref_phi_var))**2
    
    for s in range(len(seg_block_idx)):
        rmin, rmax = seg_block_idx[s][0], seg_block_idx[s][1]
        model_phi_seg = (
            model_phi[rmin : rmax + 1] - model_phi[rmin : rmax + 1][0]
        )  # -each segment start at zero

        # - Quality matrix
        if Qy_full is not None:
            Qy = Qy_full[rmin : rmax, rmin + 1 : rmax + 1]
            triu = np.triu(Qy)
            Qy = triu + triu.T - np.diag(np.diag(triu))
            if s == 0:
                Qy_block = Qy
            else:
                Qy_block = block_diag(Qy_block, Qy)
        else:
            Qy = np.diag(arc_phi_var_dc[rmin + 1 : rmax + 1])
            if s == 0:
                Qy_block = Qy
            else:
                Qy_block = block_diag(Qy_block, Qy)

        # =====================================================================
        # IMPLICIT UNWRAPPING
        # =====================================================================
        if method == "implicit":
            _, ahat[rmin : rmax + 1] = wrap_with_ambiguity(model_phi_seg)
            ts_phi_abs[rmin : rmax + 1] = np.cumsum(arc_phi_dc[rmin : rmax + 1]) + (
                2 * np.pi * ahat[rmin : rmax + 1]
            )
            offset = np.nanmean(
                ts_phi_abs[rmin : rmax + 1] - model_phi[rmin : rmax + 1]
            )
            ts_phi_abs_shifted[rmin : rmax + 1] = ts_phi_abs[rmin : rmax + 1] - offset
            ehat_rad[rmin : rmax + 1] = (
                ts_phi_abs_shifted[rmin : rmax + 1] - model_phi[rmin : rmax + 1]
            )

        # =====================================================================
        # EXPLICIT UNWRAPPING
        # =====================================================================
        elif method == "lsa":
            b_hat = np.diff(model_phi_seg)
            y = arc_phi_dc[rmin + 1 : rmax + 1]
            A = 2 * np.pi * np.eye(y.size)

            ahat_fl = (np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A) @ A.T @ np.linalg.inv(Qy) @ (y - b_hat))
            Qahat = np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A)
            # - TODO: fix when Qahat is not positive definite, which causes LAMBDA to fail.
            # - The current solution is to use the daisy chain elements only.
            # - A better solution is to regularize Qahat by adding a small value to the diagonal elements until it becomes positive definite.
            if not is_symetric_positive_definite(Qahat):
                Qy = np.diag(arc_phi_var_dc[rmin + 1 : rmax + 1])
                ahat_fl = (np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A) @ A.T @ np.linalg.inv(Qy) @ (y - b_hat))
                Qahat = np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A)
                Qy_block = np.diag(np.diag(Qy_block)) # - Only use the diagonal elements (daisy-chain)

            ahat[rmin + 1 : rmax + 1], _, _, _, _, _, _ = LAMBDA.main(ahat_fl, Qahat, method=3)
            
            if (ahat[rmin + 1 : rmax + 1] > 5).any() or (ahat[rmin + 1 : rmax + 1] < -5).any():
                ahat[rmin + 1 : rmax + 1] = np.round(ahat_fl).astype(int)

            y_hat = y - A @ ahat[rmin + 1 : rmax + 1]

            ts_phi_abs[rmin : rmax + 1] = np.cumsum(
                np.concat(([arc_phi_dc[rmin : rmax + 1][0]], y_hat))
            )
            offset = np.nanmean(
                ts_phi_abs[rmin : rmax + 1] - model_phi[rmin : rmax + 1]
            )
            ts_phi_abs_shifted[rmin : rmax + 1] = ts_phi_abs[rmin : rmax + 1] - offset
            ehat_rad[rmin : rmax + 1] = (
                ts_phi_abs_shifted[rmin : rmax + 1] - model_phi[rmin : rmax + 1]
            )

        else:
            raise ValueError("Invalid unwrapping method. Options: implicit, lsa.")

    mask = ~np.isnan(arc_phi_var_dc) & ~np.isnan(ehat_rad)
    T = ehat_rad[mask].T @ np.linalg.inv(Qy_block) @ ehat_rad[mask]
    nobs = ehat_rad[mask].shape[0]

    return ts_phi_abs_shifted, ahat, ehat_rad, T, nobs


def ts_est_group(
    ts_stm_list,
    ts_est_dir,
    ds_gids,
    meteo_data,
    start_date,
    end_date,
    # param_bounds=None,
):
    ntracks = len(ts_stm_list)
    sdate = datetime.strptime(str(start_date), "%Y%m%d").strftime("%Y-%m-%d")
    edate = datetime.strptime(str(end_date), "%Y%m%d").strftime("%Y-%m-%d")
    daily_dates = pd.date_range(pd.to_datetime(sdate), pd.to_datetime(edate))

    for ds_gid in ds_gids:
        # print(f"Estimating group parameters for DS group #{ds_gid} ...")
        mask = (ts_stm_list[0]["ds_gid"] == ds_gid).compute()
        idxs = mask.where(mask, drop=True)["space"].values

        meteo_id = mode(ts_stm_list[0]["meteo_id"].values[idxs])[0]
        meteo_df = meteo_data[meteo_data["meteo_id"] == meteo_id].reset_index(drop=True)

        xP_inits = ts_stm_list[0]["xP_init"].values[idxs]
        xE_inits = ts_stm_list[0]["xE_init"].values[idxs]
        xI_inits = ts_stm_list[0]["xI_init"].values[idxs]
        tau_inits = ts_stm_list[0]["tau_init"].values[idxs]
        rmse_inits = ts_stm_list[0]["rmse_init"].values[idxs]

        wavelength = ts_stm_list[0].attrs["wavelength"]
        rmse_tol = 104/360 * wavelength/2

        xP_mask = (np.abs(xP_inits - np.nanmean(xP_inits)) <= 3 * np.nanstd(xP_inits)) & (~np.isnan(xP_inits))
        xE_mask = (np.abs(xE_inits - np.nanmean(xE_inits)) <= 3 * np.nanstd(xE_inits)) & (~np.isnan(xE_inits))
        xI_mask = (np.abs(xI_inits - np.nanmean(xI_inits)) <= 3 * np.nanstd(xI_inits)) & (~np.isnan(xI_inits))
        rmse_mask = (rmse_inits <= rmse_tol) & (~np.isnan(rmse_inits))
        mask = xP_mask & xE_mask & xI_mask & rmse_mask

        sel = np.where(mask)[0]
        ds_ids_sel = ts_stm_list[0]["pnt_id"].values[idxs][sel]
        if sel.size <= 1:
            # print("No valid initial estimates for this group. Continue to the next group.")
            continue
        else:
            print(f"DS group #{ds_gid}: there are {sel.size} valid initial estimates out of {idxs.size} points.")
            print(f"DS IDs for group #{ds_gid}: {ds_ids_sel}")
            xP_group = np.nanmean(xP_inits[sel])
            xE_group = np.nanmean(xE_inits[sel])
            xI_group = np.nanmean(xI_inits[sel])
            tau_group = np.nanmean(tau_inits[sel]).astype(int)
            init_params = [xP_group, xE_group, xI_group, tau_group]

            insar_epoch_idxs = [None] * ntracks
            ts_povs_init = [None] * ntracks
            seg_block_idxs = [None] * ntracks

            for i in range(ntracks):
                ts_stm = ts_stm_list[i]
                t = ts_stm["time"].values
                mask = ts_stm["pnt_id"].isin(ds_ids_sel).compute()
                idxs = mask.where(mask, drop=True)["space"].values

                insar_epoch_idxs[i] = np.where(daily_dates.isin(t))[0]
                ts_povs_init[i] = ts_stm["ts_pov_init"].values[idxs]
                seg_block_idxs[i] = [
                    extract_blocks_idx(ts_stm["ds_segments"].values[idx])
                    for idx in idxs
                ]

            # - Find optimal parameters
            if all(np.isnan(a).all() for a in ts_povs_init):
                print("No observations for this group. Continue to the next group.")
            else:
                results = insar_derived_spams_x_group(
                    init_params,
                    meteo_df,
                    pd.to_datetime(sdate),
                    pd.to_datetime(edate),
                    ts_povs_init,
                    insar_epoch_idxs,
                    seg_block_idxs,
                    # bounds=param_bounds,
                )
                print(
                    f"Group #{ds_gid}: rmse={results["cost"]:.2f} mm, xP={results["xP"]:.4f}, xE={results["xE"]:.4f}, xI={results["xI"]:.4f}, tau={round(results["tau"])} days"
                )

                for i in range(ntracks):
                    ts_stm = ts_stm_list[i]
                    stack_id = ts_stm.attrs["stack_id"]
                    mask = ts_stm["pnt_id"].isin(ds_ids_sel).compute()
                    idxs = mask.where(mask, drop=True)["space"].values

                    ts_stm["xP_group"].values[idxs] = results["xP"]
                    ts_stm["xE_group"].values[idxs] = results["xE"]
                    ts_stm["xI_group"].values[idxs] = results["xI"]
                    ts_stm["tau_group"].values[idxs] = round(results["tau"])
                    ts_stm["rmse_group"].values[idxs] = results["cost"]
                    ts_stm_list[i] = ts_stm

    for ts_stm in ts_stm_list:
        stack_id = ts_stm.attrs["stack_id"]
        ts_filepath = os.path.join(ts_est_dir, "ts_stm_" + stack_id + ".zarr")
        ts_stm.to_zarr(ts_filepath, mode="a", consolidated=False,)

    return ts_stm_list


def insar_derived_spams_x_group(
    init_params,
    meteo_df,
    date1,
    date2,
    ts_povs_init,
    epoch_idxs,
    seg_block_idxs,
    # bounds=None,
):
    """Find optimal InSAR-derived SPAMS parameters by minimizing the objective function.

    Parameters
    ----------
    init_params : list or np.array
        Initial parameter values for optimization [xP, xE, xI].
    meteo_df : pd.DataFrame
        Meteorological data from a nearby station containing at least "datum", "pr", and "pet" columns.
    date1 : datetime
        Start date of observation period.
    date2 : datetime
        End date of observation period.
    ts_povs_init : list of np.array
        List of initial time series observations (multistack).
        Each stack have different shapes.
    epoch_idxs: list of np.array
        List of indices of InSAR epochs corresponding to the daily estimates (multistack).
        Each stack have different shapes.
    seg_block_idxs: list of list of tuple
        List of list of tuples indicating the start and end indices of each segment in the daisy chain (multistack).
        Each stack have different shapes.

    Returns
    -------
    dict
        A dictionary containing the optimum objective value and the corresponding parameters.
    """

    best = {
        "cost": np.inf,
        "xP": None,
        "xE": None,
        "xI": None,
        "tau": None,
    }

    # if bounds is None:
    #     bounds = [(0.0, init_params[0]*2), (0.0, init_params[1]*2), (-5e-2, 1e-8), (60, 80)]

    # - Define objective for current tau
    def objective(params):
        params[3] = round(params[3])
        if params[3] < 1:
            params[3] = 70
        meteo_subset = meteo_df.loc[
            (meteo_df["datum"] > (date1 - timedelta(days=params[3])))
            & (meteo_df["datum"] <= date2)
        ].reset_index(drop=True)

        _, _, height = spams_model(*params, meteo_subset)

        # - Model residue
        ehats = [None] * len(epoch_idxs)

        #- If using observations individually
        for i in range(len(epoch_idxs)):
            model_height = height[epoch_idxs[i]]
            ehat = np.full((ts_povs_init[i].shape), np.nan)
            for j in range(ts_povs_init[i].shape[0]):
                obs = ts_povs_init[i][j]
                for s in range(len(seg_block_idxs[i][j])):
                    rmin, rmax = (
                        seg_block_idxs[i][j][s][0],
                        seg_block_idxs[i][j][s][1],
                    )
                    model_seg = model_height[rmin : rmax + 1]
                    obs_segment = obs[rmin : rmax + 1]
                    ehat[j][rmin : rmax + 1] = obs_segment - model_seg
            ehats[i] = np.concatenate(ehat)
        
        # #- If using median of observations for each epoch
        # for i in range(len(epoch_idxs)):
        #     model_height = height[epoch_idxs[i]]
        #     obs_median = np.nanmedian(ts_povs_init[i], axis=0)
        #     mask = ~np.isnan(obs_median) & ~np.isnan(model_height)
        #     ehat = obs_median[mask] - model_height[mask]
        #     ehats[i] = ehat

        ehat = np.concatenate(ehats)
        rmse = np.sqrt(np.nanmean(ehat**2))
        return rmse

    # - Local refinement with Nelder-Mead (fmin)
    xopt, fopt, _, _, _ = fmin(
        objective,
        x0=init_params,
        xtol=1e-6,
        ftol=1e-6,
        full_output=True,
        disp=True,
    )
    if fopt < best["cost"]:
        best.update(
            {
                "cost": fopt,
                "xP": xopt[0],
                "xE": xopt[1],
                "xI": xopt[2],
                "tau": round(xopt[3]),
            }
        )
    
    # # - Global search with differential evolution
    # global_result = differential_evolution(
    #     objective,
    #     bounds=bounds,
    #     tol=1e-6,
    #     popsize=30,
    #     mutation=(0.3, 1.0),
    #     recombination=0.4,
    #     polish=False,
    # )
    # x0_refined = global_result.x
    # print(f"Initial guess for local optimization: {x0_refined}")

    # # - Local refinement with Nelder-Mead (minimize): without bounds, this is the same as fmin implementation
    # result = minimize(objective, x0=x0_refined,
    #                   bounds=bounds,
    #                   method="Nelder-Mead",
    #                   options={"xatol": 1e-6, "fatol": 1e-4, "disp": True},
    #                   )
    # if result.success and result.fun < best["cost"]:
    #     best.update({
    #         "cost": result.fun,
    #         "xP": result.x[0],
    #         "xE": result.x[1],
    #         "xI": result.x[2],
    #         "tau": round(result.x[3]),
    #     })

    return best


def lambda_ambiguity_estimation(
    ts_stm_list,
    ts_est_dir,
    pe_dir,
    ds_ids,
    meteo_data,
    start_date,
    end_date,
):
    ntracks = len(ts_stm_list)
    sdate = datetime.strptime(str(start_date), "%Y%m%d").strftime("%Y-%m-%d")
    edate = datetime.strptime(str(end_date), "%Y%m%d").strftime("%Y-%m-%d")
    daily_dates = pd.date_range(pd.to_datetime(sdate), pd.to_datetime(edate))

    for ds_id in ds_ids:
        meteo_id = ts_stm_list[0]["meteo_id"].values[0]
        meteo_df = meteo_data[meteo_data["meteo_id"] == meteo_id].reset_index(drop=True)

        mask = (ts_stm_list[0]["pnt_id"] == ds_id).compute()
        idx = mask.where(mask, drop=True)["space"].values[0]

        xP = ts_stm_list[0]["xP_init"].values[idx]
        xE = ts_stm_list[0]["xE_init"].values[idx]
        xI = ts_stm_list[0]["xI_group"].values[idx]
        tau = ts_stm_list[0]["tau_group"].values[idx]
        # if np.isnan(xI) or np.isnan(tau):
        #     xI = ts_stm_list[0]["xI_init"].values[idx]
        #     tau = ts_stm_list[0]["tau_init"].values[idx]

        if not np.isnan(xP) and not np.isnan(xE) and not np.isnan(xI):
            print(f"DS #{ds_id}: LAMBDA ambiguity estimation ...")
            meteo_subset = meteo_df.loc[
                (meteo_df["datum"] > (pd.to_datetime(sdate) - timedelta(days=tau)))
                & (meteo_df["datum"] <= pd.to_datetime(edate))
            ].reset_index(drop=True)

            _, _, height = spams_model(xP, xE, xI, tau, meteo_subset)

            ehat_mm_finals = []
            for i in range(ntracks):
                ts_stm = ts_stm_list[i]
                stack_id = ts_stm.attrs["stack_id"]
                t = ts_stm["time"].values
                mask = (ts_stm["pnt_id"] == ds_id).compute()
                idx = mask.where(mask, drop=True)["space"].values[0]

                # - Uniform noise threshold (mm)
                wavelength = ts_stm.attrs["wavelength"]
                rmse_tol = 104/360 * wavelength/2

                ts_phi_abs_init = ts_stm["ts_phi_abs_init"].values[idx]
                # ts_phi_init = wrap(ts_phi_abs_init)
                insar_epoch_idx = np.where(daily_dates.isin(t))[0]
                seg_block_idx = extract_blocks_idx(ts_stm["ds_segments"].values[idx])
                z2ph = ts_stm["z2ph"].values[idx]
                model_phi = height[insar_epoch_idx] * z2ph
                arc_phi_var_dc = ts_stm["arc_phi_var_dc"].values[idx]
                arc_phi_var_sm = ts_stm["arc_phi_var_sm"].values[idx]

                # - Full coherence matrix if available, otherwise use daisy chain variance only
                Qy_full = None
                cm_fpath = os.path.join(pe_dir, stack_id, f"ds_{ds_id}_{stack_id}.h5")
                if os.path.exists(cm_fpath) and False: # - Disable for now since the coherence matrix is not positive definite.
                    with h5py.File(cm_fpath, "r") as f:
                        ds_cpx_coh = f["cpx_coh"][()]
                    enl = ts_stm["ds_enl"].values[idx]
                    ds_phi_var_full = CRB(ds_cpx_coh, enl, mode="phi")
                    Qy_full = (np.sqrt(ds_phi_var_full) + np.sqrt(ts_stm["ref_phi_var"].values[idx]))**2

                ahat_final = np.zeros((ts_phi_abs_init.size,), dtype=int)
                ts_phi_abs_final = np.full((ts_phi_abs_init.size,), np.nan)
                ehat_rad_final = np.full((ts_phi_abs_init.size,), np.nan)
                for s in range(len(seg_block_idx)):
                    rmin, rmax = seg_block_idx[s][0], seg_block_idx[s][1]
                    model_phi_seg = (model_phi[rmin:rmax+1] - model_phi[rmin:rmax+1][0])  # -each segment start at zero

                    # - Quality matrix
                    if Qy_full is not None:
                        Qy = Qy_full[rmin : rmax, rmin + 1 : rmax + 1]
                        triu = np.triu(Qy)
                        Qy = triu + triu.T - np.diag(np.diag(triu))
                        if s == 0:
                            Qy_block = Qy
                        else:
                            Qy_block = block_diag(Qy_block, Qy)
                    else:
                        Qy = np.diag(arc_phi_var_dc[rmin + 1 : rmax + 1])
                        if s == 0:
                            Qy_block = Qy
                        else:
                            Qy_block = block_diag(Qy_block, Qy)

                    bhat = np.diff(model_phi_seg)
                    y = wrap(np.diff(ts_phi_abs_init[rmin:rmax+1]))
                    A = 2 * np.pi * np.eye(y.size)

                    ahat_fl = (np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A) @ A.T @ np.linalg.inv(Qy) @ (y - bhat))
                    ahat = np.round(ahat_fl).astype(int)
                    Qahat = np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A)
                    # - TODO: fix when Qahat is not positive definite, which causes LAMBDA to fail.
                    # - The current solution is to use the daisy chain elements only.
                    # - A better solution is to regularize Qahat by adding a small value to the diagonal elements until it becomes positive definite.
                    # while not is_symetric_positive_definite(Qahat):
                    #     D, V = np.linalg.eigh(Qahat)
                    #     D = np.clip(D, 1e-10, None)
                    #     Qahat = V @ np.diag(D) @ V.T
                    if not is_symetric_positive_definite(Qahat):
                        Qy = np.diag(arc_phi_var_dc[rmin + 1 : rmax + 1])
                        ahat_fl = (np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A) @ A.T @ np.linalg.inv(Qy) @ (y - bhat))
                        Qahat = np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A)
                        Qy_block = np.diag(np.diag(Qy_block)) # - Only use the diagonal elements (daisy-chain)

                    ahat, _, _, _, _, _, _ = LAMBDA.main(ahat_fl, Qahat, method=3)

                    y_hat = y - A @ ahat
                    ts_phi_abs_seg = np.cumsum(np.concat(([ts_phi_abs_init[rmin:rmax+1][0]], y_hat)))
                    offset = np.nanmean(ts_phi_abs_seg - model_phi[rmin:rmax+1])
                    shifted_seg = ts_phi_abs_seg - offset
                    
                    # rmse_seg = np.sqrt(np.mean((shifted_seg/z2ph - model_phi[rmin:rmax+1]/z2ph)**2))

                    # if rmse_seg > rmse_tol:
                    #     offset = np.nanmin(ts_phi_abs_seg - model_phi[rmin:rmax+1])
                    #     shifted_seg = ts_phi_abs_seg - offset
                    #     a = np.round((shifted_seg - model_phi[rmin:rmax+1]) / (2*np.pi)).astype(int)
                    #     ts_phi_abs_seg = shifted_seg - a*2*np.pi
                    #     offset = np.nanmean(ts_phi_abs_seg - model_phi[rmin:rmax+1])
                    #     shifted_seg = ts_phi_abs_seg - offset
                    #     rmse_seg = np.sqrt(np.mean((shifted_seg/z2ph - model_phi[rmin:rmax+1]/z2ph)**2))
                    #     ahat = a[1:]
                    
                    # if rmse_seg > rmse_tol:
                    #     bhat = model_phi_seg
                    #     y = wrap(ts_phi_abs_init[rmin:rmax+1])
                    #     Qy = np.diag(arc_phi_var_sm[rmin:rmax+1])
                    #     A = 2 * np.pi * np.eye(y.size)

                    #     ahat_fl = (np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A) @ A.T @ np.linalg.inv(Qy) @ (y - bhat))
                    #     ahat = np.round(ahat_fl).astype(int)
                    #     Qahat = np.linalg.inv(A.T @ np.linalg.inv(Qy) @ A)
                    #     ahat, _, _, _, _, _, _ = LAMBDA.main(ahat_fl, Qahat, method=3)

                    #     y_hat = y - A @ ahat
                    #     ts_phi_abs_seg = y_hat
                    #     offset = np.nanmean(ts_phi_abs_seg - model_phi[rmin:rmax+1])
                    #     shifted_seg = ts_phi_abs_seg - offset
                    #     rmse_seg = np.sqrt(np.mean((shifted_seg/z2ph - model_phi[rmin:rmax+1]/z2ph)**2))
                    #     ahat = ahat[1:]
                    
                    # if rmse_seg <= rmse_tol:
                    #     ahat_final[rmin+1:rmax+1] = ahat
                    #     ts_phi_abs_final[rmin:rmax+1] = shifted_seg
                    #     ehat_rad_final[rmin:rmax+1] = shifted_seg - model_phi[rmin:rmax+1]

                    ahat_final[rmin+1:rmax+1] = ahat
                    ts_phi_abs_final[rmin:rmax+1] = shifted_seg
                    ehat_rad_final[rmin:rmax+1] = shifted_seg - model_phi[rmin:rmax+1]
                ehat_mm_finals.append(ehat_rad_final / z2ph)

                ts_stm["ahat_final"].values[idx] = ahat_final
                ts_stm["ts_phi_abs_final"].values[idx] = ts_phi_abs_final
                ts_stm["ts_pov_final"].values[idx] = ts_phi_abs_final / z2ph
                ts_stm["ehat_rad_final"].values[idx] = ehat_rad_final
                ts_stm["ehat_mm_final"].values[idx] = ehat_rad_final / z2ph
                ts_stm_list[i] = ts_stm
            
            print(f"rmse = {np.sqrt(np.nanmean(np.concatenate(ehat_mm_finals)**2)):.2f} mm")

    for ts_stm in ts_stm_list:
        stack_id = ts_stm.attrs["stack_id"]
        ts_filepath = os.path.join(ts_est_dir, "ts_stm_" + stack_id + ".zarr")
        ts_stm.to_zarr(ts_filepath, mode="a", consolidated=False,)

    return ts_stm_list

        
def ts_est_final(
    ts_stm_list,
    ts_est_dir,
    pe_dir,
    ds_ids,
    meteo_data,
    start_date,
    end_date,
):
    ntracks = len(ts_stm_list)
    sdate = datetime.strptime(str(start_date), "%Y%m%d").strftime("%Y-%m-%d")
    edate = datetime.strptime(str(end_date), "%Y%m%d").strftime("%Y-%m-%d")
    daily_dates = pd.date_range(pd.to_datetime(sdate), pd.to_datetime(edate))

    for ds_id in ds_ids:
        meteo_id = ts_stm_list[0]["meteo_id"].values[0]
        meteo_df = meteo_data[meteo_data["meteo_id"] == meteo_id].reset_index(drop=True)

        mask = (ts_stm_list[0]["pnt_id"] == ds_id).compute()
        idx = mask.where(mask, drop=True)["space"].values[0]

        xP = ts_stm_list[0]["xP_init"].values[idx]
        xE = ts_stm_list[0]["xE_init"].values[idx]
        xI = ts_stm_list[0]["xI_group"].values[idx]
        tau = ts_stm_list[0]["tau_group"].values[idx]
        # if np.isnan(xI) or np.isnan(tau):
        #     xI = ts_stm_list[0]["xI_init"].values[idx]
        #     tau = ts_stm_list[0]["tau_init"].values[idx]
        init_params = [xP, xE, xI, tau]

        if not np.isnan(init_params).any():
            print(f"DS #{ds_id}: final parameter estimation ...")
            insar_epoch_idxs = [None] * ntracks
            seg_block_idxs = [None] * ntracks
            ts_povs = [None] * ntracks
            # ts_vars = [None] * ntracks
            Qy_fulls = [None] * ntracks

            for i in range(ntracks):
                ts_stm = ts_stm_list[i]
                stack_id = ts_stm.attrs["stack_id"]
                t = ts_stm["time"].values
                mask = (ts_stm["pnt_id"] == ds_id).compute()
                idx = mask.where(mask, drop=True)["space"].values[0]

                # z2ph = ts_stm["z2ph"].values[idx]
                insar_epoch_idxs[i] = np.where(daily_dates.isin(t))[0]
                seg_block_idxs[i] = extract_blocks_idx(ts_stm["ds_segments"].values[idx])
                ts_povs[i] = ts_stm["ts_pov_final"].values[idx]
                # ts_vars[i] = ts_stm["arc_phi_var_dc"].values[idx]
                # ts_vars[i] = (np.sqrt(ts_stm["arc_phi_var_dc"].values[idx]) / ts_stm["z2ph"].values[idx])**2

                # - If using the full coherence matrix as var-covar
                cm_fpath = os.path.join(pe_dir, stack_id, f"ds_{ds_id}_{stack_id}.h5")
                if os.path.exists(cm_fpath): #and False: # - Disable for now since the coherence matrix is not positive definite.
                    with h5py.File(cm_fpath, "r") as f:
                        ds_cpx_coh = f["cpx_coh"][()]
                    enl = ts_stm["ds_enl"].values[idx]
                    ds_phi_var_full = CRB(ds_cpx_coh, enl, mode="phi")
                    Qy_fulls[i] = (np.sqrt(ds_phi_var_full) + np.sqrt(ts_stm["ref_phi_var"].values[idx])) ** 2
                    # Qy_fulls[i] = ds_phi_var_full + ts_stm["ref_phi_var"].values[idx]
                    # Qy_fulls[i] = (np.sqrt(Qy_full) / z2ph)**2
                else:
                    # ts_vars[i] = ts_stm["arc_phi_var_dc"].values[idx]
                    # ts_vars[i] = (np.sqrt(ts_stm["arc_phi_var_dc"].values[idx]) / ts_stm["z2ph"].values[idx])**2
                    Qy_fulls[i] = np.diag(ts_stm["arc_phi_var_dc"].values[idx])

            if not all(np.isnan(ts_pov).all() for ts_pov in ts_povs):
                # - Find optimal parameters
                results = insar_derived_spams_x_final(
                    init_params,
                    meteo_df,
                    pd.to_datetime(sdate),
                    pd.to_datetime(edate),
                    ts_povs,
                    insar_epoch_idxs,
                    seg_block_idxs,
                )
                print(f"DS #{ds_id}: rmse={results["cost"]:.2f} mm, xP={results["xP"]:.4f}, xE={results["xE"]:.4f}, xI={results["xI"]:.4f}, tau={results["tau"]} days")

                # - Model realization with the final parameters
                meteo_subset = meteo_df.loc[
                    (meteo_df["datum"] > (pd.to_datetime(sdate) - timedelta(days=results["tau"])))
                    & (meteo_df["datum"] <= pd.to_datetime(edate))
                ].reset_index(drop=True)

                _, _, hspams = spams_model(results["xP"], results["xE"], results["xI"], results["tau"], meteo_subset)
            
                # - Parameter uncertainty estimation and overall model test
                xhat = [results["xP"], results["xE"], results["xI"], results["tau"]]
                B_full = matrix_design_B(*xhat, meteo_subset)
                Bs = [None] * ntracks
                # vars = [None] * ntracks
                Qys = [None] * ntracks
                Qys_mm = [None] * ntracks
                ehats_mm = [None] * ntracks
                ehats_rad = [None] * ntracks
                n = 0
                for i, ts_pov in enumerate(ts_povs):
                    ts_stm = ts_stm_list[i]
                    t = ts_stm["time"].values
                    mask = (ts_stm["pnt_id"] == ds_id).compute()
                    idx = mask.where(mask, drop=True)["space"].values[0]

                    z2ph = ts_stm["z2ph"].values[idx]
                    insar_epoch_idx = np.where(daily_dates.isin(t))[0]

                    B_sub = B_full[insar_epoch_idx, :]
                    mask = ~np.isnan(ts_pov) & ~np.isnan(np.diag(Qy_fulls[i]))
                    Bs[i] = B_sub[mask, :]

                    Qys[i] = Qy_fulls[i][mask][:, mask]
                    Qys_mm[i] = (np.sqrt(Qys[i]) / z2ph)**2
                    n += Qys[i].shape[0]
                    yhat = hspams[insar_epoch_idx]
                    ehats_mm[i] = ts_pov - yhat
                    ehats_rad[i] = ehats_mm[i] * z2ph

                    ts_stm["ehat_mm_final"].values[idx] = ehats_mm[i]
                    ts_stm["ehat_rad_final"].values[idx] = ehats_rad[i]
                    ts_stm["nobs_final"].values[idx] = len(ehats_mm[i])
                    ts_stm_list[i] = ts_stm

                    ehats_mm[i] = ehats_mm[i][mask]
                    ehats_rad[i] = ehats_rad[i][mask]
                
                # - Parameter uncertainty estimation
                B = np.concatenate(Bs)
                # - If using only the daisy chain variances as var-covar
                # Qy = np.diag(np.concatenate(vars))
                # - If using the full coherence matrix as var-covar
                Qy = np.zeros((n, n), dtype=Qy_fulls[0].dtype)
                Qy_mm = np.zeros((n, n), dtype=Qy_fulls[0].dtype)
                for i in range(len(Qys)):
                    start = sum(Qy.shape[0] for Qy in Qys[:i])
                    end = start + Qys[i].shape[0]
                    Qy[start:end, start:end] = Qys[i]
                    Qy_mm[start:end, start:end] = Qys_mm[i]
                try:
                    # Qxhat = np.linalg.inv(B.T @ np.linalg.inv(Qy_mm) @ B)
                    Qxhat = np.linalg.inv(B.T @ np.linalg.inv(np.diag(np.diag(Qy_mm))) @ B)
                    Qxhat = np.abs(Qxhat)  # - Ensure non-negative variances
                except np.linalg.LinAlgError:
                    print("Warning: Matrix is singular, cannot compute parameter uncertainties.")
                    Qxhat = np.full((4, 4), np.nan)
                print(f"Parameter uncertainties (std): \\sigma_{{xP}}={np.sqrt(Qxhat[0, 0]):.4f}, \\sigma_{{xE}}={np.sqrt(Qxhat[1, 1]):.4f}, \\sigma_{{xI}}={np.sqrt(Qxhat[2, 2]):.4f}")
                
                # - Overall model test
                ehat = np.concatenate(ehats_mm)
                # mask = ~np.isnan(ehat)
                # ehat = ehat[mask]
                T = ehat.T @ np.linalg.inv(Qy_mm) @ ehat
                dof = len(ehat) - len(xhat)
                K = chi2.ppf(1 - 0.10, dof)
                omt90 = 1 if T > K else 0
                K = chi2.ppf(1 - 0.05, dof)
                omt95 = 1 if T > K else 0
                K = chi2.ppf(1 - 0.01, dof)
                omt99 = 1 if T > K else 0
                print(f"Statistical tests: T={T:.2f}, dof={dof}, OMT90_reject={omt90}, OMT95_reject={omt95}, OMT99_reject={omt99}")

                for i in range(ntracks):
                    ts_stm = ts_stm_list[i]
                    stack_id = ts_stm.attrs["stack_id"]
                    mask = (ts_stm["pnt_id"] == ds_id).compute()
                    idx = mask.where(mask, drop=True)["space"].values[0]

                    ts_stm["xP_final"].values[idx] = results["xP"]
                    ts_stm["xE_final"].values[idx] = results["xE"]
                    ts_stm["xI_final"].values[idx] = results["xI"]
                    ts_stm["tau_final"].values[idx] = round(results["tau"])
                    ts_stm["rmse_final"].values[idx] = results["cost"]
                    ts_stm["T_final"].values[idx] = T
                    ts_stm["var_xP_final"].values[idx] = Qxhat[0, 0]
                    ts_stm["var_xE_final"].values[idx] = Qxhat[1, 1]
                    ts_stm["var_xI_final"].values[idx] = Qxhat[2, 2]
                    ts_stm["var_xPxE_final"].values[idx] = Qxhat[0, 1]
                    ts_stm["var_xPxI_final"].values[idx] = Qxhat[0, 2]
                    ts_stm["var_xExI_final"].values[idx] = Qxhat[1, 2]
                    ts_stm_list[i] = ts_stm
    
    for ts_stm in ts_stm_list:
        stack_id = ts_stm.attrs["stack_id"]
        ts_filepath = os.path.join(ts_est_dir, "ts_stm_" + stack_id + ".zarr")
        ts_stm.to_zarr(ts_filepath, mode="a", consolidated=False,)

    return ts_stm_list


def insar_derived_spams_x_final(
    init_params,
    meteo_df,
    date1,
    date2,
    ts_povs,
    epoch_idxs,
    seg_block_idxs,
):
    """Find optimal InSAR-derived SPAMS parameters by minimizing the objective function.

    Parameters
    ----------
    init_params : list or np.array
        Initial parameter values for optimization [xP, xE, xI].
    meteo_df : pd.DataFrame
        Meteorological data from a nearby station containing at least "datum", "pr", and "pet" columns.
    date1 : datetime
        Start date of observation period.
    date2 : datetime
        End date of observation period.
    ts_povs_init : list of np.array
        List of initial time series observations (multistack).
        Each stack have different shapes.
    epoch_idxs: list of np.array
        List of indices of InSAR epochs corresponding to the daily estimates (multistack).
        Each stack have different shapes.
    seg_block_idxs: list of list of tuple
        List of list of tuples indicating the start and end indices of each segment in the daisy chain (multistack).
        Each stack have different shapes.

    Returns
    -------
    dict
        A dictionary containing the optimum objective value and the corresponding parameters.
    """

    best = {
        "cost": np.inf,
        "xP": None,
        "xE": None,
        "xI": None,
        "tau": None,
    }

    # - Define objective function
    def objective(params):
        params[3] = round(params[3])
        if params[3] < 1:
            params[3] = 70
        if params[0] < 0:
            params[0] = 0.01
        if params[1] < 0:
            params[1] = 0.02
        meteo_subset = meteo_df.loc[
            (meteo_df["datum"] > (date1 - timedelta(days=params[3])))
            & (meteo_df["datum"] <= date2)
        ].reset_index(drop=True)

        _, _, height = spams_model(*params, meteo_subset)

        # - Model residue
        ehats = [None] * len(epoch_idxs)

        for i in range(len(epoch_idxs)):
            model_height = height[epoch_idxs[i]]
            ehats[i] = np.full((ts_povs[i].shape), np.nan)
            obs = ts_povs[i]
            for s in range(len(seg_block_idxs[i])):
                rmin, rmax = (seg_block_idxs[i][s][0], seg_block_idxs[i][s][1])
                model_seg = model_height[rmin:rmax+1]
                obs_segment = obs[rmin:rmax+1]
                ehats[i][rmin:rmax+1] = obs_segment - model_seg

        ehat = np.concatenate(ehats)
        rmse = np.sqrt(np.nanmean(ehat**2))
        return rmse

    # - Local refinement with Nelder-Mead (fmin)
    xopt, fopt, _, _, _ = fmin(
        objective,
        x0=init_params,
        # xtol=1e-3,
        # ftol=1e-4,
        full_output=True,
        disp=True,
    )
    if fopt < best["cost"]:
        best.update(
            {
                "cost": fopt,
                "xP": xopt[0],
                "xE": xopt[1],
                "xI": xopt[2],
                "tau": round(xopt[3]),
            }
        )

    return best


def matrix_design_B(xP, xE, xI, tau, meteo_df):
    """Design matrix B for variance-covariance estimation in the SPAMS parameters.


    Parameters
    ----------
    xP : float
        Scaling factor for precipitation (mm/mm).
    xE : float
        Scaling factor for evapotranspiration (mm/mm).
    xI : float
        Irreversible constant, active during dry period (mm/day).
    tau : int
        Time constant (days).
    meteo_df : pd.DataFrame
        Daily precipitation and evapotranspiration (mm).

    Returns
    -------
    np.array
        Design matrix B.
    """
    tau = round(tau)
    
    if "pr" in meteo_df.columns and "pet" in meteo_df.columns:
        p = np.lib.stride_tricks.sliding_window_view(meteo_df["pr"], tau)
        e = np.lib.stride_tricks.sliding_window_view(meteo_df["pet"], tau)

    elif "precip" in meteo_df.columns and "evapo" in meteo_df.columns:
        p = np.lib.stride_tricks.sliding_window_view(meteo_df["precip"], tau)
        e = np.lib.stride_tricks.sliding_window_view(meteo_df["evapo"], tau)

    else:
        raise NameError("Column(s) 'pr'/'pet'/'precip'/'evapo' are not in meteo data.")

    ## Reversible
    reversible = np.sum(xP * p - xE * e, axis=1)

    ## Irreversible
    dry_flag = np.zeros(reversible.size)
    dry_flag[reversible < 0] = 1
    t_dry = np.cumsum(dry_flag)

    # Construct design matrix B
    B = np.column_stack([np.sum(p, axis=1), -np.sum(e, axis=1), t_dry])

    return B


def ext_derived_spams_x(
    init_params, 
    obs, 
    date1, date2, 
    epoch_idx, 
    meteo_df, 
    bounds=None, 
    tau_range=None,
    use_global=True
):
    """Find optimal SPAMS parameters by minimizing the specified objective function.

    Parameters
    ----------
    init_params : list or np.array
        Initial parameter values for optimization [xP, xE, xI].
    obs : np.array
        Observation data to fit the model against.
    date1 : datetime
        Start date of observation period.
    date2 : datetime
        End date of observation period.
    epoch_idx : np.array
        Indices of the observation period.
    meteo_df : pd.DataFrame
        Meteorological data containing at least "datum", "pr", and "pet" columns.
    bounds : list of tuple, optional
        Bounds for each parameter in the form [(min1, max1), (min2, max2), ...], by default None
    tau_range : tuple, optional
        Bounds for tau in the form (min_tau, max_tau), by default None
    use_global : bool, optional (True/False/None)
        Whether to use global optimization (differential evolution) before local refinement, by default True.

    Returns
    -------
    float
        The optimum objective value.
    """
    
    best = {"rmse": np.inf, "xP": None, "xE": None, "xI": None, "tau": None}

    if bounds is None:
        bounds = [(0., 0.9), (0., 0.9), (-0.1, 1e-6)]
    
    if tau_range is None:
        tau_range = (60,80)

    for tau in range(tau_range[0], tau_range[1]+1):
        meteo_subset = meteo_df.loc[
            (meteo_df["datum"] > (date1 - timedelta(days=tau)))
            & (meteo_df["datum"] <= date2)
        ].reset_index(drop=True)
        
        try:
            # Define objective for current tau
            def objective(params):
                _, _, h = spams_model(*params, tau, meteo_subset)
                offset = np.nanmean(obs - h[epoch_idx])
                rmse = np.sqrt(np.nanmean((obs - (h[epoch_idx] + offset))**2))
                return rmse
            
            # Optimize parameters for current tau
            if use_global:
                # Global search with differential evolution
                global_result = differential_evolution(
                    objective,
                    bounds=bounds,
                    popsize=30,
                    mutation=(0.3, 1.0),
                    recombination=0.4,
                    polish=False,
                    # disp=True,
                )
                x0_refined = global_result.x

            else:
                # Use initial guess for local optimization
                x0_refined = init_params[:3]
            
            # print(f"Initial guess for local optimization at tau={tau}: {x0_refined}")

            # Local refinement
            result = minimize(objective, x0=x0_refined,
                              bounds=bounds,
                              method='Nelder-Mead',
                              options={"xatol": 1e-6, "fatol": 1e-6, "disp": False},
                              )
            
            if result.success:
                _, _, h = spams_model(*result.x, tau, meteo_subset)
                offset = np.nanmean(obs - h[epoch_idx])
                rmse = np.sqrt(np.nanmean((obs - (h[epoch_idx] + offset))**2))
                if rmse < best["rmse"]:
                    best.update({
                        "rmse": rmse,
                        "xP": result.x[0],
                        "xE": result.x[1],
                        "xI": result.x[2],
                        "tau": tau
                    })
                
        except Exception as e:
            print(f"$\\tau$={tau} failed: {str(e)}")
            continue
    
    return best
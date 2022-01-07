#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b1_load_data import NFLData, NotAFullPuntError
from b3_fit_control_data import calcProjfromtrack
from b5_single_bell import BayesZArr, FieldSetup

#%% Functions to calculate Fire Time


def findOptimalTime(calc_proj, targetdf, jerseyRank=0, SS_SNAP=2):
    try:
        sing_gunner = calc_proj.query(f"gunnerJerseyRank == {jerseyRank}")

        targetXY = targetdf.query(f"time_since_snap == {SS_SNAP}").iloc[0]
        gunnerXY = sing_gunner[sing_gunner["time_since_snap"] == SS_SNAP].iloc[0]

    except IndexError:
        return None

    SX, SY = gunnerXY["x"], gunnerXY["y"]
    TX, TY = targetXY["post_x"], targetXY["post_y"]

    total_dist = ((SX - TX) ** 2 + 1.5 * (SY - TY) ** 2) ** 0.5

    VMAX = 10.5
    A_CONST = 7
    time_to_max = (VMAX) / A_CONST
    dist_to_max = 0.5 * (A_CONST) * (time_to_max) ** 2
    remaining_dist = total_dist - dist_to_max
    remaining_time = remaining_dist / VMAX

    return remaining_time + time_to_max


def findPathUCM(
    calc_proj,
    targetdf,
    jerseyRank=0,
    VMAX=10.5,
    A_CONST=7,
    SS_SNAP=2,  ## seconds since snap
    MAG_AR=9.5,
):
    """
    Location S, target T
    Current velocity V0


    Returns Accel to optimal path
    Other identifiers of optimal path Px1, Px2, Vx1, Vy1, R, C etc. 

    """

    sing_gunner = calc_proj.query(f"gunnerJerseyRank == {jerseyRank}")

    targetXY = targetdf.query(f"time_since_snap == {SS_SNAP}").iloc[0]
    gunnerXY = sing_gunner[sing_gunner["time_since_snap"] == SS_SNAP].iloc[0]

    SX, SY = gunnerXY["x"], gunnerXY["y"]
    TX, TY = targetXY["post_x"], targetXY["post_y"]

    vx_0 = gunnerXY["vel_x"]
    vy_0 = gunnerXY["vel_y"]

    def getSinCos(adj_len, opp_len):
        _hyp_len = (adj_len ** 2 + opp_len ** 2) ** 0.5
        return opp_len / _hyp_len, adj_len / _hyp_len

    def getXYComponents(vec_mag, sin_ang, cos_ang):
        return vec_mag * cos_ang, vec_mag * sin_ang

    def getVxy_tfuncs(ax, ay):
        return lambda t: ax * t, lambda t: ay * t

    magnitude = lambda x, y: (x ** 2 + y ** 2) ** 0.5

    def findCxy(sin_ang, cos_ang, TX, TY, perp_vec, t1x, t1y, R):
        """find center of circle given location and direction of player"""
        xdiff = TX - t1x
        slope = sin_ang / cos_ang
        y_at_T = t1y + xdiff * slope

        if TY >= y_at_T:  ## R above t1x
            if perp_vec[1] > 0:  ##going up
                center_x = t1x + R * perp_vec[0]
                center_y = t1y + R * np.abs(perp_vec[1])
            else:
                center_x = t1x - R * perp_vec[0]
                center_y = t1y - R * perp_vec[1]

        else:  ## R below t1x
            if perp_vec[1] < 0:
                center_x = t1x + R * perp_vec[0]
                center_y = t1y + R * perp_vec[1]
            else:
                center_x = t1x - R * perp_vec[0]
                center_y = t1y - R * perp_vec[1]

        return np.array([center_x, center_y])

    def findTangPoint(Cxy, r, targx, targy, t1x, t1y):
        """Finds point on circle t2 thats tangent to target"""
        dx, dy = targx - Cxy[0], targy - Cxy[1]
        dxr, dyr = -dy, dx
        d = (dx ** 2 + dy ** 2) ** 0.5
        if d < r:
            return np.array([None, None])  ## make assertion later

        rho = r / d
        ad = rho ** 2
        bd = rho * (1 - rho ** 2) ** 0.5

        T2x = Cxy[0] + ad * dx - bd * dxr
        T2y = Cxy[1] + ad * dy - bd * dyr

        T2xa = Cxy[0] + ad * dx + bd * dxr
        T2ya = Cxy[1] + ad * dy + bd * dyr

        p1dist = np.abs(t1x - T2x) + np.abs(t1y - T2y)
        padist = np.abs(t1x - T2xa) + np.abs(t1y - T2ya)

        if padist < p1dist or T2x < t1x:
            T2x = T2xa
            T2y = T2ya

        return np.array([T2x, T2y])

    def findUCMTime(perp_vec, tang_xy, c_xy, vel_mag):
        """Finds total time spent in UCM from t1 to t2"""
        if None in tang_xy or vel_mag == 0:
            return None

        rad_vec = tang_xy - c_xy
        v1_u = -perp_vec / np.linalg.norm(perp_vec)
        v2_u = rad_vec / np.linalg.norm(rad_vec)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        return angle * np.linalg.norm(rad_vec) / vel_mag

    def findRemainTime(tang_xy, vel_mag):
        """Finds remaining time needed to target"""
        if None in tang_xy:
            return None
        remaining_dist = np.linalg.norm(tang_xy - np.array([TX, TY]))

        return remaining_dist / (0.7 * VMAX)

    def getAccelStage1(time_to_max, vx_t, vy_t, ax_c, ay_c):
        """given a velocity, accel values and a velocity function
        return location and velocity from stage 1"""
        arrT = np.linspace(0, time_to_max, 5)
        arrVx = [vx_0 + vx_t(t) for t in arrT]
        arrVy = [vy_0 + vy_t(t) for t in arrT]
        arrVmag = [(x ** 2 + y ** 2) ** 0.5 for x, y in zip(arrVx, arrVy)]

        arrPx1 = [SX + vx_t(t) * t + 0.5 * ax_c * (t ** 2) for t in arrT]
        arrPy1 = [SY + vy_t(t) * t + 0.5 * ay_c * (t ** 2) for t in arrT]

        return arrT, arrVx, arrVy, arrVmag, arrPx1, arrPy1

    def getEndStage1(time_post_max, arrT, arrVx, arrVy, arrVmag, arrPx1, arrPy1):
        """Given inputs from stage 1, returns constant velocity phase for
        time_post_max sectonds"""

        arrTpostMax = np.linspace(0, time_post_max, int(time_post_max) * 2 + 1)
        len_tpost = len(arrTpostMax)

        arrVx = arrVx + [arrVx[-1],] * len_tpost
        arrVy = arrVy + [arrVy[-1],] * len_tpost
        arrVmag = arrVmag + [arrVmag[-1],] * len_tpost

        end_x, end_y = arrPx1[-1], arrPy1[-1]
        vel_max_x, vel_max_y = arrVx[-1], arrVy[-1]
        arrPx1 = arrPx1 + [end_x + vel_max_x * t for t in arrTpostMax]
        arrPy1 = arrPy1 + [end_y + vel_max_y * t for t in arrTpostMax]

        arrTpostMax = [arrT[-1] + t for t in arrTpostMax]
        arrT = np.concatenate([arrT, arrTpostMax])

        return arrT, arrVx, arrVy, arrVmag, arrPx1, arrPy1

    sin_ang, cos_ang = getSinCos(vx_0, vy_0)
    cos_refl = cos_ang if sin_ang < 0 else -cos_ang
    perp_vec = np.array([np.abs(sin_ang), cos_refl])

    ax_c, ay_c = getXYComponents(A_CONST, sin_ang, cos_ang)
    vx_t, vy_t = getVxy_tfuncs(ax_c, ay_c)
    time_to_max = (VMAX - magnitude(vx_0, vy_0)) / A_CONST
    arrT, arrVx, arrVy, arrVmag, arrPx1, arrPy1 = getAccelStage1(
        time_to_max, vx_t, vy_t, ax_c, ay_c
    )

    if time_to_max < 3:  #################### THIS is a PARAMETER
        time_post_max = 6 - time_to_max
        arrT, arrVx, arrVy, arrVmag, arrPx1, arrPy1 = getEndStage1(
            time_post_max, arrT, arrVx, arrVy, arrVmag, arrPx1, arrPy1
        )

    calc_R = [v ** 2 / MAG_AR for v in arrVmag]

    calc_C = [
        findCxy(sin_ang, cos_ang, TX, TY, perp_vec, t1x, t1y, R)
        for t1x, t1y, R, in zip(arrPx1, arrPy1, calc_R)
    ]

    #%%

    arrYDiffs = [TY - y1 for y1 in arrPy1]
    calc_Tang2 = [
        findTangPoint(cxy, r, TX, TY, t1x, t1y)
        for cxy, r, t1x, t1y in zip(calc_C, calc_R, arrPx1, arrPy1)
    ]

    calc_UCMTime = [
        findUCMTime(perp_vec, tang_xy, c_xy, vel_mag)
        for tang_xy, c_xy, vel_mag in zip(calc_Tang2, calc_C, arrVmag)
    ]

    calc_RemainTime = [
        findRemainTime(tang_xy, vel_mag)
        for tang_xy, vel_mag in zip(calc_Tang2, arrVmag)
    ]

    total_time = [
        SS_SNAP + t_0 + t_ucm + t_rem if t_rem is not None else None
        for t_0, t_ucm, t_rem in zip(arrT, calc_UCMTime, calc_RemainTime)
    ]

    posdf = pd.DataFrame(
        {
            "time": arrT,
            "Px1": arrPx1,
            "Py1": arrPy1,
            "vx1": arrVx,
            "vy1": arrVy,
            "arrVmag": arrVmag,
            "calcR": calc_R,
            "calc_C": calc_C,
            "Px2": calc_Tang2,
            "calc_UCMTime": calc_UCMTime,
            "calc_RemainingTime": calc_RemainTime,
            "totalTime": total_time,
            "SX": SX,
            "SY": SY,
            "vx0": vx_0,
            "vy0": vy_0,
            "TX": TX,
            "TY": TY,
            "nflId": gunnerXY.nflId,
            "Name": gunnerXY.displayName,
        }
    )

    posdf = posdf[posdf["Px1"] <= TX]
    first_null = posdf[posdf.totalTime.isnull()].index.min()
    posdf = posdf[posdf.index < first_null]

    if posdf.empty:
        return None

    return posdf[posdf.totalTime == posdf.totalTime.min()].iloc[0]


def findTimeStopPivot(
    calc_proj, targetdf, jerseyRank=0, SS_SNAP=2, STOP_FAC=0.12,
):
    """

    """
    #%%
    sing_gunner = calc_proj.query(f"gunnerJerseyRank == {jerseyRank}")

    targetXY = targetdf.query(f"time_since_snap == {SS_SNAP}").iloc[0]
    gunnerXY = sing_gunner[sing_gunner["time_since_snap"] == SS_SNAP].iloc[0]

    SX, SY = gunnerXY["x"], gunnerXY["y"]
    TX, TY = targetXY["post_x"], targetXY["post_y"]

    vx_0 = gunnerXY["vel_x"]
    vy_0 = gunnerXY["vel_y"]

    time_to_stop = STOP_FAC * (vx_0 ** 2 + vy_0 ** 2) ** 0.5 + 2.4

    time_remaining = findOptimalTime(calc_proj, targetdf)
    time_remaining + time_to_stop

    return pd.Series(
        {
            "SX": SX,
            "SY": SY,
            "vx0": vx_0,
            "vy0": vy_0,
            "TX": TX,
            "TY": TY,
            "stopTime": time_to_stop,
            "endTime": time_remaining,
            "altTime": time_remaining + time_to_stop,
        }
    )


def getCircleFormula(C, R, TY):
    if TY <= R:
        flip = -1
    else:
        flip = 1
    return lambda x: flip * (R ** 2 - (x - C[0]) ** 2) ** 0.5 + C[1]


#%%%


def getFireTime(one_row, calc_proj, targetdf, common_cols=["gameId", "playId"]):
    double_results = []

    for jers in [0, 1]:
        optimal_time = findOptimalTime(calc_proj, targetdf, jerseyRank=jers)
        if optimal_time is None:
            continue

        path = findPathUCM(calc_proj, targetdf, jerseyRank=jers)
        alt_time = findTimeStopPivot(calc_proj, targetdf, jerseyRank=jers)["altTime"]

        if path is None or path.totalTime > alt_time:
            fire_time = round(alt_time - optimal_time, 3)
            fireTimeS = round((alt_time - optimal_time) / optimal_time, 3)

        else:
            fire_time = round(path.totalTime - optimal_time, 3)
            fireTimeS = round((path.totalTime - optimal_time) / optimal_time, 3)

        #%%
        if jers == 0:
            pre = one_row["jersRank0"]
        else:
            pre = "r" if one_row["jersRank0"] == "l" else "l"

        save_row = one_row[common_cols].copy()
        save_row["jerseyRank"] = jers
        save_row["isLeft"] = int(pre == "l")
        save_row["isRight"] = int(pre == "r")
        try:
            save_row["nflId"] = one_row[f"{pre}GunnerId"]
            save_row["vises"] = one_row[f"{pre}Gunner_vises"]
            save_row["vise1"] = one_row[f"{pre}Vises1"]
            save_row["vise2"] = one_row[f"{pre}Vises2"]
        except KeyError:
            pass

        save_row["optimal_time"] = optimal_time
        save_row["alt_time"] = alt_time
        save_row["path_time"] = path.totalTime if path is not None else None
        save_row["fire_time"] = fire_time
        save_row["fireTimeS"] = fireTimeS

        double_results.append(save_row)

    return double_results


#%% none Option 175
# path_to_data = "input/nfl-big-data-bowl-2022/"
# path_to_punts = "input/BDB2022-custom/punts_only/"
# data = NFLData(data_path=path_to_data, punts_path=path_to_punts)

# saved_punts = pd.read_csv("input/BDB2022-custom/punts_from_2019.csv", index_col=0)

# #%%

# all_times_collected = []

# for i in saved_punts.index:
#     try:
#         one_data = data.one_play(i)
#         calc_proj = calcProjfromtrack(one_data["track"])
#         BA = BayesZArr(one_data["track"], one_data["row"], calc_proj, saved_punts)
#         all_times_collected.extend(
#             getFireTime(one_data["row"], calc_proj, BA.calc_center())
#         )
#     except NotAFullPuntError:
#         continue

# adf = pd.DataFrame(all_times_collected).reset_index().rename(columns={"index": "pid"})
# print(f"Saving: {adf.shape}")

# adf.to_csv("firstFire2019_2.csv", index=False)


# %%

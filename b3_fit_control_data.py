#%%
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit

from b1_load_data import NFLData, NotAFullPuntError

pd.set_option("display.max_columns", 101)


#%% Play Data class
def calculateZfromtrack(
    seq_df,
    MAX_FIELD_Y=160 / 3,
    MAX_FIELD_X=120,
    MAX_PLAYER_SPEED=11.3,
    ORIENT_MAG=2,
    AIRTIME=4.0,
    VMAX_MU=9.5,
    LB_EXP=2,
    OFF_CUTOFF=0.9,
    N_GRID=121,
    K_EXPIT=3,
    SCALE_OFFENCE=1,
    SCALE_DEFENCE=1,
):
    """
    Calculates catchability and coverage for both teams.
    Parameters
    ----------
    seq_df : DATAFRAME Single play tracking data preprocessed by NFLData
    Can be used for only a few frames rather than entire play eg.
    seq_df =  play_df[play_df.frameId.isin([34,35,36,37])]

    Returns
    -------
    TYPE
        Z_diff : Numpy Array  (N by N array indexed by maxframeID)
        Z_indiv: Numpy Array (N by N by maxframeId by jerseyRank, by isKicking)

    """
    assert not seq_df.empty
    # print('Calculating Zs')
    def polar_to_z(r, theta):
        """returns x and y components given length and angle of vector"""
        xy = r * np.exp(1j * theta)
        return np.real(xy), np.imag(xy)

    def weighted_angle_magnitude(a1, a2, speed):
        """returns weighted angle betwen 2 vectors"""
        if np.isnan(a1[0]):
            return np.nan, np.nan

        def normalize(v):
            norm = np.linalg.norm(v, ord=None)
            if norm == 0:
                norm = np.finfo(v.dtype).eps
            return v / norm

        norm_weighted = speed * normalize(a1) + (1 - speed) * normalize(a2)
        angle = np.arctan2(norm_weighted[1], norm_weighted[0]) % (2 * np.pi)
        magnitude = np.sqrt(norm_weighted[0] ** 2 + norm_weighted[1] ** 2)

        return angle, magnitude

    def generate_mu(
        player_position,
        player_vel,
        player_orient,
        player_accel,
        distance_from_football,
        time_until_land,
        football_x,
        football_y,
        *argv
    ):
        """
        Calulates players catch position given current V and A
        And distance from football

        Parameters
        ----------
        THESE ARE ALL VECTORS np.array([x,y])
        player_position
        player_vel 
        player_orient 
        player_accel 
        
        THESE ARE SCALARS:
        distance_from_football 
        football_x 
        football_y 
        
        Extra inputs for debugging
        *argv : TYPE
            DESCRIPTION.

        Returns
        -------
        np.array([mu_x, mu_y])
        """
        mu_generated = (
            player_position
            + time_until_land * player_vel
            + (time_until_land ** 2) * 0.5 * player_accel
        )

        # if (np.linalg.norm(player_vel + player_accel*time_until_land) > VMAX_MU):
        #     ## reaching beyond max speed
        #     if np.linalg.norm(player_vel) > VMAX_MU \
        #     or np.linalg.norm(player_accel)<0.01 :
        #         mu_generated = player_position + time_until_land*(player_vel)
        #     else:
        #         time_to_max = (VMAX_MU - np.linalg.norm(player_vel))/np.linalg.norm(player_accel)
        #         assert(time_to_max <= time_until_land)
        #         player_max_vel = player_vel + player_accel*time_to_max
        #         time_after_max = time_until_land- time_to_max

        #         mu_generated = player_position + time_to_max*(player_vel) + \
        #             (time_to_max**2)*0.5*player_accel + player_max_vel*time_after_max

        # else: ## did not reach max speed
        #     mu_generated = player_position + time_until_land*(player_vel) + \
        #         (time_until_land**2)*0.5*player_accel

        assert np.isnan(mu_generated[0]) == False and np.isnan(mu_generated[1]) == False

        return mu_generated[0], mu_generated[1]

    # def radius_influence(x):
    #     assert x >= 0
    #     if x <= 18: return (2 + (3/(18**2))*(x**2))
    #     else: return 5

    def generate_sigma(
        influence_rad,
        player_speed,
        distance_from_football,
        time_until_land,
        player_x,
        football_x,
    ):

        R = np.array(
            [
                [np.cos(influence_rad), -np.sin(influence_rad)],
                [np.sin(influence_rad), np.cos(influence_rad)],
            ]
        )

        speed_ratio = (player_speed ** 2) / (MAX_PLAYER_SPEED ** 2)
        radius_infl = time_until_land * 10 + 5

        S = np.array(
            [
                [radius_infl + (radius_infl * speed_ratio), 0],
                [0, radius_infl - (radius_infl * speed_ratio)],
            ]
        )

        return R @ (S ** 2) @ R.T

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.
    
        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    
        """
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    def generate_data_grid(N=120):
        ## 2D positional data to fit contour
        X = np.linspace(0, MAX_FIELD_X, N)
        Y = np.linspace(0, MAX_FIELD_Y, N)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.zeros(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        return X, Y, pos

    ##END DEFINITION OF FUNCTIONS, use df.apply()
    seq_data = seq_df.copy()
    seq_data["speed_w"] = seq_data.s / MAX_PLAYER_SPEED

    ## Acquire speed
    seq_data["vel_x"], seq_data["vel_y"] = zip(
        *seq_data.apply(lambda row: polar_to_z(row.s, row.dir_rad), axis=1)
    )

    ## Acquire orientation
    seq_data["orient_x"], seq_data["orient_y"] = zip(
        *seq_data.apply(lambda row: polar_to_z(ORIENT_MAG, row.o_rad), axis=1)
    )

    ## Acquire weighted angle and magnitude
    seq_data["influence_rad"], seq_data["influence_mag"] = zip(
        *seq_data.apply(
            lambda row: weighted_angle_magnitude(
                np.array([row.vel_x, row.vel_y]),
                np.array([row.orient_x, row.orient_y]),
                row.speed_w,
            ),
            axis=1,
        )
    )
    ## Calculate change in velocity
    seq_data["dvel_x"] = 10 * (
        seq_data["vel_x"] - seq_data.groupby("nflId")["vel_x"].shift(1)
    )
    seq_data["dvel_y"] = 10 * (
        seq_data["vel_y"] - seq_data.groupby("nflId")["vel_y"].shift(1)
    )

    seq_data["dvel_x"] = seq_data["dvel_x"].fillna(0)
    seq_data["dvel_y"] = seq_data["dvel_y"].fillna(0)

    ## Merge with football's location per frame
    seq_data = seq_data.merge(
        seq_data[seq_data.team == "football"][["frameId", "x", "y"]].rename(
            columns={"x": "x_fb", "y": "y_fb"}
        ),
        on="frameId",
    )
    seq_data = seq_data[seq_data["team"] != "football"]
    ## Calculate distance to football
    seq_data["distance_from_football"] = np.sqrt(
        (seq_data.x - seq_data.x_fb) ** 2 + (seq_data.y - seq_data.y_fb) ** 2
    )

    ## Add frames since punt
    punt_frame = seq_data[seq_data["event"] == "punt"]["frameId"].iloc[0]
    seq_data["frames_since_punt"] = seq_data["frameId"] - punt_frame
    seq_data.loc[seq_data["frames_since_punt"] < 0, "frames_since_punt"] = 0
    seq_data["time_until_land"] = AIRTIME - seq_data["frames_since_punt"] * 10.0
    seq_data["time_until_land"] = seq_data["time_until_land"].clip(
        lower=0, upper=AIRTIME
    )

    ## Generate mu
    seq_data["mu_x"], seq_data["mu_y"] = zip(
        *seq_data.apply(
            lambda row: generate_mu(
                np.array([row.x, row.y]),
                np.array([row.vel_x, row.vel_y]),
                np.array([row.orient_x, row.orient_y]),
                np.array([row.dvel_x, row.dvel_y]),
                row.distance_from_football,
                row.time_until_land,
                row.x_fb,
                row.y_fb,
            ),
            axis=1,
        )
    )
    ## Generated sigma
    seq_data["sigma_mtx"] = seq_data.apply(
        lambda row: generate_sigma(
            row.influence_rad,
            row.s,
            row.distance_from_football,
            row.frames_since_punt,
            row.x,
            row.x_fb,
        ),
        axis=1,
    )

    #### DONE WITH ALL THE VECTORS PER PLAYER PER FRAME

    ## Now we generate a grid and use the multivarian gaussian
    ## Track influence in a tensor of Z value
    X, Y, GRID = generate_data_grid(N=N_GRID)
    Z_off = np.zeros((seq_data.frameId.max() + 1, N_GRID, N_GRID))
    Z_def = np.zeros((seq_data.frameId.max() + 1, N_GRID, N_GRID))
    Z_indiv = np.zeros(
        (
            seq_data.frameId.max() + 1,
            2,
            int(seq_data.jerseyRank.max() + 1),
            N_GRID,
            N_GRID,
        )
    )

    for _, row in seq_data.iterrows():
        if row.team == "football":
            continue

        Z = multivariate_gaussian(GRID, np.array([row.mu_x, row.mu_y]), row.sigma_mtx)
        Z_coarse = np.where(Z > 0.0001, Z, 0)
        if np.count_nonzero(Z_coarse) == 0:
            Z_coarse = np.where(Z >= Z.max(), Z, 0)
        if row.isKicking == 1:
            Z_norm = Z_coarse / (Z.max() * SCALE_OFFENCE)
            Z_off[row.frameId, :, :] = np.maximum(Z_off[row.frameId, :, :], Z_norm)
            Z_indiv[row.frameId, 1, row.jerseyRank, :, :] = np.where(
                Z_norm > OFF_CUTOFF, 1, 0
            )

        elif row.isKicking == 0:
            Z_norm = Z_coarse / (Z.max() * SCALE_DEFENCE)
            if row.position in ["MLB", "OLB", "ILB", "LB", "DL", "DE", "NT"]:
                Z_norm = Z_norm ** LB_EXP

            Z_def[row.frameId, :, :] += Z_norm
            Z_indiv[row.frameId, 0, row.jerseyRank, :, :] = np.where(
                Z_norm > 0.01, Z_norm, 0
            )

    Z_diff = np.zeros((seq_data.frameId.max() + 1, N_GRID, N_GRID))

    for i in range(seq_data.frameId.max() + 1):
        Z_diff[i, :, :] = expit(K_EXPIT * (Z_def[i, :, :] - Z_off[i, :, :]))

    ## Get original dataframe subset
    seq_data = seq_data[
        [
            "frameId",
            "nflId",
            "x",
            "y",
            "time_until_land",
            "vel_x",
            "vel_y",
            "dvel_x",
            "dvel_y",
            "mu_x",
            "mu_y",
        ]
    ]

    out_data = namedtuple("Influence", ["Z_tuple", "df_proj",])

    return out_data((Z_diff, Z_indiv), seq_data)


#%%


def calcProjfromtrack(
    seq_df,
    MAX_FIELD_Y=160 / 3,
    MAX_FIELD_X=120,
    MAX_PLAYER_SPEED=11.3,
    ORIENT_MAG=2,
    AIRTIME=4.0,
    VMAX_MU=9.5,
    LB_EXP=2,
    OFF_CUTOFF=0.9,
    N_GRID=121,
    K_EXPIT=3,
    SCALE_OFFENCE=1,
    SCALE_DEFENCE=1,
):
    """
    Calculates catchability and coverage for both teams.
    Parameters
    ----------
    seq_df : DATAFRAME Single play tracking data preprocessed by NFLData
    Can be used for only a few frames rather than entire play eg.
    seq_df =  play_df[play_df.frameId.isin([34,35,36,37])]

    Returns
    -------
    TYPE
        Z_diff : Numpy Array  (N by N array indexed by maxframeID)
        Z_indiv: Numpy Array (N by N by maxframeId by jerseyRank, by isKicking)

    """
    assert not seq_df.empty

    def polar_to_z(r, theta):
        """returns x and y components given length and angle of vector"""
        xy = r * np.exp(1j * theta)
        return np.real(xy), np.imag(xy)

    def weighted_angle_magnitude(a1, a2, speed):
        """returns weighted angle betwen 2 vectors"""
        if np.isnan(a1[0]):
            return np.nan, np.nan

        def normalize(v):
            norm = np.linalg.norm(v, ord=None)
            if norm == 0:
                norm = np.finfo(v.dtype).eps
            return v / norm

        norm_weighted = speed * normalize(a1) + (1 - speed) * normalize(a2)
        angle = np.arctan2(norm_weighted[1], norm_weighted[0]) % (2 * np.pi)
        magnitude = np.sqrt(norm_weighted[0] ** 2 + norm_weighted[1] ** 2)

        return angle, magnitude

    def generate_mu(
        player_position,
        player_vel,
        player_orient,
        player_accel,
        distance_from_football,
        time_until_land,
        football_x,
        football_y,
        *argv
    ):
        """
        Calulates players catch position given current V and A
        And distance from football

        Parameters
        ----------
        THESE ARE ALL VECTORS np.array([x,y])
        player_position
        player_vel 
        player_orient 
        player_accel 
        
        THESE ARE SCALARS:
        distance_from_football 
        football_x 
        football_y 
        
        Extra inputs for debugging
        *argv : TYPE
            DESCRIPTION.

        Returns
        -------
        np.array([mu_x, mu_y])
        """
        if np.linalg.norm(player_vel + player_accel * time_until_land) > VMAX_MU:
            ## reaching beyond max speed
            if (
                np.linalg.norm(player_vel) > VMAX_MU
                or np.linalg.norm(player_accel) < 0.01
            ):  ## no more acceleration remaining
                mu_generated = player_position + time_until_land * (player_vel)
            else:
                time_to_max = (VMAX_MU - np.linalg.norm(player_vel)) / np.linalg.norm(
                    player_accel
                )
                assert time_to_max <= time_until_land
                player_max_vel = player_vel + player_accel * time_to_max
                time_after_max = time_until_land - time_to_max

                mu_generated = (
                    player_position
                    + time_to_max * (player_vel)
                    + (time_to_max ** 2) * 0.5 * player_accel
                    + player_max_vel * time_after_max
                )

        else:  ## did not reach max speed
            mu_generated = (
                player_position
                + time_until_land * (player_vel)
                + (time_until_land ** 2) * 0.5 * player_accel
            )

        assert np.isnan(mu_generated[0]) == False and np.isnan(mu_generated[1]) == False

        return mu_generated[0], mu_generated[1]

    # def radius_influence(x):
    #     assert x >= 0
    #     if x <= 18: return (2 + (3/(18**2))*(x**2))
    #     else: return 5

    def generate_sigma(
        influence_rad,
        player_speed,
        distance_from_football,
        time_until_land,
        player_x,
        football_x,
    ):

        R = np.array(
            [
                [np.cos(influence_rad), -np.sin(influence_rad)],
                [np.sin(influence_rad), np.cos(influence_rad)],
            ]
        )

        speed_ratio = (player_speed ** 2) / (MAX_PLAYER_SPEED ** 2)
        radius_infl = time_until_land * 10 + 5

        S = np.array(
            [
                [radius_infl + (radius_infl * speed_ratio), 0],
                [0, radius_infl - (radius_infl * speed_ratio)],
            ]
        )

        return R @ (S ** 2) @ R.T

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.
    
        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    
        """
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    def generate_data_grid(N=120):
        ## 2D positional data to fit contour
        X = np.linspace(0, MAX_FIELD_X, N)
        Y = np.linspace(0, MAX_FIELD_Y, N)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.zeros(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        return X, Y, pos

    ##END DEFINITION OF FUNCTIONS, use df.apply()

    seq_data = seq_df[
        (seq_df.isReturner == 1)
        | (seq_df.isGunner == 1)
        | (seq_df.isVise == 1)  ## ISNEW
        | (seq_df.displayName == "football")
    ].copy()

    seq_data["speed_w"] = seq_data.s / MAX_PLAYER_SPEED
    seq_data["da_dt"] = 10 * (seq_data["a"] - seq_data.groupby("nflId")["a"].shift(1))
    ## Acquire speed
    seq_data["vel_x"], seq_data["vel_y"] = zip(
        *seq_data.apply(lambda row: polar_to_z(row.s, row.dir_rad), axis=1)
    )

    ## Acquire orientation
    seq_data["orient_x"], seq_data["orient_y"] = zip(
        *seq_data.apply(lambda row: polar_to_z(ORIENT_MAG, row.o_rad), axis=1)
    )

    ## Acquire weighted angle and magnitude
    seq_data["influence_rad"], seq_data["influence_mag"] = zip(
        *seq_data.apply(
            lambda row: weighted_angle_magnitude(
                np.array([row.vel_x, row.vel_y]),
                np.array([row.orient_x, row.orient_y]),
                row.speed_w,
            ),
            axis=1,
        )
    )
    ## Calculate change in velocity
    seq_data["dvel_x"] = 10 * (
        seq_data["vel_x"] - seq_data.groupby("nflId")["vel_x"].shift(1)
    )
    seq_data["dvel_y"] = 10 * (
        seq_data["vel_y"] - seq_data.groupby("nflId")["vel_y"].shift(1)
    )

    seq_data["dvel_x"] = seq_data["dvel_x"].fillna(0)
    seq_data["dvel_y"] = seq_data["dvel_y"].fillna(0)

    seq_data["ddvel_x"] = 10 * (
        seq_data["dvel_x"] - seq_data.groupby("nflId")["dvel_x"].shift(1)
    )
    seq_data["ddvel_y"] = 10 * (
        seq_data["dvel_y"] - seq_data.groupby("nflId")["dvel_y"].shift(1)
    )
    seq_data["ddvel_x"] = seq_data["ddvel_x"].fillna(0)
    seq_data["ddvel_y"] = seq_data["ddvel_y"].fillna(0)

    ## Merge with football's location per frame
    seq_data = seq_data.merge(
        seq_data[seq_data.team == "football"][["frameId", "x", "y"]].rename(
            columns={"x": "x_fb", "y": "y_fb"}
        ),
        on="frameId",
    )

    ## Calculate distance to football
    seq_data["distance_from_football"] = np.sqrt(
        (seq_data.x - seq_data.x_fb) ** 2 + (seq_data.y - seq_data.y_fb) ** 2
    )
    seq_data = seq_data[seq_data.displayName != "football"]
    ## Generate mu
    seq_data["mu_x"], seq_data["mu_y"] = zip(
        *seq_data.apply(
            lambda row: generate_mu(
                np.array([row.x, row.y]),
                np.array([row.vel_x, row.vel_y]),
                np.array([row.orient_x, row.orient_y]),
                np.array([row.dvel_x, row.dvel_y]),
                row.distance_from_football,
                row["time_until_end"],
                row.x_fb,
                row.y_fb,
            ),
            axis=1,
        )
    )

    return seq_data.drop(
        columns=[
            "time",
            "week",
            "quarter",
            "down",
            "yardsToGo",
            "yardsToEndzone",
            "gameClock",
            "possessionTeam",
            "gunCount",
            "viseCount",
            "kickType",
            "kickingTeamScore",
            "receivingTeamScore",
        ]
    )


#%%
if __name__ == "__main__":
    path_to_data = "input/nfl-big-data-bowl-2022/"
    path_to_punts = "input/BDB2022-custom/punts_only/"
    data = NFLData(data_path=path_to_data, punts_path=path_to_punts)
    one_data = data.one_play(59)
    out = calcProjfromtrack(one_data["track"])
    print(out.displayName.unique())
    print(out.shape, out.head())


# %%

#%% Imports

from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Arrow, FancyArrowPatch, Rectangle
from scipy import optimize

from b1_load_data import NFLData, NotAFullPuntError
from b3_fit_control_data import calcProjfromtrack

#%% Temporary field plot
MAX_FIELD_X = 120
MAX_FIELD_Y = 160 / 3


class FieldSetup:
    def __init__(self, play_df=None, play_text=None, plot_size_len=20) -> None:
        """Initializes the datasets used to animate the play.

        Parameters
        ----------
        play_df : DataFrame
            Dataframe corresponding to the play information for the play that requires
            animation. This data will come from the weeks dataframe and contains position
            and velocity information for each of the players and the football.

        Returns
        -------
        None
        """
        self._MAX_FIELD_Y = 160 / 3
        self._MAX_FIELD_X = 120
        self._MAX_FIELD_PLAYERS = 22

        self._CPLT = sns.color_palette("husl", 2)
        self._play_df = play_df

        self.fig = plt.figure(
            figsize=(
                plot_size_len,
                plot_size_len * (self._MAX_FIELD_Y / self._MAX_FIELD_X),
            )
        )

        self._ax_field = plt.gca()
        self._ax_field = self.setup_plot()

        self.base = self._ax_field.twinx()
        self.top = self._ax_field.twinx()

        plt.close()

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])

    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp(1j * theta)

    def setup_plot(self):
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self._ax_field.add_patch(
            Rectangle(
                (0, 0),
                10,
                53.3,
                linewidth=0.1,
                edgecolor="r",
                facecolor="slategray",
                alpha=0.2,
                zorder=0,
            )
        )

        self._ax_field.add_patch(
            Rectangle(
                (110, 0),
                120,
                53.3,
                linewidth=0.1,
                edgecolor="r",
                facecolor="slategray",
                alpha=0.2,
                zorder=0,
            )
        )

        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            self._ax_field.text(
                x,
                12,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,
                fontname="Times New Roman",
                color="slategray",
            )
            self._ax_field.text(
                x - 0.3,
                53.3 - 12,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,
                fontname="Times New Roman",
                color="slategray",
                rotation=180,
            )

        for x in range(11, 110):
            self._ax_field.plot([x, x], [0.4, 0.7], color="lightgray")
            self._ax_field.plot([x, x], [53.0, 52.5], color="lightgray")
            self._ax_field.plot([x, x], [22.91, 23.57], color="lightgray")
            self._ax_field.plot([x, x], [29.73, 30.39], color="lightgray")

        for idx in range(10, 120, 10):
            self._ax_field.axvline(idx, color="k", linestyle="-", alpha=0.05)
        ## used to be s = 100 and s = 500

        return self._ax_field


#%% Make countour z
def generate_data_grid(N=120):
    # Our 2-dimensional distribution will be over variables X and Y
    X = np.linspace(0, MAX_FIELD_X, N)
    Y = np.linspace(0, MAX_FIELD_Y, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.zeros(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    return X, Y, pos


def multivariate_gaussian(pos, mu, Sigma, thresh=0.0001):
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
    Z = np.exp(-fac / 2) / N

    if thresh is not None:
        Z = np.where(Z > thresh, Z, 0)

    return Z


class BayesZArr:
    """
    calculates a contour based on projected locations and a bayesian prior
    """

    def __init__(
        self, df_track, play_row, df_proj, punts_saved, N_GRID=121, N_PRIORS=25
    ):
        self.df_proj = df_proj
        self.df_track = df_track
        self.play_row = play_row
        self.N_priors = N_PRIORS
        self.punts_saved = punts_saved

        self.xp, self.yp, self.xyp = self.init_priors()
        self.gridx, self.gridy, self.gridxy = self.generate_xy_grid(N_GRID)

        self.Z = np.zeros((int(df_proj.frameId.max() + 1), 1, N_GRID, N_GRID,))

    def init_priors(self):
        try:
            punts_sub = self.punts_saved.query(
                f"punterId == {self.play_row['punterId']} & \
                kickType == '{self.play_row['kickType']}' & \
                kickDirectionIntended == '{self.play_row['kickDirectionIntended']}'"
            ).copy()
        except NameError:
            raise NotAFullPuntError

        if punts_sub.empty:
            punts_sub = self.punts_saved.query(
                f"kickType == '{self.play_row['kickType']}' & \
                kickDirectionIntended == '{self.play_row['kickDirectionIntended']}'"
            ).copy()

        punts_sub["rowYardsTo"] = self.play_row["yardsToEndzone"]

        punts_sub["diffYardsTo"] = np.abs(
            punts_sub["yardsToEndzone"] - punts_sub["rowYardsTo"]
        )

        priors_df = punts_sub.sort_values(by="diffYardsTo")

        if priors_df.shape[0] < self.N_priors:
            priors_df = pd.concat(
                [priors_df] * int(self.N_priors / punts_sub.shape[0] + 1)
            )

        priors_df = priors_df.iloc[: self.N_priors]

        xp = priors_df["football_lx"]
        yp = priors_df["football_ly"]

        return xp, yp, np.array([[a, b] for a, b in zip(xp.ravel(), yp.ravel())])

    def generate_xy_grid(self, N_GRID):
        X = np.linspace(0, MAX_FIELD_X, N_GRID)
        Y = np.linspace(0, MAX_FIELD_Y, N_GRID)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.zeros(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        return X, Y, pos

    def calc_center(self):
        df_returner = self.df_proj[self.df_proj["isReturner"] == 1]
        df_returner = df_returner.set_index("frameId")
        posterior_x = self.xp
        posterior_y = self.yp

        for cut_frame in range(df_returner.index.min(), df_returner.index.max() + 1):

            evidence = df_returner.loc[cut_frame][["mu_x", "mu_y"]]
            posterior_x = np.append(posterior_x, evidence["mu_x"])
            posterior_y = np.append(posterior_y, evidence["mu_y"])

            df_returner.loc[cut_frame, "post_x"] = posterior_x.mean()
            df_returner.loc[cut_frame, "post_y"] = posterior_y.mean()

        df_returner = df_returner.reset_index()
        return df_returner[
            [
                "gameId",
                "playId",
                "frameId",
                "time_since_snap",
                "time_until_end",
                "mu_x",
                "mu_y",
                "post_x",
                "post_y",
            ]
        ]

    def calc_heatmap(
        self,
        jerseyRank=0,
        singleSigma=np.array([[3, 0], [0, 3]]),
        evidence_multiplier=2,
    ):
        ## add all priors on heatmap
        for px, py in self.xyp:
            Z = multivariate_gaussian(self.gridxy, np.array([px, py]), singleSigma)
            Z_coarse = np.where(Z > 0.0001, Z, 0)
            self.Z[0, jerseyRank, :, :] += Z_coarse

        df_returner = self.df_proj[self.df_proj["isReturner"] == 1]
        df_returner = df_returner.set_index("frameId")

        for cut_frame in range(df_returner.index.min(), df_returner.index.max() + 1):

            if cut_frame != 1 and cut_frame == df_returner.index.min():
                self.Z[cut_frame, jerseyRank, :, :] += self.Z[0, jerseyRank, :, :]

            self.Z[cut_frame, jerseyRank, :, :] += self.Z[
                cut_frame - 1, jerseyRank, :, :
            ]
            evidence_x = df_returner.loc[cut_frame]["mu_x"]
            evidence_y = df_returner.loc[cut_frame]["mu_y"]
            Z = multivariate_gaussian(
                self.gridxy,
                np.array([evidence_x, evidence_y]),
                singleSigma * evidence_multiplier,
            )

            Z_coarse = np.where(Z > 0.001, Z, 0)
            self.Z[cut_frame, jerseyRank, :, :] += Z_coarse


#%%
if __name__ == "__main__":

    path_to_data = "input/nfl-big-data-bowl-2022/"
    path_to_punts = "input/BDB2022-custom/punts_only/"
    data = NFLData(data_path=path_to_data, punts_path=path_to_punts)

    saved_punts = pd.read_csv("input/BDB2022-custom/punts_from_2018.csv", index_col=0)
    #%% Evidence data from 1 play
    one_data = data.one_play(51)
    one_track = one_data["track"]
    one_row = one_data["row"]
    calc_proj = calcProjfromtrack(one_track)

    #%%

    BA = BayesZArr(one_track, one_row, calc_proj, saved_punts)

    #%%

    priors_xy = BA.xyp
    BA.calc_heatmap()

    #%%
    #%% Plot priors data
    scrimmage = 25  ## not right
    field = FieldSetup()
    ax = field.setup_plot()
    ax.plot([scrimmage, scrimmage], [0, 160 / 3])
    ax.scatter(x=BA.xp.ravel(), y=BA.yp.ravel())
    field.fig

    # %% Plot evidence data
    field = FieldSetup()
    ax = field.setup_plot()
    sns.scatterplot(data=BA.df_proj, x="mu_x", y="mu_y", s=100, hue="frameId", ax=ax)
    field.fig
    #%%
    #%% Plot contourf
    field = FieldSetup()
    ax = field.setup_plot()

    ax.plot([scrimmage, scrimmage], [0, 160 / 3])
    ax.contour(BA.gridx, BA.gridy, BA.Z[25, 0, :, :])
    field.fig


# %%

# %%

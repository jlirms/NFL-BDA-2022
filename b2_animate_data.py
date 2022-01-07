#%% One play animation

import math
import os
import typing
import webbrowser

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import HTML
from matplotlib import animation
from matplotlib.cm import get_cmap
from matplotlib.patches import Arrow, FancyArrowPatch, Rectangle
from scipy.special import expit
from sklearn.base import BaseEstimator

from b1_load_data import NFLData


#%%
def open_html_plot(animation, fname="temp_ani.html", open_browser=True):
    with open(fname, "wb") as f:
        f.write((HTML(animation.to_jshtml())).data.encode("UTF-8"))
    if open_browser == True:
        webbrowser.get().open("file://" + os.path.realpath(fname), new=2)


class AnimatePlay:
    def __init__(self, play_df, play_text=None, plot_size_len=20) -> None:
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
        self._MAX_FIELD_Y = 53.3
        self._MAX_FIELD_X = 120
        self._MAX_FIELD_PLAYERS = 22

        self._CPLT = sns.color_palette("husl", 2)
        self._play_df = play_df
        self._times = sorted(play_df.time.unique())
        self._stream = self.data_stream()

        self._date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        self._mean_interval_ms = 100

        self._fig = plt.figure(
            figsize=(
                plot_size_len,
                plot_size_len * (self._MAX_FIELD_Y / self._MAX_FIELD_X),
            )
        )

        self._ax_field = plt.gca()

        self._ax_home = self._ax_field.twinx()
        self._ax_away = self._ax_field.twinx()
        self._ax_jersey = self._ax_field.twinx()

        self.ani = animation.FuncAnimation(
            self._fig,
            self.update,
            frames=len(self._times),
            interval=self._mean_interval_ms,
            init_func=self.setup_plot,
            blit=False,
        )
        self._play_text = play_text
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

    def data_stream(self):
        for atime in self._times:
            yield self._play_df[self._play_df.time == atime]

    def setup_plot(self):
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        ball_snap_ser = self._play_df[
            (self._play_df.event == "ball_snap") & (self._play_df.team == "football")
        ].iloc[0]
        self._ax_field.axvline(ball_snap_ser.x, color="k", linestyle="--")

        if ball_snap_ser.yardsToEndzone <= ball_snap_ser.yardsToGo:
            self._ax_field.axvline(110, color="yellow", lw=4, linestyle="-")
        else:
            self._ax_field.axvline(
                ball_snap_ser.x + ball_snap_ser.yardsToGo,
                color="yellow",
                lw=4,
                linestyle="-",
            )

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

        self._ax_field.text(
            100,
            55.3,
            "G {game} | P {play}".format(
                game=ball_snap_ser.gameId, play=ball_snap_ser.playId
            ),
        )

        downst = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        self._ax_field.text(
            0,
            55.3,
            "Q: {quarter} | Time: {minutes}:{seconds}".format(
                quarter=ball_snap_ser.quarter,
                minutes=ball_snap_ser.gameClock.seconds // 60,
                seconds=(ball_snap_ser.gameClock.seconds % 60),
            ),
        )

        if ball_snap_ser.yardsToEndzone <= ball_snap_ser.yardsToGo:
            dista = "Goal"
        else:
            dista = ball_snap_ser.yardsToGo

        self._ax_field.text(
            0,
            53.8,
            "{down} and {dist}".format(down=downst[ball_snap_ser.down], dist=dista),
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

        if self._play_text is not None:
            self._ax_field.text(10, 53.8, self._play_text)

        self._ax_field.text(54.8, 56.3, "Frame: ")
        self._frameId_text = self._ax_field.text(59, 56.3, "")

        self.set_axis_plots(self._ax_home, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_away, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        for idx in range(10, 120, 10):
            self._ax_field.axvline(idx, color="k", linestyle="-", alpha=0.05)
        ## used to be s = 100 and s = 500
        self._scat_field = self._ax_field.scatter([], [], s=100, color="black")
        self._scat_home = self._ax_home.scatter(
            [], [], s=500, color=self._CPLT[0], edgecolors="k"
        )
        self._scat_away = self._ax_away.scatter(
            [], [], s=500, color=self._CPLT[1], edgecolors="k"
        )

        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(
                self._ax_jersey.text(
                    0,
                    0,
                    "",
                    horizontalalignment="center",
                    verticalalignment="center",
                    c="white",
                )
            )
            self._scat_number_list.append(
                self._ax_jersey.text(
                    0,
                    0,
                    "",
                    horizontalalignment="center",
                    verticalalignment="center",
                    c="black",
                )
            )
            self._scat_name_list.append(
                self._ax_jersey.text(
                    0,
                    0,
                    "",
                    horizontalalignment="center",
                    verticalalignment="center",
                    c="black",
                )
            )

            self._a_dir_list.append(
                self._ax_field.add_patch(Arrow(0, 0, 0, 0, color="k"))
            )
            self._a_or_list.append(
                self._ax_field.add_patch(Arrow(0, 0, 0, 0, color="k"))
            )

        return (
            self._scat_field,
            self._scat_home,
            self._scat_away,
            *self._scat_jersey_list,
            *self._scat_number_list,
            *self._scat_name_list,
        )

    def update(self, anim_frame):
        pos_df = next(self._stream)
        self._frameId_text.set_text(pos_df.frameId.iloc[0])

        for label in pos_df.team.unique():
            label_data = pos_df[pos_df.team == label]

            if label == "football":
                self._scat_field.set_offsets(np.hstack([label_data.x, label_data.y]))
            elif label == "home":
                self._scat_home.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == "away":
                self._scat_away.set_offsets(np.vstack([label_data.x, label_data.y]).T)

        for (index, row) in pos_df[pos_df.position.notnull()].reset_index().iterrows():
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y + 1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            self._scat_name_list[index].set_position((row.x, row.y - 1.9))
            self._scat_name_list[index].set_text(row.displayName.split()[-1])

            player_orientation_rad = row.o_rad
            player_direction_rad = row.dir_rad
            player_speed = row.s

            player_vel = np.array(
                [
                    np.real(self.polar_to_z(player_speed, player_direction_rad)),
                    np.imag(self.polar_to_z(player_speed, player_direction_rad)),
                ]
            )
            player_orient = np.array(
                [
                    np.real(self.polar_to_z(2, player_orientation_rad)),
                    np.imag(self.polar_to_z(2, player_orientation_rad)),
                ]
            )

            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self._ax_field.add_patch(
                Arrow(row.x, row.y, player_vel[0], player_vel[1], color="k")
            )
            self._a_or_list[index].remove()
            self._a_or_list[index] = self._ax_field.add_patch(
                Arrow(
                    row.x,
                    row.y,
                    player_orient[0],
                    player_orient[1],
                    color="grey",
                    width=2,
                )
            )

        return (
            self._scat_field,
            self._scat_home,
            self._scat_away,
            *self._scat_jersey_list,
            *self._scat_number_list,
            *self._scat_name_list,
        )


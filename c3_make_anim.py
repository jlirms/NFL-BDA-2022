#%% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.cm import get_cmap
from matplotlib.patches import Arrow, FancyArrowPatch, Rectangle

plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

from b1_load_data import NFLData
from b2_animate_data import AnimatePlay, open_html_plot
from b3_fit_control_data import calcProjfromtrack
from b4_heatmap_anim import getEventsfromtrack, getGVConnfromtrack, getNamesdffromtrack
from b5_single_bell import BayesZArr
from b6_heatmap_contour import AnimatePlayPitchControl

#%%

path_to_data = "input/nfl-big-data-bowl-2022/"
path_to_punts = "input/BDB2022-custom/punts_only/"
saved_punts = pd.read_csv("input/BDB2022-custom/punts_from_2018.csv", index_col=0)

data = NFLData(data_path=path_to_data, punts_path=path_to_punts)

for newplay in [
    4139,
]:
    one_data = data.one_play(newplay)
    calc_proj = calcProjfromtrack(one_data["track"])

    BA = BayesZArr(
        one_data["track"], one_data["row"], calc_proj, saved_punts, N_PRIORS=10
    )
    BA.calc_heatmap()

    #%%
    names = getNamesdffromtrack(one_data["track"])
    events = getEventsfromtrack(one_data["track"])

    #%%
    new_anim = AnimatePlayPitchControl(
        one_data["track"],
        BA.Z,
        play_text=one_data["row"].playDescription,
        names=names,
        events=events,
    )

    #%%

    writer = animation.FFMpegWriter(fps=10)

    new_anim.ani.save(f"plots/TEMP{newplay}.mp4", writer=writer)
# %%

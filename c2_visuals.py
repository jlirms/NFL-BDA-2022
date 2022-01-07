#%% Imports
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.core.numeric import full

from b1_load_data import NFLData, NotAFullPuntError

#%% Read Data
path_to_data = "input/nfl-big-data-bowl-2022/"
path_to_punts = "input/BDB2022-custom/punts_only/"
saved_punts = pd.read_csv("input/BDB2022-custom/punts_from_2019.csv", index_col=0)

data = NFLData(data_path=path_to_data, punts_path=path_to_punts)
players = pd.read_csv("input/nfl-big-data-bowl-2022/players.csv")
#%% Subset 2019 Week 1 to 7

saved_punts = saved_punts.query("week >= 1 & week <= 7")

# %%
common_cols = [
    "gameId",
    "playId",
    "down",
    "yardsToGo",
    "yardsToEndzone",
    "quarter",
    "secondsUntilHalf",
    "specialTeamsResult",
    "football_lx",
    "football_ly",
]

r_cols = [
    "rGunnerId",
    "rGunner_y",
    "rGunner_vises",
    "rVises1",
    "rVises2",
    "rGunner_lx",
    "rGunner_ly",
]

l_cols = [
    "lGunnerId",
    "lGunner_y",
    "lGunner_vises",
    "lVises1",
    "lVises2",
    "lGunner_lx",
    "lGunner_ly",
]

half_r = saved_punts[common_cols + r_cols].rename(
    columns={sname: sname[1:] for sname in r_cols}
)
half_r["RSIDE"] = 1
half_r["LSIDE"] = 0
half_l = saved_punts[common_cols + l_cols].rename(
    columns={sname: sname[1:] for sname in l_cols}
)
half_l["RSIDE"] = 0
half_l["LSIDE"] = 1

gunners = pd.concat([half_r, half_l])
# %%

full_names = [
    ["Bethel", "Hardee", "Mostert", "Holton", "Ford", "Slater", "Robinson", "Bellamy"],
    [
        "Wilson",
        "Cruikshank",
        "Patterson",
        "Sherfield",
        "Thomas",
        "Goodwin",
        "Amadi",
        "Brown",
    ],
    ["Jamerson", "Bunting", "Virgin", "Aikens", "Core", "Pringle", "Nixon", "Gage"],
    ["Abdullah", "Hodge", "Facyson", "Johnson", "Apke", "Moore", "Brown", "Odum"],
]

for row in full_names:
    for name in row:
        query = players[players.displayName.str.contains(name)]
        if query.shape[0] >= 1:
            print(query.iloc[0].nflId, end=",")
        else:
            print(name, query[["displayName", "nflId"]])
            break

    print("NEW ROW NEW ROW\n")

#%%

gunners_array = np.array(
    [
        [38707, 45648, 42718, 43988, 45021, 33234, 38596, 39222,],
        [45020, 46221, 39975, 46671, 43495, 42094, 47915, 45797,],
        [46233, 47822, 45657, 41354, 43488, 46522, 48241, 46263],
        [42397, 46992, 46729, 42359, 46178, 43396, 46724, 46349],
    ]
)

gunners_tracking = {nflid: [] for nflid in gunners_array.ravel()}


#%%

for nflid in gunners_tracking.keys():
    one_gun = gunners[gunners.GunnerId == nflid].copy()
    one_gun["dx"] = one_gun["Gunner_lx"] - one_gun["football_lx"]
    one_gun["dy"] = one_gun["Gunner_ly"] - one_gun["football_ly"]
    one_gun["c"] = (one_gun["specialTeamsResult"] == "Return").astype(int)
    gunners_tracking[nflid] = one_gun[["dy", "dx", "c"]].dropna().values

#%%

fig, ax = plt.subplots(4, 8, figsize=(23, 15))
for r in range(gunners_array.shape[0]):
    for c in range(gunners_array.shape[1]):
        dta = gunners_tracking[gunners_array[r][c]]
        title_str = f"{full_names[r][c]} \n V:{np.mean(np.abs(dta[:,0])):.2f}, H:{np.mean(np.abs(dta[:,1])):.2f}"
        ax[r][c].scatter([0,], [0,], s=300, c="black", alpha=0.5)
        scatter = ax[r][c].scatter(dta[:, 0], dta[:, 1], c=dta[:, 2], cmap=cm.bwr)
        ax[r][c].set_aspect(1)
        ax[r][c].set(xlim=[-20, 20], ylim=[-30, 20], title=title_str)
        if (r != 0 or c != 0) and not (r == 3 and c == 7):
            ax[r][c].set_xticklabels([])
            ax[r][c].set_yticklabels([])
        else:
            legend = ax[r][c].legend(
                *scatter.legend_elements(), title="isReturned", loc="lower right"
            )
        ax[r][c].grid(axis="y", ls="--")
        ax[r][c].grid(axis="x", ls="--")


# plt.show()

#%%

fig.savefig("FullReturning.png", dpi=1600, bbox_inches="tight")


#%%
gunners_tracking2 = {nflid: [] for nflid in gunners_array.ravel()}

for nflid in gunners_array.ravel()[:2]:

    sing_gunner = gunners.query(f"GunnerId == {nflid} & Gunner_vises == 1")

    all_collected_paths = []
    for si in sing_gunner.index:
        one_play = data.one_play(si)
        row = one_play["row"]
        track = one_play["track"]
        #%%
        gtrack = track.query(f"nflId == {nflid} & time_since_snap <= 4.5")[
            ["x", "y", "s", "a"]
        ]
        gtrack["ox"] = gtrack["x"] - gtrack["x"].iloc[0]
        gtrack["oy"] = 160 / 6 - gtrack["y"]

        gtrack["px"] = gtrack["oy"]
        gtrack["py"] = gtrack["ox"]
        gtrack["pid"] = si
        gunners_tracking2[nflid].append(gtrack[["px", "py", "s", "a"]])

#%%

# with open("input/BDB2022-custom/gunners0tracking.pickle", "wb") as handle:
#     pickle.dump(gunners_tracking2, handle)

#%%

with open("input/BDB2022-custom/gunners0tracking.pickle", "rb") as handle:
    gt2 = pickle.load(handle)

# %%
from matplotlib import colors

gunner_collected_paths = gunners_tracking2[45648]

fig, ax = plt.subplots(figsize=(3, 3))

for gtrack in gunner_collected_paths:
    gtrack["dv"] = (gtrack["s"] - gtrack["s"].shift(1)).fillna(0)
    scatter = ax.scatter(
        "px",
        "py",
        c="dv",
        data=gtrack,
        s=10,
        alpha=0.8,
        norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1),
        cmap=cm.seismic,
    )

legend = ax.legend(*scatter.legend_elements(num=4), title="Accel", loc="upper center")

ax.set(xlim=[-160 / 6, 160 / 6], ylim=[-1, 50])

ax.set_aspect(1)
ax.grid(axis="y", ls="--")

#%%
full_names = [
    ["Bethel", "Hardee", "Mostert", "Holton", "Ford", "Slater", "Robinson", "Bellamy"],
    [
        "Wilson",
        "Cruikshank",
        "Patterson",
        "Sherfield",
        "Thomas",
        "Goodwin",
        "Amadi",
        "Brown",
    ],
    ["Jamerson", "Bunting", "Virgin", "Aikens", "Core", "Pringle", "Nixon", "Gage"],
    ["Abdullah", "Hodge", "Facyson", "Johnson", "Apke", "Moore", "Brown", "Odum"],
]


gunners_array = np.array(
    [
        [38707, 45648, 42718, 43988, 45021, 33234, 38596, 39222,],
        [45020, 46221, 39975, 46671, 43495, 42094, 47915, 45797,],
        [46233, 47822, 45657, 41354, 43488, 46522, 48241, 46263],
        [42397, 46992, 46729, 42359, 46178, 43396, 46724, 46349],
    ]
)

# %%

fig, ax = plt.subplots(4, 8, figsize=(23, 15))
for r in range(gunners_array.shape[0]):
    for c in range(gunners_array.shape[1]):
        gunner_collected_paths = gt2[gunners_array[r][c]]
        title_str = f"{full_names[r][c]}"
        for gtrack in gunner_collected_paths:
            gtrack["dv"] = (gtrack["s"] - gtrack["s"].shift(1)).fillna(0)
            scatter = ax[r][c].scatter(
                "px",
                "py",
                c="dv",
                data=gtrack,
                s=10,
                alpha=0.8,
                norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1),
                cmap=cm.seismic,
            )

        ax[r][c].set(xlim=[-160 / 6, 160 / 6], ylim=[-1, 50])

        ax[r][c].set_aspect(1)
        ax[r][c].set(xlim=[-30, 30], ylim=[0, 50], title=title_str)
        if (r != 0 or c != 0) and not (r == 3 and c == 7):
            ax[r][c].set_xticklabels([])
            ax[r][c].set_yticklabels([])

        if r == 3 and c == 7:
            legend = ax[1][c].legend(
                *scatter.legend_elements(num=5), title="Accel", loc="upper right"
            )

        ax[r][c].grid(axis="y", ls="--")
        ax[r][c].grid(axis="x", ls="--")
# %%

fig.savefig("AccelPaths.png", dpi=1600, bbox_inches="tight")
# %%


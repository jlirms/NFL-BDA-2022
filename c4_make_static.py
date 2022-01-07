#%% Imports

from b7_accel_profiles import *

#%%


def showPlay(one_row, calc_proj, targetdf, SS_SNAP=2):
    scrimmage = 110 - one_row["yardsToEndzone"]
    field = FieldSetup()
    ax = field.setup_plot()
    ax.plot([scrimmage, scrimmage], [0, 160 / 3])

    saved_path_rows = []
    for jers in [0, 1]:

        optimal_time = findOptimalTime(
            calc_proj, targetdf, jerseyRank=jers, SS_SNAP=SS_SNAP
        )
        main_path = findPathUCM(calc_proj, targetdf, jerseyRank=jers, SS_SNAP=SS_SNAP)

        alt_path = findTimeStopPivot(
            calc_proj, targetdf, jerseyRank=jers, SS_SNAP=SS_SNAP
        )

        if main_path is None:  # or main_path.totalTime > alt_path.altTime:
            path = alt_path
            path["totalTime"] = path["altTime"]
        else:
            path = main_path.rename(
                index={
                    "calc_UCMTime": "UCMTime",
                    "calc_RemainingTime": "S3Time",
                    "time": "S1Time",
                }
            )
            path["altTime"] = alt_path.altTime

        path["OptimalTime"] = optimal_time
        FIRE_TIME = round((path.totalTime - optimal_time) / optimal_time, 3)
        path["FIRETIME"] = FIRE_TIME
        if jers == 0:
            path["Side"] = one_row["jersRank0"].upper()
        else:
            path["Side"] = "R" if one_row["jersRank0"] == "l" else "L"

        saved_path_rows.append(path)

        ##now plot
        SX, SY, TX, TY = path.SX, path.SY, path.TX, path.TY
        ax.scatter([SX,], [SY,], c="r", s=100)
        ax.scatter([TX,], [TY,], c="b", s=100)
        if hasattr(path, "Px1"):
            circfunc = getCircleFormula(path.calc_C, path.calcR, TY)
            _x0 = path.Px1
            _y0 = path.Py1

            _x1, _y01 = path.Px2
            # ax.scatter(_x1, _y01, c="pink")
            ax.plot([_x1, TX], [_y01, TY], c="pink")
            ax.plot([SX, _x0], [SY, _y0], c="red")

            _xcurve = np.linspace(min(_x0, _x1), max(_x0, _x1), 2 * int(abs(_x0 - _x1)))
            _ycurve = circfunc(_xcurve)
            ax.plot(_xcurve, _ycurve, c="orange")
            # circle = plt.Circle(path.calc_C, path.calcR, fill=False, lw=0.15)
            # ax.add_patch(circle)
            ax.arrow(SX, SY, path.vx0, path.vy0, head_width=1.25)
            # ax.arrow(_x0, _y0, path.vx1 / 5, path.vy1 / 5, head_width=1.25)

        ax.text(x=10, y=5.5, s=f"Index : {one_row.name}")
        ax.text(x=10, y=4.25, s=one_row.playDescription)
    try:
        dfout = pd.DataFrame(saved_path_rows)[
            [
                "nflId",
                "Name",
                "S1Time",
                "UCMTime",
                "S3Time",
                "totalTime",
                "altTime",
                "OptimalTime",
                "FIRETIME",
                "Side",
            ]
        ].reset_index(drop=True)
    except KeyError:
        dfout = pd.DataFrame(saved_path_rows)
    dfout = dfout.round(2)
    ax.text(
        x=10, y=0.5, s=dfout.to_string(), fontdict={"family": "monospace"},
    )
    print(dfout)

    vises = calc_proj.query("isVise == 1 and time_since_snap == 2")
    for _, row in vises.iterrows():
        ax.arrow(row.x, row.y, row.vel_x, row.vel_y, head_width=1.25)
        ax.scatter(row.x, row.y, c="black", s=100)
    ax.set_yticklabels([])
    print(path.index)
    return field.fig


#%%

path_to_data = "input/nfl-big-data-bowl-2022/"
path_to_punts = "input/BDB2022-custom/punts_only/"
data = NFLData(data_path=path_to_data, punts_path=path_to_punts)

saved_punts = pd.read_csv("input/BDB2022-custom/punts_from_2019.csv", index_col=0)

#%%
one_data = data.one_play(2645)
one_track = one_data["track"]
one_row = one_data["row"]
calc_proj = calcProjfromtrack(one_track)
BA = BayesZArr(one_track, one_row, calc_proj, saved_punts)
targetdf = BA.calc_center()
print(f"Left Gunner {one_row['lGunnerJers']}, Right: {one_row['rGunnerJers']}")

#%%
showPlay(one_row, calc_proj, targetdf, SS_SNAP=2)


# %%


ind = 2358  ## 2702
print(f"----------{ind}---------")
one_data = data.one_play(ind)
one_track = one_data["track"]
one_row = one_data["row"]
calc_proj = calcProjfromtrack(one_track)
BA = BayesZArr(one_track, one_row, calc_proj, saved_punts)
targetdf = BA.calc_center()
showPlay(one_row, calc_proj, targetdf, SS_SNAP=2)

# %%


#%%
ind = np.random.randint(2000, 3000)
ind = 4139  # or 3005 also good
print(f"----------{ind}---------")
one_data = data.one_play(ind)
one_track = one_data["track"]
one_row = one_data["row"]
calc_proj = calcProjfromtrack(one_track)
BA = BayesZArr(one_track, one_row, calc_proj, saved_punts)
targetdf = BA.calc_center()
showPlay(one_row, calc_proj, targetdf, SS_SNAP=2)


#%%

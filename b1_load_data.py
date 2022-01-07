#%% IMPORTS
import math
import os
from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 101)
pd.set_option("display.max_colwidth", 201)
#%%
class NFLDataError(ValueError):
    pass


class NotAFullPuntError(NFLDataError):
    pass


class NFLData:
    """
    A helper class to read NFL source files, join data and standardize directions
    
    .games_df() .plays_df() and .track_df()
    :param data_path: location of data

    
    """

    def __init__(self, data_path, punts_path, punts_only=True):
        self.data_path = data_path
        self.punts_path = punts_path
        self.punts_only = punts_only
        self._dftrack = None
        self._dfplays = None
        self._dfgames = None
        self._dfplayers = None
        self._dfpff = None
        self._dfplays_merged = None
        self._yr_numb = None
        self._wk_numb = None

    def __repr__(self):
        if self._wk_numb is not None:
            return (
                f"Merged Data with tracking from {self._yr_numb} week {self._wk_numb}"
            )
        return "Merged DataFrame of NFL Data"

    def _read_csv(self, filename, index_col=None):
        assert filename.endswith(".csv")
        return pd.read_csv(os.path.join(self.data_path, filename), index_col=index_col)

    def games_df(self):
        if self._dfgames is None:
            print("Reading in Games Data")
            self._dfgames = self._read_csv("games.csv", index_col="gameId")
        return self._dfgames

    def players_df(self):
        if self._dfplayers is None:
            print("Reading in Players Data")
            self._dfplayers = self._read_csv("players.csv")

            _ftin = self._dfplayers.height.str.extract(r"(?P<FT>\d)-(?P<IN>\d+)")
            _ftin = _ftin.dropna().astype(int)

            height = _ftin["FT"] * 12 + _ftin["IN"]
            self._dfplayers.loc[height.index, "height"] = height
            self._dfplayers["height"] = self._dfplayers["height"].astype(int)

        return self._dfplayers

    def pff_df(self):
        if self._dfpff is None:
            print("Reading PFF Scouting Data")
            self._dfpff = self._read_csv("PFFScoutingData.csv")

            self._dfpff = self._dfpff.merge(
                self.plays_df()[["gameId", "playId"]],
                how="inner",
                on=["gameId", "playId"],
            )

        return self._dfpff

    def _gunvice_df(self, column_name="gunners", col_sub="gun"):
        gstr = self.pff_df()[column_name] + ";"
        gnum = gstr.str.findall("(\d+?);")
        _dfgunners = gnum.apply(lambda x: pd.Series(x))
        _dfgunners = _dfgunners.astype(np.float32)

        _dfgunners = _dfgunners.set_index(self.pff_df().index)
        _dfgunners.columns = [f"{col_sub}_{i}" for i in _dfgunners.columns]
        _dfgunners.insert(
            loc=0,
            column=f"{col_sub}Count",
            value=gstr.str.count(";").fillna(0).astype(int),
        )

        return _dfgunners

    def plays_df(self):
        if self._dfplays is None:
            df_games = self.games_df()

            print("Reading in Plays Data")
            self._dfplays = self._read_csv("plays.csv")

            if self.punts_only:
                print("Taking Punts Only")
                self._dfplays = self._dfplays.query('specialTeamsPlayType == "Punt"')

            ## Add game data and team abbreviations
            self._dfplays = self._dfplays.merge(
                df_games[["homeTeamAbbr", "visitorTeamAbbr", "week"]].reset_index(),
                on=["gameId"],
            )

            self._dfplays["year"] = self._dfplays.gameId // 1000000
            ## Dealing with the clock
            _vtemp = self._dfplays["gameClock"].str.split(":", expand=True).astype(int)
            self._dfplays["gameClock"] = pd.to_timedelta(
                _vtemp[0], unit="m"
            ) + pd.to_timedelta(_vtemp[1], unit="s")
            self._dfplays["secondsUntilHalf"] = (
                self._dfplays["quarter"] % 2 * 900
                + self._dfplays["gameClock"].dt.seconds
            )

            ## Dealing with the score
            self._dfplays.loc[
                self._dfplays.possessionTeam == self._dfplays.homeTeamAbbr,
                "kickingTeamScore",
            ] = self._dfplays.preSnapHomeScore
            self._dfplays.loc[
                self._dfplays.possessionTeam == self._dfplays.visitorTeamAbbr,
                "kickingTeamScore",
            ] = self._dfplays.preSnapVisitorScore

            self._dfplays.loc[
                self._dfplays.possessionTeam == self._dfplays.homeTeamAbbr,
                "receivingTeamScore",
            ] = self._dfplays.preSnapVisitorScore
            self._dfplays.loc[
                self._dfplays.possessionTeam == self._dfplays.visitorTeamAbbr,
                "receivingTeamScore",
            ] = self._dfplays.preSnapHomeScore

            self._dfplays.loc[
                self._dfplays.possessionTeam == self._dfplays.yardlineSide,
                "yardsToEndzone",
            ] = (100 - self._dfplays["yardlineNumber"])
            self._dfplays.loc[
                self._dfplays.possessionTeam != self._dfplays.yardlineSide,
                "yardsToEndzone",
            ] = self._dfplays["yardlineNumber"]

            self._dfplays = self._dfplays.sort_values(
                by=["gameId", "playId"]
            ).reset_index(drop=True)
            col_order = [
                "gameId",
                "playId",
                "down",
                "yardsToGo",
                "yardsToEndzone",
                "quarter",
                "specialTeamsPlayType",
                "specialTeamsResult",
                "playDescription",
                "secondsUntilHalf",
                "gameClock",
                "possessionTeam",
                "kickerId",
                "returnerId",
                "kickBlockerId",
                "yardlineSide",
                "yardlineNumber",
                "penaltyCodes",
                "penaltyJerseyNumbers",
                "penaltyYards",
                "preSnapHomeScore",
                "preSnapVisitorScore",
                "passResult",
                "kickLength",
                "kickReturnYardage",
                "playResult",
                "absoluteYardlineNumber",
                "homeTeamAbbr",
                "visitorTeamAbbr",
                "week",
                "year",
                "kickingTeamScore",
                "receivingTeamScore",
            ]

            self._dfplays = self._dfplays[col_order]

        return self._dfplays

    def merged_plays(self):
        if self._dfplays_merged is None:
            pff_to_add = [
                "gameId",
                "playId",
                "snapDetail",
                "snapTime",
                "operationTime",
                "hangTime",
                "kickType",
                "kickDirectionIntended",
                "kickDirectionActual",
                "returnDirectionIntended",
                "returnDirectionActual",
                # 'missedTackler',
                # 'assistTackler',
                # 'tackler',
                # 'kickoffReturnFormation',
                # 'gunners',
                # 'puntRushers',
                # 'specialTeamsSafeties',
                # 'vises',
                "kickContactType",
            ]
            ppf_merged = pd.concat(
                [
                    self.pff_df()[pff_to_add],
                    self._gunvice_df("gunners", "gun"),
                    self._gunvice_df("vises", "vise"),
                ],
                axis=1,
            )

            self._dfplays_merged = self.plays_df().merge(
                ppf_merged, how="inner", on=["gameId", "playId"]
            )

        return self._dfplays_merged

    def track_df(
        self,
        yr_numb=2018,
        wk_numb=1,
        flip_left=True,
        add_target=False,
        target_path=None,
    ):

        if (
            self._dftrack is None
            or self._yr_numb != yr_numb
            or self._wk_numb != wk_numb
        ):
            ## Games data first
            df_games = self.games_df()

            print("Reading in Tracking Data")
            self._dftrack = pd.read_csv(
                self.punts_path + f"tracking{yr_numb}_wk{wk_numb}.csv"
            )
            self._yr_numb = yr_numb
            self._wk_numb = wk_numb

            ## Subset of plays data to merge with tracking data
            play_downinfo = self.merged_plays()[
                [
                    "gameId",
                    "playId",
                    "week",
                    "quarter",
                    "down",
                    "yardsToGo",
                    "yardsToEndzone",
                    "gameClock",
                    "possessionTeam",
                    "snapTime",
                    "operationTime",
                    "hangTime",
                    "gunCount",
                    "viseCount",
                    "kickType",
                    "gun_0",
                    "gun_1",
                    "gun_2",
                    "gun_3",
                    "vise_0",
                    "vise_1",
                    "vise_2",
                    "vise_3",
                    "vise_4",
                    "kickingTeamScore",
                    "receivingTeamScore",
                ]
            ]

            if wk_numb is not None:
                print(f"Taking Subset of Week {wk_numb}")
                play_downinfo = play_downinfo.query(f"week == {wk_numb}")

            self._dftrack = self._dftrack.merge(
                play_downinfo, how="inner", on=["gameId", "playId"]
            )

            ## Standardize everything, add new columns to track players
            self._dftrack["o_rad"] = np.mod(90 - self._dftrack.o, 360) * math.pi / 180.0
            self._dftrack["dir_rad"] = (
                np.mod(90 - self._dftrack.dir, 360) * math.pi / 180.0
            )
            self._dftrack["jerseyNumber"] = (
                self._dftrack["jerseyNumber"].fillna(value=-1).astype(int)
            )
            self._dftrack["nflId"] = self._dftrack["nflId"].fillna(value=-1).astype(int)

            self._dftrack["jerseyRank"] = (
                self._dftrack.groupby(["gameId", "playId", "team"])[
                    "jerseyNumber"
                ].rank("dense")
                - 1
            )
            self._dftrack["jerseyRank"] = (
                self._dftrack["jerseyRank"].fillna(value=-1).astype(int)
            )
            ## Merge with games data, add team names
            self._dftrack = self._dftrack.merge(
                df_games[["homeTeamAbbr", "visitorTeamAbbr"]].reset_index(),
                on=["gameId"],
            )
            self._dftrack.loc[
                self._dftrack.team == "home", "nflTeam"
            ] = self._dftrack.loc[self._dftrack.team == "home"]["homeTeamAbbr"]
            self._dftrack.loc[
                self._dftrack.team == "away", "nflTeam"
            ] = self._dftrack.loc[self._dftrack.team == "away"]["visitorTeamAbbr"]
            self._dftrack["isKicking"] = (
                self._dftrack["nflTeam"] == self._dftrack["possessionTeam"]
            ).astype(int)
            self._dftrack.loc[
                self._dftrack[self._dftrack["team"] == "football"].index, "isKicking"
            ] = -1
            self._dftrack = self._dftrack.drop(
                columns=["homeTeamAbbr", "visitorTeamAbbr"]
            )

            if flip_left:
                print("Standardizing moving right")
                left_plays = self._dftrack[self._dftrack.playDirection == "left"].index
                self._dftrack.loc[left_plays, "o"] = np.mod(
                    180 + self._dftrack.loc[left_plays, "o"], 2 * np.pi
                )
                self._dftrack.loc[left_plays, "dir"] = np.mod(
                    180 + self._dftrack.loc[left_plays, "dir"], 2 * np.pi
                )

                self._dftrack.loc[left_plays, "o_rad"] = np.mod(
                    np.pi + self._dftrack.loc[left_plays, "o_rad"], 2 * np.pi
                )
                self._dftrack.loc[left_plays, "dir_rad"] = np.mod(
                    np.pi + self._dftrack.loc[left_plays, "dir_rad"], 2 * np.pi
                )

                self._dftrack.loc[left_plays, "x"] = (
                    -self._dftrack.loc[left_plays, "x"] + 120
                )
                self._dftrack.loc[left_plays, "y"] = (
                    -self._dftrack.loc[left_plays, "y"] + 160 / 3
                )

            ## Downcast
            for col in ["x", "y", "s", "a", "dis", "o", "dir", "o_rad", "dir_rad"]:
                self._dftrack[col] = pd.to_numeric(self._dftrack[col], downcast="float")

            for col in [
                "nflId",
                "frameId",
                "gameId",
                "playId",
                "quarter",
                "down",
                "yardsToGo",
                "yardsToEndzone",
            ]:
                self._dftrack[col] = pd.to_numeric(
                    self._dftrack[col], downcast="integer"
                )

            self._dftrack["isGunner"] = 0
            self._dftrack["isVise"] = 0

            jersey_matches_gunner = (
                (self._dftrack["jerseyNumber"] == self._dftrack["gun_0"])
                | (self._dftrack["jerseyNumber"] == self._dftrack["gun_1"])
                | (self._dftrack["jerseyNumber"] == self._dftrack["gun_2"])
                | (self._dftrack["jerseyNumber"] == self._dftrack["gun_3"])
            )

            jersey_matches_vise = (
                (self._dftrack["jerseyNumber"] == self._dftrack["vise_0"])
                | (self._dftrack["jerseyNumber"] == self._dftrack["vise_1"])
                | (self._dftrack["jerseyNumber"] == self._dftrack["vise_2"])
                | (self._dftrack["jerseyNumber"] == self._dftrack["vise_3"])
            )

            #%%
            self._dftrack.loc[
                (self._dftrack["isKicking"] == 1) & jersey_matches_gunner, "isGunner"
            ] = 1

            self._dftrack.loc[
                (self._dftrack["isKicking"] == 0) & jersey_matches_vise, "isVise"
            ] = 1

            # self._dftrack = self._dftrack.drop(columns = [
            #     'gun_0', 'gun_1',
            #     'gun_2', 'gun_3', 'vise_0', 'vise_1', 'vise_2', 'vise_3', 'vise_4',
            # ])

        return self._dftrack

    def one_play(
        self, pIndex, cutoff=True, missing_verbose=False, add_to_play_row=True
    ):
        """The function to get one single play from data
        pIndex: numerical index from punt plays to analyze"""

        play_row = self.merged_plays().iloc[pIndex].copy()
        year = play_row.gameId // 1000000
        week = play_row.week
        assert year in (2018, 2019, 2020)

        ## Update tracking df if needed
        _track_df = self.track_df(yr_numb=year, wk_numb=week)
        one_play = _track_df[
            (_track_df.gameId == play_row.gameId)
            & (_track_df.playId == play_row.playId)
        ].copy()

        land_events = {
            "fair_catch",
            "punt_received",
            "punt_land",
            "out_of_bounds",
            "touchback",
            "punt_downed",
        }
        end_event_df = one_play[one_play.event.isin(land_events)]

        ## Check if a full punt
        if (
            "punt" not in one_play.event.unique()
            or "ball_snap" not in one_play.event.unique()
            or end_event_df.empty
        ):
            if missing_verbose:
                print(f"Not a proper punt play, events: {one_play.event.unique()}")
            raise NotAFullPuntError

        ## Find returner at snap
        snap_df = one_play[one_play.event == "ball_snap"]
        snap_frame = snap_df["frameId"].iloc[0]

        snap_returning = snap_df.query(f"isKicking == 0")

        snap_returner = snap_returning[
            snap_returning["x"] == snap_returning["x"].max()
        ].iloc[0]

        one_play["isReturner"] = (one_play["nflId"] == snap_returner["nflId"]).astype(
            int
        )
        ## Returner
        one_play["returnerJerseyRank"] = (
            one_play.loc[one_play.isReturner == 1]["jerseyNumber"].rank(method="dense")
            - 1
        )
        one_play["returnerJerseyRank"] = one_play["returnerJerseyRank"].fillna(-1)

        ## Gunner
        one_play["gunnerJerseyRank"] = (
            one_play.loc[one_play.isGunner == 1]["jerseyNumber"].rank(method="dense")
            - 1
        )
        one_play["gunnerJerseyRank"] = one_play["gunnerJerseyRank"].fillna(-1)

        ## Time calculations
        one_play["time_since_snap"] = 0.1 * (one_play["frameId"] - snap_frame)

        end_frame = end_event_df["frameId"].iloc[0]
        ## Calculate travel time using tracking markings
        one_play["time_until_end"] = 0.1 * (end_frame - one_play["frameId"])
        total_punt_frames = end_frame - snap_frame
        one_play["time_until_end"] = one_play["time_until_end"].clip(
            lower=0.1, upper=0.1 * total_punt_frames
        )

        ## Travel time using operation and hang
        land_time = one_play["operationTime"].iloc[0] + one_play["hangTime"].iloc[0]
        one_play["time_until_land"] = 0.1 * (
            snap_frame + land_time * 10 - one_play["frameId"]
        )
        one_play["time_until_land"] = one_play["time_until_land"].clip(
            lower=0.1, upper=land_time
        )

        if cutoff:
            max_frame = end_frame + 10
            one_play = one_play[
                (one_play.frameId >= snap_frame) & (one_play.frameId <= max_frame)
            ]

        one_play = one_play.drop(
            columns=[
                "snapTime",
                "operationTime",
                "hangTime",
                "gun_0",
                "gun_1",
                "gun_2",
                "gun_3",
                "vise_0",
                "vise_1",
                "vise_2",
                "vise_3",
                "vise_4",
            ]
        )

        #%% Changes to play row
        play_row["end_frame"] = end_frame

        if add_to_play_row:
            ## Add additional information to play row
            punter = snap_df[snap_df.position == "P"]
            if not punter.empty:
                play_row["punterId"] = punter.iloc[0]["nflId"]
            else:
                play_row["punterId"] = np.nan

            land_df = one_play.query(f"frameId == {end_frame}")
            land_ftb = land_df.query("team == 'football'").iloc[0]
            snap_ftb_y = snap_df[snap_df.team == "football"].iloc[0]["y"]

            for prefix in ["r", "l"]:
                try:
                    if prefix == "r":
                        _gunner = snap_df[
                            (snap_df.isGunner == 1) & (snap_df.y < snap_ftb_y)
                        ].iloc[0]
                        _vise = snap_df[
                            (snap_df.isVise == 1) & (snap_df.y < snap_ftb_y)
                        ]
                    if prefix == "l":
                        _gunner = snap_df[
                            (snap_df.isGunner == 1) & (snap_df.y > snap_ftb_y)
                        ].iloc[0]
                        _vise = snap_df[
                            (snap_df.isVise == 1) & (snap_df.y > snap_ftb_y)
                        ]

                    play_row[f"{prefix}GunnerId"] = _gunner["nflId"]
                    play_row[f"{prefix}GunnerJers"] = _gunner["jerseyNumber"]
                    play_row[f"{prefix}Gunner_x"] = _gunner["x"]
                    play_row[f"{prefix}Gunner_y"] = _gunner["y"]
                    play_row[f"{prefix}Gunner_vises"] = _vise.shape[0]
                    play_row[f"{prefix}Vises1"] = _vise["nflId"].values[0]
                    if _vise.shape[0] == 2:
                        play_row[f"{prefix}Vises2"] = _vise["nflId"].values[1]
                    else:
                        play_row[f"{prefix}Vises2"] = np.nan

                    land_gunner = land_df.query(f"nflId == {_gunner['nflId']}").iloc[0]
                    play_row[f"{prefix}Gunner_lx"] = land_gunner["x"]
                    play_row[f"{prefix}Gunner_ly"] = land_gunner["y"]

                except IndexError:
                    play_row[f"{prefix}GunnerJers"] = 404  ## no gunner found
                    continue

            play_row["jersRank0"] = (
                "l" if play_row["lGunnerJers"] < play_row["rGunnerJers"] else "r"
            )
            play_row["football_lx"] = land_ftb["x"]
            play_row["football_ly"] = land_ftb["y"]

        return {"track": one_play, "row": play_row}


def collected_xy_gunfoot(ind, nfl_data):
    play_collect = {}

    one_data = nfl_data.one_play(ind, missing_verbose=True)
    play_collect["gameId"] = one_data["row"]["gameId"]
    play_collect["playId"] = one_data["row"]["playId"]
    tracking = one_data["track"]
    snap_df = tracking[tracking.event == "ball_snap"]

    snap_gunners = snap_df[snap_df.isGunner == 1][
        ["x", "y", "jerseyNumber"]
    ].sort_values(by="jerseyNumber")

    if snap_gunners.empty:
        print(f"No gunners found at snap_df {ind}")
        raise NotAFullPuntError

    for i in range(snap_gunners.shape[0]):
        play_collect[f"gun{i}jersey"] = snap_gunners.iloc[i]["jerseyNumber"]
        play_collect[f"gun{i}x"] = snap_gunners.iloc[i]["x"]
        play_collect[f"gun{i}y"] = snap_gunners.iloc[i]["y"]

    end_frame = one_data["row"]["end_frame"]
    end_football = tracking.query(f"frameId == {end_frame} & team == 'football'")

    if end_football.empty:
        print(f"No football found at end, frame:{end_frame}, play:{ind}")
        raise NotAFullPuntError

    play_collect["end_x"] = end_football.iloc[0]["x"]
    play_collect["end_y"] = end_football.iloc[0]["y"]

    return play_collect


#%%

# path_to_data = "input/nfl-big-data-bowl-2022/"
# path_to_punts = "input/BDB2022-custom/punts_only/"
# data = NFLData(data_path=path_to_data, punts_path=path_to_punts)
# out = data.one_play(150)
# out["row"]

#%%

if __name__ == "__main__":
    path_to_data = "input/nfl-big-data-bowl-2022/"
    path_to_punts = "input/BDB2022-custom/punts_only/"
    data = NFLData(data_path=path_to_data, punts_path=path_to_punts)
    #%%
    # out = data.one_play(150)
    # out["row"]
    #%%

    # Collecting punt rows
    # rows_collected = []
    # for i in data.plays_df().query("year == 2020").index:
    #     if i % 500 == 0:
    #         print("ANOTHER 500 ROWS HAVE BEEN PROCESSED")
    #     try:
    #         rows_collected.append(
    #             data.one_play(i, missing_verbose=True, add_to_play_row=True)["row"]
    #         )
    #     except NotAFullPuntError:
    #         continue

    # df_collected = pd.concat(rows_collected, axis=1).T
    # #%%
    # # #%%
    # df_collected[
    #     [
    #         "gameId",
    #         "playId",
    #         "down",
    #         "yardsToGo",
    #         "yardsToEndzone",
    #         "quarter",
    #         "secondsUntilHalf",
    #         "specialTeamsResult",
    #         "kickingTeamScore",
    #         "receivingTeamScore",
    #         "kickDirectionIntended",
    #         "kickType",
    #         "snapTime",
    #         "operationTime",
    #         "hangTime",
    #         "week",
    #         "year",
    #         "end_frame",
    #         "punterId",
    #         "rGunnerId",
    #         "rGunner_y",
    #         "rGunner_vises",
    #         "rVises1",
    #         "rVises2",
    #         "rGunner_lx",
    #         "rGunner_ly",
    #         "lGunnerId",
    #         "lGunner_y",
    #         "lGunner_vises",
    #         "lVises1",
    #         "lVises2",
    #         "lGunner_lx",
    #         "lGunner_ly",
    #         "football_lx",
    #         "football_ly",
    #     ]
    # ].to_csv("input/BDB2022-custom/punts_from_2020.csv")

    #%%
    ## for each kick get: xy position of each gunner
    ## eventual location of kick xy

    # all_collected_xy_gunfoot = []

    # for i in data.plays_df().query('year == 2019').index.iloc[10]:
    #     if i%500 == 0: print("ANOTHER 500 ROWS HAVE BEEN PROCESSED")

    #     try:
    #         all_collected_xy_gunfoot.append(collected_xy_gunfoot(i, data))
    #     except NotAFullPuntError:
    #         continue

    #     except IndexError:
    #         print(i)

    # df_xy_gunfoot = pd.DataFrame(all_collected_xy_gunfoot)
    # df_xy_gunfoot.to_csv('input/BDB2022-custom/xy_gunner_football.csv', index=False)

    #%%save

    #%% Making an abbreviated version of tracking, punts only
    # dftrack = pd.read_csv("input/nfl-big-data-bowl-2022/tracking2020.csv")

    # track_punts = dftrack.merge(
    #     data.plays_df()[["gameId", "playId", "week"]],
    #     how="inner",
    #     on=["gameId", "playId",],
    # )
    # for wk_numb, dfg in track_punts.groupby("week"):
    #     print(f"Saving Week number {wk_numb} with total rows: {dfg.shape[0]}")
    #     dfg.drop(columns=["week"]).to_csv(
    #         "input/BDB2022-custom/punts_only/" + f"tracking2020_wk{wk_numb}.csv",
    #         index=False,
    #     )

# dfin = pd.read_csv('input/BDB2022-custom/punts_only/' + f'tracking2018_wk{wk_numb}.csv')


### check if any plays don't have these end events
# end_events = ('fair_catch','punt_received','punt_land','out_of_bounds','touchback','punt_downed')

# for ind, dfp in dftrack.groupby(['gameId','playId']):
#     if 'punt' not in dfp.event.unique():
#         print(f"No punt {ind}, events: {dfp.event.unique()}")
#     if any([a in dfp.event.unique() for a in end_events]):
#         continue
#     else:
#         print(ind, '\n', dfp.event.unique())


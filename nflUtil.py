

# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:14:53 2020

@author: wonky
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
import plotly.graph_objects as go

import seaborn as sns
import webbrowser
from IPython.display import HTML

import math
import typing
from sklearn.base import BaseEstimator
from scipy.special import expit

pd.set_option("display.max_columns", 101)

#%%

class NFLData:
    """
    A helper class to read NFL source files, join data and standardize directions
    
    .games_df() .plays_df() and .track_df()
    :param data_path: location of data
    :param yr_numb: year number
    
    """

    def __init__(self, data_path, yr_numb=None):
        self._dfin = None
        self._dfplays = None
        self._dfgames = None
        self._dfplayers = None
        self.yr_numb = yr_numb
        self.data_path = data_path

    def _read_csv(self, filename, index_col = None):
        assert filename.endswith(".csv")
        return pd.read_csv(os.path.join(self.data_path, filename), index_col=index_col)
                  
    def games_df(self):
        if self._dfgames is None:
            print('Reading in Games Data')
            self._dfgames = self._read_csv("games.csv", index_col='gameId')
        return self._dfgames
    
    def players_df(self):
        if self._dfplayers is None:
            print('Reading in Players Data')
            self._dfplayers = self._read_csv("players.csv")
            
            _ftin = self._dfplayers.height.str.extract(r'(?P<FT>\d)-(?P<IN>\d+)')
            _ftin = _ftin.dropna().astype(int)
            
            height = _ftin['FT']*12 + _ftin['IN']
            self._dfplayers.loc[height.index, 'height'] = height
            self._dfplayers['height'] = self._dfplayers['height'].astype(int)

        return self._dfplayers

    def plays_df(self, add_epa = False):
        if self._dfplays is None:
            print('Reading in Plays Data')
            
            df_games = self.games_df()
            self._dfplays = self._read_csv("plays.csv")

            _vtemp = self._dfplays['gameClock'].str.split(':', 
                                                          expand = True).astype(int)

            self._dfplays['gameClock'] = pd.to_timedelta(_vtemp[0], unit='m') \
                + pd.to_timedelta(_vtemp[1], unit='s')
            
            self._dfplays['secondsUntilHalf'] = self._dfplays['quarter']%2*900 + \
                self._dfplays['gameClock'].dt.seconds

            
            self._dfplays = self._dfplays.merge(df_games[['homeTeamAbbr', 
                                                          'visitorTeamAbbr',
                                                          'week']].reset_index(), 
                                                on = ['gameId'])
            
            self._dfplays.loc[self._dfplays.possessionTeam == self._dfplays.homeTeamAbbr,
                         'possTeamScore'] = self._dfplays.preSnapHomeScore
            self._dfplays.loc[self._dfplays.possessionTeam == self._dfplays.visitorTeamAbbr,
                         'possTeamScore'] = self._dfplays.preSnapVisitorScore
            self._dfplays.loc[self._dfplays.possessionTeam == self._dfplays.homeTeamAbbr,
                         'oppTeamScore'] = self._dfplays.preSnapVisitorScore
            self._dfplays.loc[self._dfplays.possessionTeam == self._dfplays.visitorTeamAbbr,
                         'oppTeamScore'] = self._dfplays.preSnapHomeScore
      
            self._dfplays.loc[self._dfplays.possessionTeam == self._dfplays.yardlineSide,
                         'yardsToEndzone'] = 100 - self._dfplays['yardlineNumber']
            self._dfplays.loc[self._dfplays.possessionTeam != self._dfplays.yardlineSide, 
                         'yardsToEndzone'] =  self._dfplays['yardlineNumber']
            
            self._dfplays = self._dfplays.sort_values(by = ['gameId','playId']).reset_index(drop = True)

            col_order = ['gameId', 'playId', 'down', 'yardsToGo','yardsToEndzone', 'quarter',
                        'specialTeamsPlayType', 'specialTeamsResult',
                        'playDescription', 'secondsUntilHalf','gameClock',
                        'possessionTeam', 
                        'kickerId', 'returnerId', 'kickBlockerId', 'yardlineSide',
                        'yardlineNumber', 'penaltyCodes', 'penaltyJerseyNumbers',
                        'penaltyYards', 'preSnapHomeScore', 'preSnapVisitorScore', 'passResult',
                        'kickLength', 'kickReturnYardage', 'playResult',
                        'absoluteYardlineNumber', 'homeTeamAbbr',
                        'visitorTeamAbbr', 'week', 'possTeamScore', 'oppTeamScore',
                        ]
            self._dfplays = self._dfplays[col_order]

        return self._dfplays
        
    def track_df(self, join_plays = True, flip_left = True, add_target = False, 
                 target_path = None):
        if self._dfin is None:
            assert(self.yr_numb != None)
            print('Reading in Tracking Data')
            
            df_games = self.games_df()
            play_downinfo = self.plays_df()[['gameId','playId','quarter','down','yardsToGo', 
                          'yardsToEndzone', 'gameClock', #'offenseFormation',
                          'possessionTeam',
                          # 'possTeamScore', 'oppTeamScore'
                          ]]
            
            self._dfin = self._read_csv('tracking{}.csv'.format(self.yr_numb))
            
            self._dfin['o_rad'] = np.mod(90 - self._dfin.o, 360)*math.pi/180.0
            self._dfin['dir_rad'] = np.mod(90 - self._dfin.dir, 360)*math.pi/180.0
            
            self._dfin['jerseyRank'] = self._dfin.groupby(['gameId','playId','team'])['jerseyNumber'].rank("dense") - 1
            self._dfin['jerseyRank'] = self._dfin['jerseyRank'].fillna(value = -1).astype(int)
            self._dfin['jerseyNumber'] = self._dfin['jerseyNumber'].fillna(value = -1).astype(int)
            self._dfin['nflId'] = self._dfin['nflId'].fillna(value = -1).astype(int)
            
            ##%% merge self._dfin with df_games
            self._dfin = self._dfin.merge(df_games[['homeTeamAbbr', 'visitorTeamAbbr']].reset_index(), on = ['gameId'])
            self._dfin.loc[self._dfin.team == 'home','nflTeam'] = self._dfin.loc[self._dfin.team == 'home']['homeTeamAbbr']
            self._dfin.loc[self._dfin.team == 'away','nflTeam'] = self._dfin.loc[self._dfin.team == 'away']['visitorTeamAbbr']
            self._dfin.loc[self._dfin.team == 'home','nflOppTeam'] = self._dfin.loc[self._dfin.team == 'home']['visitorTeamAbbr']
            self._dfin.loc[self._dfin.team == 'away','nflOppTeam'] = self._dfin.loc[self._dfin.team == 'away']['homeTeamAbbr']
            self._dfin = self._dfin.drop(columns = ['homeTeamAbbr', 'visitorTeamAbbr'])
            
            if join_plays:
                print('Merging with Plays Data')
                self._dfin = self._dfin.merge(play_downinfo, on = ['gameId','playId'])
                self._dfin['isOffence'] = (self._dfin['nflTeam'] == self._dfin['possessionTeam'])
                self._dfin['isOffence'] = self._dfin['isOffence'].astype(int)
                self._dfin.loc[self._dfin[self._dfin['team'] == 'football'].index, 'isOffence'] = -1

                
            if flip_left:
                print('Standardizing moving right')
                left_plays = self._dfin[self._dfin.playDirection == 'left'].index
                self._dfin.loc[left_plays, 'o'] = np.mod(180+ self._dfin.loc[left_plays,'o'], 2*np.pi)
                self._dfin.loc[left_plays, 'dir'] = np.mod(180 + self._dfin.loc[left_plays,'dir'], 2*np.pi)
                
                self._dfin.loc[left_plays,'o_rad'] = np.mod(np.pi + self._dfin.loc[left_plays,'o_rad'], 2*np.pi)
                self._dfin.loc[left_plays,'dir_rad'] = np.mod(np.pi + self._dfin.loc[left_plays,'dir_rad'], 2*np.pi)
                
                self._dfin.loc[left_plays,'x'] = -self._dfin.loc[left_plays,'x'] +120
                self._dfin.loc[left_plays,'y'] = -self._dfin.loc[left_plays,'y'] +160/3
                
            for col in ['x','y','s','a','dis','o','dir','o_rad','dir_rad']:
                self._dfin[col] = pd.to_numeric(self._dfin[col], downcast = 'float')
                
            for col in ['nflId','frameId','gameId','playId','quarter','down','yardsToGo', 'yardsToEndzone']:
                self._dfin[col] = pd.to_numeric(self._dfin[col], downcast = 'integer')
                
            if add_target: 
                if target_path is None: target_path = self.data_path + 'targetedReceiver.csv'
                df_targetReceiver = pd.read_csv(target_path)
                self._dfin = self._dfin.reset_index().merge(
                    df_targetReceiver, on = ['gameId','playId']).set_index('index')

        return self._dfin
  

    
def open_html_plot(animation, fname = 'temp_ani.html'):
    with open(fname,'wb') as f:
        f.write((HTML(animation.to_jshtml())).data.encode("UTF-8"))
    webbrowser.open(fname, new=2)


class AnimatePlay:
    def __init__(self, play_df,  play_text = None, plot_size_len = 20) -> None:
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
        self._mean_interval_ms = np.mean([delta.microseconds/1000 for delta in np.diff(sorted(pd.to_datetime(play_df.time).unique())) ])
        
        self._fig = plt.figure(figsize = (plot_size_len, plot_size_len*(self._MAX_FIELD_Y/self._MAX_FIELD_X)))

        self._ax_field = plt.gca()
        
        self._ax_home = self._ax_field.twinx()
        self._ax_away = self._ax_field.twinx()
        self._ax_jersey = self._ax_field.twinx()

        self.ani = animation.FuncAnimation(self._fig, self.update, frames=len(self._times), interval = self._mean_interval_ms, 
                                          init_func=self.setup_plot, blit=False)
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
        return r * np.exp( 1j * theta)
    
    def data_stream(self):
        for atime in self._times:
            yield self._play_df[self._play_df.time == atime]
    
    def setup_plot(self): 
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        
        ball_snap_ser = self._play_df[(self._play_df.event == 'ball_snap') 
                                        & (self._play_df.team == 'football')].iloc[0]
        self._ax_field.axvline(ball_snap_ser.x, color = 'k', linestyle = '--')
        
        if ball_snap_ser.yardsToEndzone <= ball_snap_ser.yardsToGo:
            self._ax_field.axvline(110,
                       color = 'yellow', lw = 4, linestyle = '-')
        else:
            self._ax_field.axvline(ball_snap_ser.x + ball_snap_ser.yardsToGo,
                                   color = 'yellow', lw = 4, linestyle = '-')
        
        self._ax_field.add_patch(Rectangle((0, 0), 10, 53.3,linewidth=0.1,edgecolor='r',
                                facecolor='slategray',alpha=0.2,zorder=0))
        
        self._ax_field.add_patch(Rectangle((110, 0), 120, 53.3,linewidth=0.1,edgecolor='r',
                                facecolor='slategray',alpha=0.2,zorder=0))
        
        self._ax_field.text(100,55.3, "G {game} | P {play}".format(
            game = ball_snap_ser.gameId, 
            play = ball_snap_ser.playId))

        downst = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
        self._ax_field.text(0,55.3,"Q: {quarter} | Time: {minutes}:{seconds}".format(
            quarter = ball_snap_ser.quarter, 
            minutes = ball_snap_ser.gameClock.seconds // 60,
            seconds = (ball_snap_ser.gameClock.seconds%60)
            ))
            
        if ball_snap_ser.yardsToEndzone <= ball_snap_ser.yardsToGo:
            dista = 'Goal'
        else: 
            dista = ball_snap_ser.yardsToGo 
        
        self._ax_field.text(0,53.8,"{down} and {dist}".format(
            down = downst[ball_snap_ser.down], 
            dist = dista))
        
        for x in range(20, 110, 10): 
                numb = x
                if x > 50: numb = 120 - x
                self._ax_field.text(x, 12, str(numb - 10),horizontalalignment='center',
                         fontsize=20,fontname='Times New Roman',color='slategray')
                self._ax_field.text(x - 0.3, 53.3 - 12, str(numb - 10),
                         horizontalalignment='center',
                         fontsize=20, fontname='Times New Roman',color='slategray', rotation=180)
                
        for x in range(11, 110):
            self._ax_field.plot([x, x], [0.4, 0.7], color='lightgray')
            self._ax_field.plot([x, x], [53.0, 52.5], color='lightgray')
            self._ax_field.plot([x, x], [22.91, 23.57], color='lightgray')
            self._ax_field.plot([x, x], [29.73, 30.39], color='lightgray')
        
        if self._play_text is not None:
            self._ax_field.text(10,53.8, self._play_text)

        self._ax_field.text(54.8,56.3,'Frame: ')    
        self._frameId_text = self._ax_field.text(59, 56.3, '')

        self.set_axis_plots(self._ax_home, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_away, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        
        for idx in range(10,120,10):
            self._ax_field.axvline(idx, color = 'k', linestyle = '-', alpha = 0.05)
        ## used to be s = 100 and s = 500
        self._scat_field = self._ax_field.scatter([], [], s = 100, color = 'black')
        self._scat_home = self._ax_home.scatter([], [], s = 500, color = self._CPLT[0], edgecolors = 'k')
        self._scat_away = self._ax_away.scatter([], [], s = 500, color = self._CPLT[1], edgecolors = 'k')
        
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self._ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            self._scat_name_list.append(self._ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color = 'k')))
            
        return (self._scat_field, self._scat_home, self._scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
        
    def update(self, anim_frame):
        pos_df = next(self._stream)
        self._frameId_text.set_text(pos_df.frameId.iloc[0])

        for label in pos_df.team.unique():
            label_data = pos_df[pos_df.team == label]

            if label == 'football':
                self._scat_field.set_offsets(np.hstack([label_data.x, label_data.y]))
            elif label == 'home':
                self._scat_home.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'away':
                self._scat_away.set_offsets(np.vstack([label_data.x, label_data.y]).T)

        for (index, row) in pos_df[pos_df.position.notnull()].reset_index().iterrows():
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            self._scat_name_list[index].set_position((row.x, row.y-1.9))
            self._scat_name_list[index].set_text(row.displayName.split()[-1])
            
            player_orientation_rad = row.o_rad
            player_direction_rad = row.dir_rad
            player_speed = row.s
            
            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)), 
                                   np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)), 
                                      np.imag(self.polar_to_z(2, player_orientation_rad))])
            
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self._ax_field.add_patch(Arrow(row.x, 
                            row.y, player_vel[0], player_vel[1], color = 'k'))
            self._a_or_list[index].remove()
            self._a_or_list[index] = self._ax_field.add_patch(Arrow(row.x, 
                            row.y, player_orient[0], player_orient[1], 
                            color = 'grey', width = 2))
                
        
        return (self._scat_field, self._scat_home, self._scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)


class AnimatePlayPitchControl(AnimatePlay):
    '''Plots pitch control animation given: single play tracking'''
    
    def __init__(self, play_df, Z_diff_indiv, play_text = None, names = None, conn_colors = None,
                 NGrid = 121, plot_size_len = 30, show_control = True, opt_text = None) -> None:
        super().__init__(play_df, plot_size_len)
        """Initializes the datasets used to animate the play.

        Parameters
        ----------
        play_df : DataFrame
            Dataframe corresponding to the play information for the play that requires
            animation. This data will come from the weeks dataframe and contains position
            and velocity information for each of the players and the football.
        Z_diff_indiv: Tuple of 2 numpy arrays: 
            Z_diff : Numpy Array  (frameId x N x N aray)
            Z_indiv: Numpy Array (frameId x isOffense x jerseyRank x N x N ) array

        Returns
        -------
        None
        """
        # assert Z_array.shape[2] == play_df.frameId.max()+1
        self._NGrid = NGrid
        self._MAX_PLAYER_SPEED = 11.3
        self._NUM_RECEIVERS = play_df[~play_df.route.isnull()].jerseyNumber.nunique()
        self._conn_colors = conn_colors
        self._show_control = show_control
        self._play_text = play_text
        self._opt_text = opt_text
        self._names = names
        self._Z_array = Z_diff_indiv[0]
        self._Z_indiv = Z_diff_indiv[1]
        self._X, self._Y, self._pos = self.generate_data_grid()
        self._ax_football = self._ax_field.twinx()
        plt.close()
    

    def generate_data_grid(self):
        assert(self._NGrid == self._Z_indiv.shape[-1])
        X = np.linspace(0, self._MAX_FIELD_X, self._NGrid)
        Y = np.linspace(0, self._MAX_FIELD_Y, self._NGrid)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.zeros(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        
        return X, Y, pos
 
    def setup_plot(self): 
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        assert (self._play_df[(self._play_df.event == 'ball_snap') 
                    & (self._play_df.team == 'football')].empty == False)
        
        ball_snap_ser = self._play_df[(self._play_df.event == 'ball_snap') 
                                & (self._play_df.team == 'football')].iloc[0]
        self._ax_field.axvline(ball_snap_ser.x, color = 'k', linestyle = '--')
        if ball_snap_ser.yardsToEndzone <= ball_snap_ser.yardsToGo:
            self._ax_field.axvline(110,
                       color = 'yellow', lw = 4, linestyle = '-')
        else:
            self._ax_field.axvline(ball_snap_ser.x + ball_snap_ser.yardsToGo,
                                   color = 'yellow', lw = 4, linestyle = '-')
        
        self._ax_field.add_patch(Rectangle((0, 0), 10, 53.3,linewidth=0.1,\
                                           edgecolor='r',
                                facecolor='slategray',alpha=0.2,zorder=0))
        
        self._ax_field.add_patch(Rectangle((110, 0), 120, 53.3,linewidth=0.1,\
                                           edgecolor='r',
                                facecolor='slategray',alpha=0.2,zorder=0))
        
        self._ax_field.text(100,55.3, "G {game} | P {play}".format(
            game = ball_snap_ser.gameId, 
            play = ball_snap_ser.playId))
        
        downst = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
        self._ax_field.text(0, 55.3,
            "Q: {quarter} | Time: {minutes}:{seconds}".format(
            quarter = ball_snap_ser.quarter, 
            minutes = ball_snap_ser.gameClock.seconds // 60,
            seconds = (ball_snap_ser.gameClock.seconds%60)
            ))
        
        if ball_snap_ser.yardsToEndzone <= ball_snap_ser.yardsToGo:
            dista = 'Goal'
        else: 
            dista = ball_snap_ser.yardsToGo 
            
        self._ax_field.text(0,53.8,"{down} and {dist}".format(
            down = downst[ball_snap_ser.down], 
            dist = dista ))
        
        for x in range(20, 110, 10): 
                numb = x
                if x > 50: numb = 120 - x
                self._ax_field.text(x, 12, str(numb - 10),\
                                    horizontalalignment='center',
                         fontsize=20,fontname='Times New Roman', \
                         color='lightgray')
                self._ax_field.text(x - 0.3, 53.3 - 12, str(numb - 10),
                         horizontalalignment='center',
                         fontsize=20, fontname='Times New Roman', \
                         color='lightgray', rotation=180)
                
        for x in range(11, 110):
            self._ax_field.plot([x, x], [0.4, 0.7], color='lightgray')
            self._ax_field.plot([x, x], [53.0, 52.5], color='lightgray')
            self._ax_field.plot([x, x], [22.91, 23.57], color='lightgray')
            self._ax_field.plot([x, x], [29.73, 30.39], color='lightgray')
            
        if self._play_text is not None:
            self._ax_field.text(10,53.8, self._play_text)
        
        if self._opt_text is not None:
            self._ax_field.text(0,-1.5, self._opt_text, va = 'top')
       
        if self._names is not None:
            fontsize = 7 if self._show_control else 8
            self._ax_field.text(0.8,1, self._names.to_string(), \
                                size = fontsize, fontdict = {'family' : 'monospace'})
        
        self._ax_field.text(54.8,56.3,'Frame: ')    
        self._frameId_text = self._ax_field.text(59, 56.3, '')

        self.set_axis_plots(self._ax_home, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_away, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_football, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        
        for idx in range(10,120,10):
            self._ax_field.axvline(idx, color = 'k', linestyle = '-', alpha = 0.05)
            
        self._scat_football = self._ax_football.scatter(\
                [], [], s = 50, color = 'black')
        self._scat_home = self._ax_home.scatter([], [], \
                s = 200, color = self._CPLT[0], edgecolors = 'k')
        self._scat_away = self._ax_away.scatter([], [], \
                s = 200, color = self._CPLT[1], edgecolors = 'k')
        
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        # self._a_inf_list = []
        self._inf_contours_list = []
        self._conn_colors_list = []
   
        
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_jersey.text(0, 0, '', \
                horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self._ax_jersey.text(0, 0, '', \
                horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            self._scat_name_list.append(self._ax_jersey.text(0, 0, '', \
                horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color = 'k')))
            # self._a_inf_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color = 'k')))
            
            if self._show_control:
                self._inf_contours_list.append(self._ax_field.contour([0, 0], [0, 0], [[0,1],[0,1]]))
            else:
                self._inf_contours_list.append(self._ax_field.contour([0, 0], [0, 0], [[0,1],[0,1]]))
        
             
        if self._show_control:
            self._contour_kw = dict(levels=np.linspace(0, 1, 41), cmap='PiYG', 
                                    vmin=0, vmax=1, origin='lower', alpha = 0.5)
            self._pitch_control_contour = self._ax_field.contourf([0, 0], [0, 0], 
                                            [[0,0],[0,0]], **self._contour_kw)
            self._fig.colorbar(self._pitch_control_contour, ax=self._ax_field, pad = 0.015)
        
        if self._conn_colors is not None:
            self._conn_cmap = get_cmap('Spectral')
            self._marker_style = dict(linestyle='', marker='o', 
                                      fillstyle = 'none')
            color_max = self._conn_colors.connIndex.max()
            
            for c in self._conn_colors.connIndex:
                self._conn_colors_list.append(self._ax_field.plot([],[],
                                            color = self._conn_cmap(c / color_max), 
                                            markeredgewidth = 2 - 0.1*c,
                                            markersize = int(18*(1+0.1*c)), **self._marker_style))
       
        return (self._scat_football, self._scat_home, self._scat_away, 
                *self._scat_jersey_list, *self._scat_number_list, 
                *self._scat_name_list,*self._conn_colors_list)
        
    def update(self, anim_frame):
        pos_df = next(self._stream)
        pos_df = pos_df.sort_values(by = 'nflId')
        current_frame = pos_df.frameId.iloc[0]
        ## update current text
        self._frameId_text.set_text(current_frame)
        
        ## home is OFFENCE
        for label in pos_df.isOffence.unique():
            label_data = pos_df[pos_df['isOffence']== label]
            if label == -1:
                self._scat_football.set_offsets(np.hstack([label_data.x, label_data.y]))
            elif label == 1:
                self._scat_home.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 0:
                self._scat_away.set_offsets(np.vstack([label_data.x, label_data.y]).T)
        
        if self._conn_colors is not None:
            current_conn = self._conn_colors[self._conn_colors.frameId == current_frame]
            
            if current_conn.empty:
                for conn_plot in self._conn_colors_list:
                    conn_plot[0].set_data([],[])
                
            for _, conn in current_conn.iterrows():
                if conn.est_epa <= 0:
                    c_index = int(conn.connIndex)
                    self._conn_colors_list[c_index][0].set_data([],[])
                    continue
                pair = pos_df[pos_df.nflId.isin([conn.nflId, conn.def0nflId])]
                assert (pair.shape[0] == 2)
                c_index = int(conn.connIndex)
                self._conn_colors_list[c_index][0].set_data(np.vstack([pair.x, pair.y]))
        
        for (index, row) in pos_df[pos_df.jerseyNumber.notnull()].reset_index().iterrows():
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(int(row.jerseyNumber))

            vel_comp = self.polar_to_z(row.s, row.dir_rad)
            orient_comp = self.polar_to_z(2, row.o_rad)
   
            self._a_or_list[index].remove()
            self._a_or_list[index] = self._ax_field.add_patch(Arrow(row.x, \
                row.y, np.real(orient_comp), np.imag(orient_comp), color = 'grey', width = 5))
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self._ax_field.add_patch(Arrow(row.x, \
                row.y, np.real(vel_comp), np.imag(vel_comp), color = 'k'))

            if current_frame >= self._Z_array.shape[0] \
                or np.all(self._Z_array[current_frame,:,:] == 0.5):
                continue ## for empty frames - Z_diff not calculated
            
            if row.jerseyRank != -1: # football
                for cont_info in self._inf_contours_list[index].collections:
                    try:
                      cont_info.remove()
                    except ValueError:
                        print(row.jerseyRank, row.displayName, end = ' ')

            if row.isOffence == 1 and row.jerseyRank < self._Z_indiv.shape[2]:
                if self._show_control \
                and not np.all(self._Z_indiv[current_frame,1,row.jerseyRank,:,:] == 0):
                    self._inf_contours_list[index] = \
                        self._ax_field.contour(self._X, self._Y, 
                                                self._Z_indiv[current_frame,1,row.jerseyRank,:,:], 
                                                linestyles = 'dotted', levels = 1, 
                                                linewidths = 0.8, alpha = 0.7)
                else:
                    self._inf_contours_list[index] = \
                        self._ax_field.contour(self._X, self._Y, 
                                                self._Z_indiv[current_frame,1,row.jerseyRank,:,:], 
                                                cmap='Reds', levels = 1, alpha = 0.8,
                                                linestyles = 'dotted', linewidths = 2)
                    
            elif row.isOffence == 0 and row.jerseyRank < self._Z_indiv.shape[2]:
                if self._show_control:
                    self._inf_contours_list[index] = \
                        self._ax_field.contour([0, 0], [0, 0], [[0,1],[0,1]])
                else:
                    self._inf_contours_list[index] = \
                        self._ax_field.contour(self._X, self._Y, 
                                                self._Z_indiv[current_frame,0,row.jerseyRank,:,:], 
                                                cmap='Greens', levels = 7, 
                                                alpha = 0.5, vmin=0.2, vmax=0.9)
        
        if self._show_control and current_frame < self._Z_array.shape[0]:
            for cont_info in self._pitch_control_contour.collections:
                    cont_info.remove()
            self._pitch_control_contour = self._ax_field.contourf(self._X, self._Y, self._Z_array[current_frame,:,:], **self._contour_kw)
        
        return (self._scat_football, self._scat_home, self._scat_away,
                *self._scat_jersey_list, *self._scat_number_list, 
                *self._scat_name_list,  *self._conn_colors_list)    




class CatchProba:
    """
    A helper class to find CatchProba heatmap
    and display using Plotly
    
    """
    def __init__(self, model = None, Nx = 100, Ny = 70):
        self.cov_perc_list = [0.00, 0.25, 0.5, 0.95] 
        self.cov_perc_str = [str(r) for r in  self.cov_perc_list]
        self._model = model
        self._Zall = None
        self._X = None
        self._Y = None
        self._Nx = Nx
        self._Ny = Ny
        

    def find_Z(self):
          """creating heatmaps for catch probability"""
          x = np.linspace(10, 120, self._Nx)
          y = np.linspace(0, 160/3, self._Ny)
          self._X, self._Y = np.meshgrid(x, y)
          
          x_ball = 30
          y_ball = 23.5
          
          if self._Zall is None:
              def f(Xarr, Yarr, cov_perc_list):
                  Z = np.zeros([len(cov_perc_list), self._Ny, self._Nx],)
                  x_ball = 30
                  y_ball = 23.5
                  for x in range(self._Nx):
                      for y in range(self._Ny):
                          features = []
                          for cov_perc in cov_perc_list:
                              features.append([cov_perc, 1, 2, 
                                        Xarr[y,x] - x_ball, np.abs(Yarr[y,x] - y_ball),
                                        Xarr[y,x] - x_ball, np.abs(Yarr[y,x] - y_ball),
                                        ])
                          Z[:,y,x] = self._model.predict_proba(features)[:,1]
                  return Z
              
              self._Zall = f(self._X, self._Y, self.cov_perc_list)
          
          return None

    def make_plotly(self):
        hash_x = [i for i in range(11, 109) for _ in range(8)]
        hash_y = [i for _ in range(11, 109) for i in [0.4,0.7,53.0,52.5,22.91,23.57,29.73,30.39]]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                  80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                y=[0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                  53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                mode="lines",
                line=go.scatter.Line(color="gray"),
                showlegend=False,
                hoverinfo='skip')
        )

        fig.add_trace(
            go.Scatter(
                x=[40, 40],
                y=[0, 53.3],
                mode="lines",
                line=go.scatter.Line(color="yellow", width = 4),
                showlegend=False,
                hoverinfo='skip')
        )

        fig.add_trace(
            go.Scatter(
                x=hash_x,
                y=hash_y,
                mode="markers",
                marker_line_width=1, marker_size=5,
                marker_symbol = "line-ns",
                showlegend=False,
                hoverinfo='skip')
        )


        fig.add_trace(
            go.Scatter(
                x=[30],
                y=[23.5],
                mode="markers",
                marker_size = 12,
                showlegend=False,
                hoverinfo='skip')
        )


        fontdict = dict(
                    family="Courier New, monospace",
                    size=12,
                    color="#ffffff"
                    )
        fig.add_annotation(
                x=30,y=23.5,xref="x",yref="y",text="QB",showarrow=True,
                font=fontdict,align="center",arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor="#636363",
                ax=0,ay=-20,bordercolor="#c7c7c7",borderwidth=2,borderpad=4,bgcolor="#ff7f0e",opacity=0.8
                )

        fig.add_annotation(
                x=40,y=52,xref="x",yref="y",text="10 YARDS",showarrow=True,
                font=fontdict,align="center",arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor="#636363",
                ax=0,ay=-20,bordercolor="#c7c7c7",borderwidth=1,borderpad=1,bgcolor="#ff7f0e",opacity=0.8
                )

        fig.add_annotation(
                x=80,y=52,xref="x",yref="y",text="50 YARDS",showarrow=True,
                font=fontdict,align="center",arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor="#636363",
                ax=0,ay=-20,bordercolor="#c7c7c7",borderwidth=1,borderpad=1,bgcolor="#ff7f0e",opacity=0.8
                )


        fig.update_layout(
            autosize=False,
            width=1100,
            height=450,
            margin=dict(
                l=50,
                r=50,
                b=10,
                t=60,
                pad=4
            ),
            yaxis_range = [0,53.3],
            xaxis_range = [0,120],
            paper_bgcolor="White",
            title = 'Catch Probability given Contant Defender Influence - Coverage = 0.0',
            yaxis_nticks=1
        )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)


        for i in range(self._Zall.shape[0]):
          fig.add_trace(go.Contour(
                  z=self._Zall[i],
                  x=self._X[0],
                  y=self._Y[:,0],
                  connectgaps=True, 
                  colorscale='Hot',
                  contours=dict(
                    start=0,
                    end=1,
                    size=0.1,
                    ),
                  visible = (i == 0),
                  hoverinfo  = "z",
                  opacity = 0.75))


        title_string = 'Catch Probability when '
        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                active=0,
                buttons=list(
                    [
                    dict(label = 'Coverage = ' +  self.cov_perc_str[0],
                          method = 'update',
                          args = [{'visible': [True, True, True, True, True, False, False, False,]}, # the index of True aligns with the indices of plot traces
                                  {'title': title_string+'completely open - Coverage = ' +  self.cov_perc_str[0],
                                  }]),
                    dict(label = 'Coverage = ' +  self.cov_perc_str[1],
                          method = 'update',
                          args = [{'visible': [True, True, True, True,False, True, False, False]},
                                  {'title': title_string+'mostly open - Coverage = ' +  self.cov_perc_str[1],
                                  }]),
                    dict(label = 'Coverage = ' +  self.cov_perc_str[2],
                          method = 'update',
                          args = [{'visible': [True, True, True, True,False, False, True, False]},
                                  {'title': title_string+'well covered - Coverage = '+ self.cov_perc_str[2],
                                  }]),
                    dict(label = 'Coverage = ' +  self.cov_perc_str[3],
                          method = 'update',
                          args = [{'visible': [True, True, True, True,False, False, False, True]},
                                  {'title': title_string+'very well covered - Coverage = ' +  self.cov_perc_str[3],
                                  }]),
                    ])
                ),])
        
        fig.add_annotation(x=-0.225, y=0.85,xref = 'paper',yref = 'paper',
            text="Use dropdown to<br>change coverage. <br><br><br><br><br><br><br><br> Ligher colours show<br>higher catch probability, <br> legend located far right",
            showarrow = False)

        return fig





class PlayData():
    """A helper class to handle data from a single play and make appropriate
    coverage calculations
    This class initializes with the basic play data and makes all calculations
    neccessary to find redEPA"""
    
    actions = ['pass_forward','pass_shovel','qb_sack','qb_strip_sack']
    estYardsFeatures = ['x', 'y', 'vel_x', 'vel_y', 'mu_x', 'mu_y', 'def0x_diff', 'def0y_diff',
                       'def1x_diff', 'def1y_diff', 'snap_x', 'qb_x','qb_o_rad',
                       'projxDiff','projyDiff', 'avgCoverage', 'frameId']
    epaFeatures = ['down', 'yardsToGo', 'yardsToEndzone','secondsUntilHalf']
    compProbaFeatures = ['avgCoverage','vel_x','vel_y_abs', 
                     'xDiff','yDiff_abs','projxDiff', 'projyDiff']
    qb_playact_orad = (2*math.pi/3, 4*math.pi/3)

    def __init__(self, df_plays: pd.DataFrame, df_track: pd.DataFrame, 
                 pIndex: int, len_seq: int = 200, epModel: BaseEstimator = None,
                 compProbaModel: BaseEstimator = None, 
                 estYardsModel: BaseEstimator = None,
                 cov_query = None):
        """
        Parameters
        ----------
        df_plays : pd.DataFrame
            all plays data preprocessed by NFLData
        df_track : pd.DataFrame
            all tracking data for given week, preprocessed by NFLData
        pIndex : int
            index of play from df_track
        len_seq : int, optional
            number of frames to analyze before catch. The default is 5.
        epmodel : BaseEstimator, optional
            loaded estimator for estimating EP. The default is None.

        """
        assert (pIndex in df_plays.index)
        
        # self.pIndex = pIndex
        # self._df_plays = df_plays
        # self._df_track = df_track
        self.pIndex = pIndex
        self.a_play = df_plays.loc[pIndex].copy()
        self.play_text = self.a_play.playDescription
        self.play_df = df_track[(df_track.gameId == self.a_play.gameId) 
           & (df_track.playId == self.a_play.playId)]
        self.snap_df = self.play_df[self.play_df.event == 'ball_snap']  
        
        if self.play_df.empty or self.snap_df.empty:
            raise AssertionError('Tracking or snap data not found, check week')
            
        self._2teams = self.snap_df.nflTeam.dropna().unique()
        self.poss_team = self.a_play['possessionTeam']
        self.def_team = self._2teams[self._2teams != self.poss_team][0]
        self._snap_frame = self.snap_df.frameId.unique()[0]
        self.len_seq = len_seq
        self._nothrow_frames = 9
        ## extent of what we want upon init
        self.out_pocket_frame = None
        self.act_frame = None
        self.seq_range = None
        self.seq_df = None
        self._Z_diff_indiv = None
        self._proj_vel_df = None
        ## extent of how far we can get without model
        self._epModel = epModel
        self._compProbaModel = compProbaModel
        self._estYardsModel = estYardsModel
        self._cov_query = cov_query
        self._epa_play = None
        self._cov_seq = None
        self._cov_grp = None
   
    
    def _sequence_settings(self):
        if self.act_frame is None:
            try: 
                self.act_frame = self.play_df[self.play_df.event.isin(
                    self.actions)].frameId.iloc[0]
            except IndexError:
                raise AssertionError('Action not found in current play, check description')

            self.len_seq = min(self.len_seq, self.act_frame - self._snap_frame)
            self.seq_range = np.arange(self.act_frame - self.len_seq + 1, 
                                       self.act_frame + 1)
            self.seq_df = self.play_df[self.play_df.frameId.isin(
                        self.seq_range)].copy()
            
            qb_snap = self.snap_df[self.snap_df.position == 'QB'][['nflId','y']].copy()
            if qb_snap.shape[0] > 1: ## 2 QBs, thanks taysom
                ftb_snap_y = self.snap_df[self.snap_df.displayName == 'Football'].iloc[0]['y']
                qb_snap['diffy'] = np.abs(qb_snap['y'] - ftb_snap_y)
                qb_snap = qb_snap.sort_values(by = ['diffy'])
                fakeqbId = qb_snap.iloc[1:].nflId.values
                self.seq_df.loc[self.seq_df.nflId.isin(fakeqbId), 'position'] = 'WR'

            if (isinstance(self.a_play['typeDropback'], str)) and \
                ('SCRAMBLE' in self.a_play['typeDropback']): ## end sequence at scram
                qb_df = self.seq_df[self.seq_df.position == 'QB'].copy()
                ftb_snap_y = self.snap_df[self.snap_df.displayName == 'Football'].iloc[0]['y']
                qb_df['diffY'] = np.abs(qb_df['y'] - ftb_snap_y)
                self.out_pocket_frame = qb_df[qb_df.diffY > 4]['frameId'].min()
                if not np.isnan(self.out_pocket_frame):
                    self.seq_df = self.seq_df[self.seq_df.frameId < self.out_pocket_frame]
            
            if self.seq_df[self.seq_df.isOffence == 1].empty or \
                self.seq_df[self.seq_df.isOffence == 0].empty or \
                self.seq_df[self.seq_df.isOffence == -1].empty or \
                self.seq_df[self.seq_df.position == 'QB'].empty:
                raise AssertionError('No Offensive/Defensive Players/QB/Football in data,\
                                 check .seq_df')
         
        return None

    @property
    def coverage_Z(self):
        if self._Z_diff_indiv is None:
            self._sequence_settings() 
            
            calculations = self.calculateZfromtrack(self.seq_df) ##static method
            self._Z_diff_indiv = calculations['Z_tuple']
            self._proj_vel_df = calculations['proj_vel_df']
            self.seq_df = self.seq_df.merge(self._proj_vel_df, on = ['frameId','nflId'])
            
        return self._Z_diff_indiv

    @property
    def coverage_seq(self):
        '''Returns individual coverage for each row in sequence (offence only)
        Calculates Z using coverage_Z if needed, 
        Calls getCoveragefromZ (using df.apply) after retrieving Z and epa
        '''
        if self._cov_seq is None:
            #filter offense
            # print('Calculating Coverage')
            
            z_diff_indiv = self.coverage_Z
            seq_df_filter = self.seq_df[(self.seq_df.isOffence == 1)
                                 & (self.seq_df.position != 'QB')]
            
            if self._cov_query is not None:
                seq_df_filter = seq_df_filter.query(self._cov_query)
                if seq_df_filter.empty:
                    raise AssertionError('Query left dataframe empty, likely\
                                     likely no targetNflId')
                
            self._cov_seq = seq_df_filter[['gameId','playId','frameId',
                    'x','y', 's','o_rad','dir_rad',
                    'position','jerseyNumber','nflId','displayName','route',
                    'vel_x', 'vel_y', 'dvel_x', 'dvel_y']].copy()
            
            if hasattr(seq_df_filter, 'targetNflId'):
                self._cov_seq['targetNflId'] = seq_df_filter['targetNflId']
                
            defenders_cov = seq_df_filter.apply(lambda row:
                      self.getCoveragefromZ(row[['nflId','frameId','jerseyRank','mu_x','mu_y']],
                                            self.a_play,
                                            z_diff_indiv,
                                            self.seq_df)
                                            , axis = 1)
           
            self._cov_seq = self._cov_seq.merge(defenders_cov, on = ['nflId','frameId'])
                
            self._cov_seq['connIndex'] = self._cov_seq['jerseyNumber'\
                        ].rank(method = 'dense', ascending = False) - 1

            qb_throw = self.seq_df[(self.seq_df.position == 'QB')][['frameId','x','y', 'o_rad']]
            qb_throw.columns = ['qb_'+ col for col in qb_throw.columns]
            self._cov_seq = self._cov_seq.merge(qb_throw, left_on = 'frameId', right_on = 'qb_frameId')
            
            self._cov_seq['vel_y_abs'] = np.abs(self._cov_seq['vel_y'])
            self._cov_seq['xDiff'] = self._cov_seq.x - self._cov_seq.qb_x
            self._cov_seq['yDiff_abs'] = np.abs(self._cov_seq.y - self._cov_seq.qb_y)
            self._cov_seq['projxDiff'] = self._cov_seq.mu_x - self._cov_seq.qb_x
            self._cov_seq['projyDiff'] = np.abs(self._cov_seq.mu_y - self._cov_seq.qb_y)
            self._cov_seq['snap_x'] = self.snap_df.query("displayName == 'Football'").iloc[0].x
   
        return self._cov_seq
    
    
    @property
    def coverage_full(self): 
        '''Returns individual coverage with redEPA for each defender involved'''
        assert(self._compProbaModel is not None and self._estYardsModel is not None)
        
        cov_seq = self.coverage_seq.copy()
        cov_seq['estYards'] = self._estYardsModel.predict(
                                    cov_seq[self.estYardsFeatures]).round(0)
        
        cov_seq = cov_seq.merge(self.epa_df['est_epa'], left_on = 'estYards', 
                                right_index = True, how = 'left')
        
        cov_seq.loc[cov_seq.frameId < self._snap_frame + self._nothrow_frames,
                    'est_epa'] = 0
        cov_seq.loc[cov_seq.estYards > self.epa_df.index.max(), 
                    'est_epa'] = self.epa_df['est_epa'].max()
        cov_seq.loc[cov_seq.estYards < self.epa_df.index.min(), 
                    'est_epa'] = self.epa_df['est_epa'].min()

        cov_seq['_zero_cov'] = 0.0
        cov_seq['OpenCatchProba'] = self._compProbaModel.predict_proba(cov_seq[['_zero_cov',                                             
                        'vel_x','vel_y_abs', 'xDiff','yDiff_abs','projxDiff', 'projyDiff', 
                         ]])[:,1]
        cov_seq['CovCatchProba'] = self._compProbaModel.predict_proba(\
                                cov_seq[self.compProbaFeatures])[:,1]
            
        oppo_qb = ((cov_seq['qb_o_rad'] > self.qb_playact_orad[0]) & 
                   (cov_seq['qb_o_rad'] < self.qb_playact_orad[1]) )
        end_oppo_frame = cov_seq.loc[oppo_qb].frameId.max()
        end_oppo = ((cov_seq.frameId >= end_oppo_frame) &
                    (cov_seq.frameId < end_oppo_frame + self._nothrow_frames))
        blocking_rbte = ((cov_seq['position'].isin(['RB','TE'])) & 
                       (cov_seq['mu_x'] < cov_seq['snap_x']))            
        to_zero = (oppo_qb | end_oppo | blocking_rbte)
        cov_seq.loc[to_zero,'OpenCatchProba'] = 0    
        cov_seq.loc[to_zero,'CovCatchProba'] = 0       
        cov_seq.loc[to_zero,'est_epa'] = 0   
        
        cov_seq['est_epa_POS'] = (cov_seq['est_epa'] > 0)*cov_seq['est_epa']
        cov_seq['zeroEpa'] = cov_seq['OpenCatchProba'] * cov_seq['est_epa_POS']
        cov_seq['reducedEpa'] = (cov_seq['OpenCatchProba'] - cov_seq['CovCatchProba'])*cov_seq['est_epa_POS']
        cov_seq['reducedEpa'] = (cov_seq['reducedEpa'] > 0)*cov_seq['reducedEpa']
        
        cov_seq['def0_weight'] = 1 - (cov_seq['def0dist_to_rec'] / (cov_seq['def0dist_to_rec'] \
                                                                  + cov_seq['def1dist_to_rec'] ))
        cov_seq['oppsEpaAsMain'] = cov_seq['def0_weight']*cov_seq['zeroEpa']
        cov_seq['redEpaAsMain'] = cov_seq['def0_weight']*cov_seq['reducedEpa']
        cov_seq['oppsEpaAsHelp'] = cov_seq['zeroEpa'] - cov_seq['oppsEpaAsMain']
        cov_seq['redEpaAsHelp'] = cov_seq['reducedEpa'] - cov_seq['redEpaAsMain']
        
        return cov_seq[['frameId','position', 'jerseyNumber', 'nflId', 'displayName', 
              'route', 'avgCoverage','connIndex','def0jersey','def0position', 'def0nflId', 
              'def0dist_to_rec', 'def1jersey', 'def1position', 'def1nflId','def1dist_to_rec',
              'estYards', 'est_epa','OpenCatchProba', 'CovCatchProba','zeroEpa', 'reducedEpa', 
              'def0_weight', 'oppsEpaAsMain','redEpaAsMain', 'oppsEpaAsHelp', 'redEpaAsHelp',
              'gameId','playId']]
    

        
    @property
    def epa_df(self):
        # print("Called get epa_df")
        if self._epa_play is None:
            assert (self._epModel is not None)
            if not hasattr(self.a_play, 'est_ep'):
                ep_feats = self.a_play[self.epaFeatures].values.reshape(1,-1)
                self.a_play['est_ep'] = self._epModel.predict(ep_feats)[0]
                
            self._epa_play = self._getEstEpa_Result(self.a_play, 
                                                     self._epModel,
                lowest_x = self.play_df[self.play_df.isOffence == 1].x.min(),
                highest_x = min(self.play_df[self.play_df.isOffence == 1].x.max()+15, 120))
        
        return self._epa_play

    
    def coverage_agg(self, add_pIndex = False):
        cov_full = self.coverage_full[self.coverage_full['est_epa'] > 0]
        cov_full0 = cov_full.groupby(['def0nflId', 'def0jersey', 
                    'def0position'])[['redEpaAsMain','oppsEpaAsMain']].sum()
        cov_full1 = cov_full.groupby(['def1nflId', 'def1jersey', 
                    'def1position'])[['redEpaAsHelp','oppsEpaAsHelp']].sum()
        
        cov_full0['avgWeightAsMain'] = cov_full.groupby(['def0nflId', 'def0jersey', 
                    'def0position'])[['def0_weight']].mean()
        cov_full1['avgWeightAsHelp'] = (1 - cov_full.groupby(['def1nflId', 'def1jersey', 
                    'def1position'])[['def0_weight']].mean())
        
        cov_full0['totFramesAsMain'] = cov_full.groupby(['def0nflId', 'def0jersey', 
                    'def0position'])[['frameId']].nunique().fillna(0).astype(int)
        cov_full1['totFramesAsHelp'] = cov_full.groupby(['def1nflId', 'def1jersey', 
                    'def1position'])[['frameId']].nunique().fillna(0).astype(int)

        cov_fulltot = pd.concat([cov_full0, cov_full1], axis = 1).fillna(0)

        cov_fulltot['redEpa'] = cov_fulltot['redEpaAsMain']+ cov_fulltot['redEpaAsHelp']
        cov_fulltot['redEpaOpps'] = cov_fulltot['oppsEpaAsMain']+ cov_fulltot['oppsEpaAsHelp']
        cov_fulltot['redEpaPerc'] = cov_fulltot['redEpa'] / cov_fulltot['redEpaOpps']

        cov_fulltot = cov_fulltot[['redEpaPerc','redEpa', 'redEpaOpps', 
                                   'redEpaAsMain', 'oppsEpaAsMain', 'redEpaAsHelp',  
                                    'oppsEpaAsHelp','totFramesAsMain','totFramesAsHelp',
                                    'avgWeightAsMain', 'avgWeightAsHelp']]
        
        if add_pIndex: 
                cov_fulltot.index.names = ['nflId','jerseyNumber','position']
                cov_fulltot = cov_fulltot.reset_index()
                cov_fulltot['pIndex'] = self.pIndex
                cov_fulltot['week'] = self.a_play['week']
                cov_fulltot['defTeam'] = self.def_team

        return cov_fulltot.sort_values('redEpaOpps', ascending = False).round(2)

    def plot_data(self, pitch_control = True):
        '''Returns data for plotting using Animate Classes'''
        if pitch_control: 
            conn_colors = self.coverage_full
            conn_colors = conn_colors[['frameId','nflId','def0nflId',\
                'est_epa','connIndex']].sort_values(by = ['frameId','connIndex'])
            
            ## names to Jersey
            names = self.play_df[self.play_df.frameId == 1].sort_values(by = ['isOffence',\
                                'jerseyNumber'])[['jerseyNumber','displayName','position']]
            names = names.dropna().rename(columns = {'position':'POS'})
            player_name = names.displayName.str.extract(r'(?P<INITIAL>\w{1}).+\s(?P<LAST>\w+)')
            player_name = player_name.INITIAL + '.' + player_name.LAST
            names['dName'] = player_name.str[:7]
            names = names.set_index('jerseyNumber', drop = True).drop(columns = 'displayName')
            names.index.name = None        

            return (self.play_df[self.play_df.frameId < 100], 
                    self.coverage_Z, self.play_text, names, conn_colors)
        
        else: ##normal animation
            return (self.play_df, self.play_text)
    
    
    @staticmethod
    def calculateZfromtrack(seq_df, MAX_FIELD_Y = 160/3 , MAX_FIELD_X = 120, 
        MAX_PLAYER_SPEED = 11.3, ORIENT_MAG = 2,
        MU_A = 0.000486,MU_B = 0.023,MU_C = 0.29, VMAX_MU = 9.5,
        LB_EXP = 2, OFF_CUTOFF = 0.9,
        N_GRID = 121, K_EXPIT = 3, SCALE_OFFENCE = 1, SCALE_DEFENCE = 1):
        """
        Calculates catchability and coverage for both teams.
        Parameters
        ----------
        ad : DATAFRAME Single play tracking data preprocessed by NFLData
        Can be used for only a few frames rather than entire play eg.
        ad =  play_df[play_df.frameId.isin([34,35,36,37])]
    
        Returns
        -------
        TYPE
            Z_diff : Numpy Array  (N by N array indexed by maxframeID)
            Z_indiv: Numpy Array (N by N by maxframeId by jerseyRank, by isOffence)
    
        """
        # print('Calculating Zs')
        def polar_to_z(r, theta):
            """returns x and y components given length and angle of vector"""
            xy = r * np.exp( 1j * theta)
            return np.real(xy), np.imag(xy)
        
        def weighted_angle_magnitude(a1, a2, speed):
            if np.isnan(a1[0]):
                return np.nan, np.nan
            def normalize(v):
                norm=np.linalg.norm(v, ord=None)
                if norm==0:
                    norm=np.finfo(v.dtype).eps
                return v/norm
            
            norm_weighted = speed*normalize(a1) + (1-speed)*normalize(a2)
            angle = np.arctan2(norm_weighted[1], norm_weighted[0]) % (2*np.pi)
            magnitude = np.sqrt(norm_weighted[0]**2 + norm_weighted[1]**2)
            
            return angle, magnitude
        
        def generate_mu(player_position, player_vel, player_orient, player_accel, 
                        distance_from_football,football_x, football_y, *argv ):
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
    
            if player_position[0] <= football_x:
                time_of_pass = MU_C 
                mu_generated = player_position + time_of_pass*(player_vel) + \
                    (time_of_pass**2)*0.5*player_accel
                return mu_generated[0], mu_generated[1]
            
            ## recursively find location of pass and distance to where player is GOING
            time_of_pass1 = MU_A*(distance_from_football**2) + \
                MU_B*distance_from_football + MU_C
                
            player_future = player_position + \
                time_of_pass1*(player_vel) + (time_of_pass1**2)*0.5*player_accel
                
            distance_future = np.sqrt((player_future[0] - football_x)**2 + \
                                      (player_future[1] - football_y)**2)
            
            time_of_pass2 = MU_A*(distance_future**2) + MU_B*distance_future + MU_C
            
            if (np.linalg.norm(player_vel + player_accel*time_of_pass2) > VMAX_MU):
                ## reaching beyond max speed
                if np.linalg.norm(player_vel) > VMAX_MU \
                or np.linalg.norm(player_accel)<0.01 :
                    
                    mu_generated = player_position + time_of_pass2*(player_vel)
                else:     
                    time_to_max = (VMAX_MU - np.linalg.norm(player_vel))/np.linalg.norm(player_accel)
                    assert(time_to_max <= time_of_pass2)
                    player_max_vel = player_vel + player_accel*time_to_max
                    time_after_max = time_of_pass2 - time_to_max
                    
                    mu_generated = player_position + time_to_max*(player_vel) + \
                        (time_to_max**2)*0.5*player_accel + player_max_vel*time_after_max
            
            else: ## did not reach max speed
                mu_generated = player_position + time_of_pass2*(player_vel) + \
                    (time_of_pass2**2)*0.5*player_accel
            assert(np.isnan(mu_generated[0]) == False 
                   and np.isnan(mu_generated[1]) == False)
            return mu_generated[0], mu_generated[1]
        
        def radius_influence(x):
            assert x >= 0
            if x <= 18: return (2 + (3/(18**2))*(x**2))
            else: return 5
    
        def generate_sigma(influence_rad, player_speed, distance_from_football, 
                           player_x, football_x):
            
            R = np.array([[np.cos(influence_rad), -np.sin(influence_rad)],
                          [np.sin(influence_rad), np.cos(influence_rad)]])
        
            speed_ratio = (player_speed**2)/(MAX_PLAYER_SPEED**2)
            if player_x > football_x:
                radius_infl = radius_influence(distance_from_football)
            else: radius_infl = 2
            
            S = np.array([[radius_infl + (radius_infl*speed_ratio), 0], 
            [0, radius_infl - (radius_infl*speed_ratio)]])
            
            return R@(S**2)@R.T
        
        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos.
        
            pos is an array constructed by packing the meshed arrays of variables
            x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        
            """
            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2*np.pi)**n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        
            return np.exp(-fac / 2) / N
        
        def generate_data_grid(N = 120):
            # Our 2-dimensional distribution will be over variables X and Y
            X = np.linspace(0, MAX_FIELD_X, N)
            Y = np.linspace(0, MAX_FIELD_Y, N)
            X, Y = np.meshgrid(X, Y)
             
            # # Mean vector and covariance matrix
            # mu = np.array([0., 1.])
            # Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])
             
            # Pack X and Y into a single 3-dimensional array
            pos = np.zeros(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
             
            return X, Y, pos
        
        ##END DEFINITION OF FUNCTIONS, use df.apply()
        ad = seq_df.copy()
        ad['speed_w'] = ad.s / MAX_PLAYER_SPEED 
        ad['vel_x'], ad['vel_y'] = zip(*ad.apply(lambda row: \
                                                 polar_to_z(row.s, 
                                                            row.dir_rad), 
                                                 axis = 1))
            
        ad['orient_x'], ad['orient_y'] = zip(*ad.apply(lambda row: \
                                                       polar_to_z(ORIENT_MAG, 
                                                                  row.o_rad), 
                                                       axis = 1))
 
        ad['influence_rad'], ad['influence_mag']= zip(*ad.apply(lambda row: \
                         weighted_angle_magnitude(np.array([row.vel_x, row.vel_y]), 
                                                  np.array([row.orient_x, row.orient_y]), 
                                                  row.speed_w),
                         axis = 1))
    
        ad['dvel_x'] = 10*(ad['vel_x'] - ad.groupby('nflId')['vel_x'].shift(1))
        ad['dvel_y'] = 10*(ad['vel_y'] - ad.groupby('nflId')['vel_y'].shift(1))
        
        ad['dvel_x'] = ad['dvel_x'].fillna(0)
        ad['dvel_y'] = ad['dvel_y'].fillna(0)
        
        ad = ad.merge(ad[ad.team == 'football'][['frameId','x','y']].rename(
                columns = {'x':'x_fb', 'y':'y_fb'}),
                on = 'frameId')
        
        ad['distance_from_football'] = np.sqrt((ad.x - ad.x_fb)**2 + (ad.y - ad.y_fb)**2)
        
        ## generate mu
        ad['mu_x'], ad['mu_y'] = zip(*ad.apply(lambda row: \
                                                generate_mu(np.array([row.x, row.y]), 
                                                np.array([row.vel_x, row.vel_y]), 
                                                np.array([row.orient_x, row.orient_y]),
                                                np.array([row.dvel_x, row.dvel_y]),
                                                row.distance_from_football,
                                                row.x_fb,
                                                row.y_fb ), axis = 1))
        
        ad['sigma_mtx'] = ad.apply(lambda row: \
                                   generate_sigma(row.influence_rad, 
                                                  row.s, 
                                                  row.distance_from_football, 
                                                  row.x, 
                                                  row.x_fb), axis = 1)

        
        X, Y, GRID = generate_data_grid(N = N_GRID)
        Z_off = np.zeros((ad.frameId.max()+1, N_GRID, N_GRID))
        Z_def = np.zeros((ad.frameId.max()+1, N_GRID, N_GRID))
        Z_indiv = np.zeros((ad.frameId.max()+1, 2, int(ad.jerseyRank.max()+1),
                            N_GRID, N_GRID))

        for _,row in ad.iterrows():
            if row.team == 'football':
                continue
            
            Z = multivariate_gaussian(GRID, 
                                      np.array([row.mu_x, row.mu_y]), 
                                      row.sigma_mtx)
            Z_coarse = np.where(Z > 0.0001, Z, 0)
            if np.count_nonzero(Z_coarse) == 0:
                Z_coarse = np.where(Z >= Z.max(), Z, 0)
            if row.isOffence == 1:
                Z_norm = Z_coarse/(Z.max()*SCALE_OFFENCE)
                Z_off[row.frameId,:,:] = np.maximum(Z_off[row.frameId,:,:], Z_norm)
                Z_indiv[row.frameId, 1, row.jerseyRank,:,:] = np.where(Z_norm > OFF_CUTOFF, 1, 0)
                               
            elif row.isOffence == 0:
                Z_norm = Z_coarse/(Z.max()*SCALE_DEFENCE)
                if row.position in ['MLB','OLB', 'ILB', 'LB','DL','DE','NT']:
                    Z_norm = Z_norm**LB_EXP
                    
                Z_def[row.frameId,:,:] += Z_norm
                Z_indiv[row.frameId,0, row.jerseyRank,:,:] = np.where(Z_norm > 0.01, Z_norm, 0) 
        
        Z_diff = np.zeros((ad.frameId.max()+1, N_GRID, N_GRID))
        
        for i in range(ad.frameId.max()+1):
            Z_diff[i,:,:] = expit(K_EXPIT*(Z_def[i,:,:] - Z_off[i,:,:]))
        
        
        ad = ad[['frameId','nflId','vel_x','vel_y','dvel_x', 'dvel_y','mu_x','mu_y']]
        
        return { 'Z_tuple': (Z_diff, Z_indiv), 'proj_vel_df': ad}
    
     
    def _getEstEpa_Result(self, a_play: pd.Series,
                      model: BaseEstimator, 
                      lowest_x: float,
                      highest_x: float) -> pd.DataFrame:
        """
        Parameters
        ----------
        a_play : pd.Series
            Row of current data
        model : BaseEstimator
            To estimate EP
        lowest_x : float
            farthest point back from scrimmage
        highest_x : float
            farthest point fwd from scrimmage

        Returns
        -------
        epa_play : TYPE
            dataframe of possible outcomes and EPA of those outcomes
        """
        # print('Getting EPA Result')
        def getNextEPFeats(row):
            """apply functin to dataframe with these four features:
                'down', 'yardsToGo', 'yardsToEndzone',
               'secondsUntilHalf'
            Returns: next set of features (downs, yardsToGo etc..) given result of play """
            if hasattr(row, 'playResult'): result = row.playResult
            elif hasattr(row, 'catch_Result'): result = row.catch_Result
            else: raise Exception(row)
            
            if hasattr(row, 'framesToEvent'): playtime = row.framesToEvent/10
            else: playtime = 10
            
            is_turnover = False

            if result >= row.yardsToGo:
                ##made first down
                next_down = 1 
            elif result < row.yardsToGo:
                next_down = row.down + 1
            elif row.down  == 4  and result < row.yardsToGo:
                is_turnover = True
                next_down = 1

            if not is_turnover:
                next_yardsEZ = row.yardsToEndzone - result
                next_secondsUF = row.secondsUntilHalf - playtime - 30
                if row.secondsUntilHalf < 120:
                    next_secondsUF = max([row.secondsUntilHalf - playtime - 10, 0])
                if next_down == 1:
                    next_yardsTG = 10
                else:
                    next_yardsTG = row.yardsToGo - result  
            return next_down, next_yardsTG, next_yardsEZ, next_secondsUF     

        yards_min = lowest_x - (110 - a_play.yardsToEndzone)
        yards_max = highest_x - (110 - a_play.yardsToEndzone) 

        epa_play = pd.DataFrame(index = np.arange(math.floor(yards_min),math.ceil(yards_max)))
        
        for col in ['down', 'yardsToGo', 'yardsToEndzone','secondsUntilHalf', 'est_ep']:
            epa_play[col] = a_play[col]
        epa_play['catch_Result'] = np.arange(math.floor(yards_min),math.ceil(yards_max))
        
        epa_play['next_down'], epa_play['next_yardsTG'], epa_play['next_yardsEZ'], \
            epa_play['next_secondsUF'] = zip(*epa_play.apply(lambda X: getNextEPFeats(X), axis = 1))
        
        inputs = [list(item) for item in epa_play.apply(lambda X: getNextEPFeats(X), axis = 1)]
        
        epa_play['next_est_ep'] = model.predict(inputs)
        epa_play['est_epa'] = epa_play['next_est_ep'] - epa_play['est_ep']
    
        return epa_play
    

    @staticmethod
    def getCoveragefromZ(row: pd.Series,
                     a_play: pd.Series,
                     Z_diff_indiv: typing.Tuple,
                     seq_df: pd.DataFrame) -> pd.Series:

        Z_diff = Z_diff_indiv[0]
        Z_indiv = Z_diff_indiv[1]

        rowZindiv = Z_indiv[row.frameId, 1, row.jerseyRank,:,:]         
        rowZdiff = Z_diff[row.frameId,:,:]
        indivBool = np.array(rowZindiv, dtype = bool)
        Zdiffmasked = rowZdiff[indivBool]
        

        r_sta = pd.Series(dtype = 'float64', name = row.name)

        for col in ['nflId','frameId','mu_x', 'mu_y']:
            r_sta[col] = row[col]

        track_def = seq_df[(seq_df.frameId == row.frameId) & (seq_df.isOffence == 0)].copy()
        
        track_def['x_diff'] = track_def['mu_x'] - r_sta['mu_x']
        track_def['y_diff'] = track_def['mu_y'] - r_sta['mu_y']
        track_def['dist_to_rec'] = np.sqrt(track_def['x_diff']**2 + track_def['y_diff']**2) 
        
        track_def = track_def.sort_values(by = 'dist_to_rec').rename(columns = {'jerseyNumber':'jersey'})
           
        for nearest in [0,1]:
            _r_def = track_def.iloc[nearest]
            for cat in ['jersey','position','nflId','dist_to_rec', 'x_diff','y_diff']:
                r_sta['def'+str(nearest)+cat] = _r_def[cat]
        
        r_sta['avgCoverage'] = Zdiffmasked.mean()
            
        return r_sta





# %%

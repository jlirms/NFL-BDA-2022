#%% Imports

from b2_animate_data import AnimatePlay 
from b1_load_data import NFLData

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import get_cmap
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
pd.set_option("display.max_columns", 101)


#%%

class AnimatePlayPitchControl(AnimatePlay):
    '''Plots pitch control animation given: single play tracking'''
    
    def __init__(self, play_df, Z_diff_indiv, 
                play_text = None, names = None, conn_colors = None, events = None,
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
        self._MAX_FIELD_PLAYERS = 24
        self._conn_colors = conn_colors
        self._show_control = show_control
        self._play_text = play_text
        self._opt_text = opt_text
        self._names = names
        self._events = events
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

        if self._events is not None:
            fontsize = 7 if self._show_control else 8
            self._ax_field.text(20, 1, self._events.to_string(),\
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
            for c in self._conn_colors.connIndex.unique():
                self._conn_colors_list.append(self._ax_field.plot([],[],
                                            color = self._conn_cmap(255*c), 
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

        for label in pos_df.isKicking.unique():
            label_data = pos_df[pos_df['isKicking']== label]
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
                
            for conn_index, dfconn in current_conn.groupby('connIndex'):         
                group = pos_df[pos_df.nflId.isin(dfconn['nflId'].values)]
                self._conn_colors_list[conn_index][0].set_data(np.vstack([group.x, group.y]))
        
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

            if row.isKicking == 1 and row.jerseyRank < self._Z_indiv.shape[2]:
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
                    
            elif row.isKicking == 0 and row.jerseyRank < self._Z_indiv.shape[2]:
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


#%%Names 

def getEventsfromtrack(play_df):
    events = play_df.groupby('event').first()['frameId'].sort_values()
    events = events.drop('None')
    events.index.name = None
    return events

def getNamesdffromtrack(play_df):
    names = play_df[
        (play_df.frameId == play_df.frameId.min()) &
        ((play_df.isGunner == 1) | (play_df.isVise == 1))]

    names = names.sort_values(
        by = ['isKicking','jerseyNumber'])[['isKicking','jerseyNumber','displayName', 'position']]
    names = names.dropna().rename(columns = {'position':'Pos'})
    player_name = names.displayName.str.extract(r'(?P<INITIAL>\w{1}).+\s(?P<LAST>\w+)')

    player_name = player_name.INITIAL + '.' + player_name.LAST
    names['DName'] = player_name.str[:7]
    names = names.set_index('jerseyNumber', drop = True).drop(columns = 'displayName')
    names.index.name = None    

    return names

def getGVConnfromtrack(play_df):
    gunners_vises = play_df[
        (play_df['isGunner'] == 1) |
        (play_df['isVise'] == 1)
    ][['frameId','y','nflId']]
    ## Take half of field for now
    gunners_vises['connIndex'] = (gunners_vises['y'] <= 160/3/2).astype(int)

    return gunners_vises


#%%

### keep it simple with simple functions

## one function for getting the out, from data in 
## one functin for getting the names
## one function for getting the connections



## I like fish and chips






#%% Imports
from b4_heatmap_anim import getNamesdffromtrack, \
        getGVConnfromtrack, getEventsfromtrack, \
            AnimatePlayPitchControl
from b3_fit_control_data import calculateZfromtrack
from b2_animate_data import open_html_plot
from b1_load_data import NFLData

#%% Loading in data
path_to_data = 'input/nfl-big-data-bowl-2022/'
path_to_punts = 'input/BDB2022-custom/punts_only/' 
data = NFLData(data_path = path_to_data, punts_path = path_to_punts)
dfgames = data.games_df()
dfplays = data.plays_df()

def makeAnimation(data, pIndex):
    one_play = data.one_play(pIndex)
    one_track = one_play['track']
    out_z_data = calculateZfromtrack(one_track)
    names_df = getNamesdffromtrack(one_track)
    connect_df = getGVConnfromtrack(one_track)
    events_df = getEventsfromtrack(one_track)

    ani_obj = AnimatePlayPitchControl(
        play_df = one_track, 
        Z_diff_indiv = out_z_data.Z_tuple, 
        play_text = one_play['row'].playDescription, 
        names = names_df,
        events = events_df, 
        conn_colors= connect_df,
        show_control=False)

    open_html_plot(
        ani_obj.ani, 
        fname = f"plots/p{pIndex}_{events_df.index[-1]}.html", 
        open_browser = False)


makeAnimation(data,50)
#%%Full game TB vs NOR
# single_game_plays = dfplays[dfplays.gameId == 2018090906]

# for ind in single_game_plays.index:
#     makeAnimation(data,ind)


#%% Imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import re
pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 101)
pd.set_option('display.max_colwidth', 201)

#%%Plays data

plays = pd.read_csv('input/nfl-big-data-bowl-2022/plays.csv')
plays['scoreDiff'] = abs(plays.preSnapHomeScore-plays.preSnapVisitorScore)
plays.head(2)

#%%Null values
plays.isnull().sum()
#%% Play types and result
plays['specialTeamsPlayType'].value_counts()
# plays['specialTeamsResult'].value_counts()

(plays.query('specialTeamsPlayType == "Kickoff"')['specialTeamsResult'] \
    .value_counts() / len(plays.query('specialTeamsPlayType == "Kickoff"'))) \
    .to_frame()

#%%
(plays.query('specialTeamsPlayType == "Punt"')['specialTeamsResult'] \
    .value_counts() / len(plays.query('specialTeamsPlayType == "Punt"'))) \
    .to_frame()

#%%
(plays.query('specialTeamsPlayType == "Field Goal"')['specialTeamsResult'] \
    .value_counts() / len(plays.query('specialTeamsPlayType == "Field Goal"'))) \
    .to_frame()

# %% Scouting data
pff = pd.read_csv('input/nfl-big-data-bowl-2022/PFFScoutingData.csv')
print("Num plays: {}, num scount: {}".format(plays.shape[0], pff.shape[0]))
pff.head(5)

#%% ## Just focusing on Punts
punts = plays.query('specialTeamsPlayType == "Punt"')
punts.head()
#%% 
punts['specialTeamsResult'].value_counts()

#%% Look at each result independently, for which plays are important columns null
for result, dfr in punts.groupby('specialTeamsResult'):
    print(result, dfr.shape[0], dfr['kickLength'].isnull().sum())
    # print("\n\n")
## kickReturnYardage is only NOT null for ["Return", "Muffed"]
## kickBlockerId NOT null only for ["Blocked"]
## returnerId NOT null only for ["Fair Catch", "Muffed", "Return"]
## kickReturnYardage NOT null only for ["Muffed", "Return"]
## kickLength NOT null only for ["Downed","Fair Catch", "Muffed", "Out of Bounds", "Return", "Touchback"] ## null for Blocked and Non Special...


#%%
# fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8), (ax9,ax10)) = plt.subplots(5,2, figsize=(15,20))
# plays.specialTeamsPlayType.value_counts().plot.bar(
#     title='Play type', ax=ax1)
# plays.specialTeamsResult.value_counts().plot.bar(
#     title='Play result breakdown', ax=ax2)
# plays.yardsToGo.plot.hist(
#     bins=20, title='Yards to go at play start', grid=True, ax=ax3)
# plays.playResult.plot.hist(
#     bins=50, title='Play result (yds)', grid=True, ax=ax4)
# plays.kickLength.plot.hist(
#     bins=50, title='Kick length', grid=True, ax=ax7)
# plays.loc[plays.kickReturnYardage.notnull()]['kickReturnYardage'].plot.hist(
#     bins=50, title='Return result (yds)', grid=True, ax=ax8)
# plays.penaltyYards.plot.hist(
#     title='Penalty yards', grid=True, ax=ax5)
# plays.penaltyCodes.value_counts()[:10].plot.bar(
#     title='Penalty codes (top 10)', ax=ax6)
# plays.loc[plays.passResult.notnull()]['passResult'].value_counts().plot.bar(
#     title='Pass result breakdown', ax=ax9)
# plays.yardlineNumber.plot.hist(
#     bins=20, title='Where plays happen (yardline #)', grid=True, ax=ax10)
# plt.tight_layout()


#%%
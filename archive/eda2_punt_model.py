#%% Imports
from b1_load_data import NFLDataError, NFLData
import pandas as pd
import numpy as np

#%% Data
path_to_data = 'input/nfl-big-data-bowl-2022/'
path_to_punts = 'input/BDB2022-custom/punts_only/' 
data = NFLData(data_path = path_to_data, punts_path = path_to_punts)

dfp = data.merged_plays()
dfp.columns
#%% Look at features, EDA pt 2 
end_events = ('fair_catch','punt_received','punt_land','out_of_bounds','touchback','punt_downed')

#%% Plots of hangTime

dfp.hangTime.hist(bins = 20)

#%%
dfp.specialTeamsResult.value_counts() / dfp.shape[0]

### For now just look at Return and Fair Catch
### Later consider touchbacks vs brought out 

#%%

puntret = dfp[dfp.specialTeamsResult == 'Return'].copy()
puntret['kickReturnYardage'] = puntret['kickReturnYardage'].clip(lower = 1)
puntret['logkickReturnYardage'] = np.log(puntret['kickReturnYardage']-min(puntret.kickReturnYardage)+1)
puntret.logkickReturnYardage.hist(bins = 20)

#%%
puntfair = dfp[dfp.specialTeamsResult == 'Fair Catch'].copy()
puntfair['kickReturnYardage'] = 1
puntfair['logkickReturnYardage'] = np.log(1)

## Fair catch generally means: good gunning
##      OR risk adverse with bad gunning
## Negative return generally means: good gunning
##      OR  


## if classification decide if negative or zero returns should be lumped with fair catch

## if regression....

# %%




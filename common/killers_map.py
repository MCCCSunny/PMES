"""Define the maps config with rewards changed.
"""

from pysc2.maps import lib


class MineMiniGame(lib.Map):
    directory = "killers_map"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


mini_games_edited = [
    "BuildMarines_15min",
    "BuildMarinesA_4-6min_RP_new1",
    "BuildMarinesB_4-6min_RP_new1",
    "BuildMarinesC_4-6min_ARP_newx",   
    "BuildMarines_15min_shaping_ARP_111", 
    "BuildMarines_15min_shaping_ARP_124",
    "BuildMarines_15min_shaping_ARP_139"  
]

for name in mini_games_edited:
    globals()[name] = type(name, (MineMiniGame, ), dict(filename=name))

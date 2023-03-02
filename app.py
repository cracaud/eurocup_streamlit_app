import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import numpy as np
import math
from scipy import stats
from PIL import Image
from mplsoccer import PyPizza, add_image, FontManager
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle, Arc

import matplotlib as mpl
mpl.rcParams["axes.spines.right"] = True
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

font_dirs = ["//Users//sissigarduno//Downloads"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = "Poppins"
plt.rcParams['font.size'] = '10.6'

image = Image.open("EUROCUP.png")
st.sidebar.image(image)


#Web scraping of data
df = pd.read_csv("Data Eurocup - Stats.csv")
df_shots = pd.read_csv("Data Eurocup - Shots.csv")
df_round = pd.read_csv("Data Eurocup - Shots.csv")

#Filters
st.sidebar.header('Filters')
st.sidebar.write("Select a range of rounds :")
begin = st.sidebar.slider('First game :', 1, 18, 1)
end = st.sidebar.slider('Last game :', 1, 18, 18)

#Filtering data (TO FINISH)
df = df[df['Round'].between(begin, end)]
df['Pace_home'] = df["FieldGoalsAttempted2_home"] + df["FieldGoalsAttempted3_home"] + 0.44*df["FreeThrowsAttempted_home"] - df["OffensiveRebounds_home"] + df["Turnovers_home"]
df['Pace_away'] = df["FieldGoalsAttempted2_away"] + df["FieldGoalsAttempted3_away"] + 0.44*df["FreeThrowsAttempted_away"] - df["OffensiveRebounds_away"] + df["Turnovers_away"]

home = df.groupby(['Team_home']).agg({'Points_home':'sum','Pace_home':'sum','FieldGoalsMade2_home':'sum','FieldGoalsAttempted2_home':'sum',
                                      'FieldGoalsMade3_home':'sum','FieldGoalsAttempted3_home':'sum','FreeThrowsMade_home':'sum',
                                      'FreeThrowsAttempted_home':'sum','OffensiveRebounds_home':'sum','DefensiveRebounds_home':'sum',
                                      'TotalRebounds_home':'sum','Assistances_home':'sum','Steals_home':'sum','Turnovers_home':'sum',
                                      'BlocksFavour_home':'sum','BlocksAgainst_home':'sum','FoulsCommited_home':'sum',
                                      'FoulsReceived_home':'sum','Valuation_home':'sum','Points_away':'sum','Pace_away':'sum','FieldGoalsMade2_away':'sum',
                                      'FieldGoalsAttempted2_away':'sum','FieldGoalsMade3_away':'sum','FieldGoalsAttempted3_away':'sum',
                                      'FreeThrowsMade_away':'sum','FreeThrowsAttempted_away':'sum','OffensiveRebounds_away':'sum',
                                      'DefensiveRebounds_away':'sum','TotalRebounds_away':'sum','Assistances_away':'sum',
                                      'Steals_away':'sum','Turnovers_away':'sum','BlocksFavour_away':'sum','BlocksAgainst_away':'sum',
                                      'FoulsCommited_away':'sum','FoulsReceived_away':'sum','Valuation_away':'sum'})
home.rename(columns={'Points_home': 'Points_team', 'Pace_home': 'Pace_team', 'FieldGoalsMade2_home': 'FieldGoalsMade2_team', 'FieldGoalsAttempted2_home': 'FieldGoalsAttempted2_team', 
                     'FieldGoalsMade3_home': 'FieldGoalsMade3_team', 'FieldGoalsAttempted3_home': 'FieldGoalsAttempted3_team', 
                     'FreeThrowsMade_home': 'FreeThrowsMade_team', 'FreeThrowsAttempted_home': 'FreeThrowsAttempted_team', 
                     'OffensiveRebounds_home': 'OffensiveRebounds_team', 'DefensiveRebounds_home': 'DefensiveRebounds_team', 
                     'TotalRebounds_home': 'TotalRebounds_team', 'Assistances_home': 'Assistances_team', 'Steals_home': 'Steals_team', 
                     'Turnovers_home': 'Turnovers_team', 'BlocksFavour_home': 'BlocksFavour_team', 'BlocksAgainst_home': 'BlocksAgainst_team', 
                     'FoulsCommited_home': 'FoulsCommited_team', 'FoulsReceived_home': 'FoulsReceived_team', 'Valuation_home': 'Valuation_team', 
                     'Points_away': 'Points_opp', 'Pace_away': 'Pace_opp', 'FieldGoalsMade2_away': 'FieldGoalsMade2_opp', 'FieldGoalsAttempted2_away': 'FieldGoalsAttempted2_opp', 
                     'FieldGoalsMade3_away': 'FieldGoalsMade3_opp', 'FieldGoalsAttempted3_away': 'FieldGoalsAttempted3_opp', 
                     'FreeThrowsMade_away': 'FreeThrowsMade_opp', 'FreeThrowsAttempted_away': 'FreeThrowsAttempted_opp', 
                     'OffensiveRebounds_away': 'OffensiveRebounds_opp', 'DefensiveRebounds_away': 'DefensiveRebounds_opp', 
                     'TotalRebounds_away': 'TotalRebounds_opp', 'Assistances_away': 'Assistances_opp', 'Steals_away': 'Steals_opp', 
                     'Turnovers_away': 'Turnovers_opp', 'BlocksFavour_away': 'BlocksFavour_opp', 'BlocksAgainst_away': 'BlocksAgainst_opp', 
                     'FoulsCommited_away': 'FoulsCommited_opp', 'FoulsReceived_away': 'FoulsReceived_opp', 
                     'Valuation_away': 'Valuation_opp'}, inplace=True)

away = df.groupby(['Team_away']).agg({'Points_home':'sum','Pace_home':'sum','FieldGoalsMade2_home':'sum','FieldGoalsAttempted2_home':'sum',
                                      'FieldGoalsMade3_home':'sum','FieldGoalsAttempted3_home':'sum','FreeThrowsMade_home':'sum',
                                      'FreeThrowsAttempted_home':'sum','OffensiveRebounds_home':'sum','DefensiveRebounds_home':'sum',
                                      'TotalRebounds_home':'sum','Assistances_home':'sum','Steals_home':'sum','Turnovers_home':'sum',
                                      'BlocksFavour_home':'sum','BlocksAgainst_home':'sum','FoulsCommited_home':'sum',
                                      'FoulsReceived_home':'sum','Valuation_home':'sum','Points_away':'sum','Pace_away':'sum','FieldGoalsMade2_away':'sum',
                                      'FieldGoalsAttempted2_away':'sum','FieldGoalsMade3_away':'sum','FieldGoalsAttempted3_away':'sum',
                                      'FreeThrowsMade_away':'sum','FreeThrowsAttempted_away':'sum','OffensiveRebounds_away':'sum',
                                      'DefensiveRebounds_away':'sum','TotalRebounds_away':'sum','Assistances_away':'sum',
                                      'Steals_away':'sum','Turnovers_away':'sum','BlocksFavour_away':'sum','BlocksAgainst_away':'sum',
                                      'FoulsCommited_away':'sum','FoulsReceived_away':'sum','Valuation_away':'sum'})
away.rename(columns={'Points_home': 'Points_opp', 'Pace_home': 'Pace_opp', 'FieldGoalsMade2_home': 'FieldGoalsMade2_opp', 'FieldGoalsAttempted2_home': 'FieldGoalsAttempted2_opp', 
                     'FieldGoalsMade3_home': 'FieldGoalsMade3_opp', 'FieldGoalsAttempted3_home': 'FieldGoalsAttempted3_opp', 
                     'FreeThrowsMade_home': 'FreeThrowsMade_opp', 'FreeThrowsAttempted_home': 'FreeThrowsAttempted_opp', 
                     'OffensiveRebounds_home': 'OffensiveRebounds_opp', 'DefensiveRebounds_home': 'DefensiveRebounds_opp', 
                     'TotalRebounds_home': 'TotalRebounds_opp', 'Assistances_home': 'Assistances_opp', 'Steals_home': 'Steals_opp', 
                     'Turnovers_home': 'Turnovers_opp', 'BlocksFavour_home': 'BlocksFavour_opp', 'BlocksAgainst_home': 'BlocksAgainst_opp', 
                     'FoulsCommited_home': 'FoulsCommited_opp', 'FoulsReceived_home': 'FoulsReceived_opp', 'Valuation_home': 'Valuation_opp', 
                     'Points_away': 'Points_team', 'Pace_away': 'Pace_team', 'FieldGoalsMade2_away': 'FieldGoalsMade2_team', 'FieldGoalsAttempted2_away': 'FieldGoalsAttempted2_team', 
                     'FieldGoalsMade3_away': 'FieldGoalsMade3_team', 'FieldGoalsAttempted3_away': 'FieldGoalsAttempted3_team', 
                     'FreeThrowsMade_away': 'FreeThrowsMade_team', 'FreeThrowsAttempted_away': 'FreeThrowsAttempted_team', 
                     'OffensiveRebounds_away': 'OffensiveRebounds_team', 'DefensiveRebounds_away': 'DefensiveRebounds_team', 
                     'TotalRebounds_away': 'TotalRebounds_team', 'Assistances_away': 'Assistances_team', 'Steals_away': 'Steals_team', 
                     'Turnovers_away': 'Turnovers_team', 'BlocksFavour_away': 'BlocksFavour_team', 'BlocksAgainst_away': 'BlocksAgainst_team', 
                     'FoulsCommited_away': 'FoulsCommited_team', 'FoulsReceived_away': 'FoulsReceived_team', 
                     'Valuation_away': 'Valuation_team'}, inplace=True)

result = pd.concat([home, away])
result = result.rename_axis('Team_name').reset_index()

result1 = result.groupby(['Team_name']).agg({'Points_team':'sum','Pace_team':'sum','FieldGoalsMade2_team':'sum','FieldGoalsAttempted2_team':'sum',
                                             'FieldGoalsMade3_team':'sum','FieldGoalsAttempted3_team':'sum','FreeThrowsMade_team':'sum',
                                             'FreeThrowsAttempted_team':'sum','OffensiveRebounds_team':'sum','DefensiveRebounds_team':'sum',
                                             'TotalRebounds_team':'sum','Assistances_team':'sum','Steals_team':'sum',
                                             'Turnovers_team':'sum','BlocksFavour_team':'sum','BlocksAgainst_team':'sum',
                                             'FoulsCommited_team':'sum','FoulsReceived_team':'sum','Valuation_team':'sum',
                                             'Points_opp':'sum','Pace_opp':'sum','FieldGoalsMade2_opp':'sum','FieldGoalsAttempted2_opp':'sum',
                                             'FieldGoalsMade3_opp':'sum','FieldGoalsAttempted3_opp':'sum','FreeThrowsMade_opp':'sum',
                                             'FreeThrowsAttempted_opp':'sum','OffensiveRebounds_opp':'sum','DefensiveRebounds_opp':'sum',
                                             'TotalRebounds_opp':'sum','Assistances_opp':'sum','Steals_opp':'sum','Turnovers_opp':'sum',
                                             'BlocksFavour_opp':'sum','BlocksAgainst_opp':'sum','FoulsCommited_opp':'sum',
                                             'FoulsReceived_opp':'sum','Valuation_opp':'sum'})
result2 = result1.rename_axis('Team_name').reset_index()

#Team Filter
st.sidebar.write("Select a team : ")
teamname = result2['Team_name']
teamselection = st.sidebar.selectbox('Team :',(teamname), label_visibility="collapsed")
st.sidebar.write("##")
st.sidebar.write('*Rounds available : from ',df_round['Round'].min(),' to ', df_round['Round'].max())

#VIZ 1
#Calculate Offensive Ratings
result1['ORTG'] = result1['Points_team'] / result1['Pace_team'] * 100
#Calculate defensive efficiency
result1['DRTG'] = result1['Points_opp'] / result1['Pace_opp'] * 100

#Graph
result1.sort_index()
def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=0.225)

paths = [
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/7BET-LIETKABELIS PANEVEZYS.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/BUDUCNOST VOLI PODGORICA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/CEDEVITA OLIMPIJA LJUBLJANA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/DOLOMITI ENERGIA TRENTO.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/FRUTTI EXTRA BURSASPOR.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/GERMANI BRESCIA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/GRAN CANARIA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/HAPOEL TEL AVIV.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/JOVENTUT BADALONA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/LONDON LIONS.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/MINCIDELICE JL BOURG EN BRESSE.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/PARIS BASKETBALL.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/PROMETEY SLOBOZHANSKE.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/PROMITHEAS PATRAS.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/RATIOPHARM ULM.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/SLASK WROCLAW.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/TURK TELEKOM ANKARA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/U-BT CLUJ-NAPOCA.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/UMANA REYER VENICE.png',
    '/Users/sissigarduno/Desktop/DATA/IMAGES/EUROCUP/LOGOS/V2/VEOLIA TOWERS HAMBURG.png'
]
    
x = result1['ORTG']
y = result1['DRTG']

fig, ax = plt.subplots()
ax.scatter(x, y, c="white") 
# Move left y-axis and bottom x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Others
ax.invert_yaxis()
ax.tick_params(axis='both', which='major', labelsize=8)
ax.text(108, 129.8, 'Defensive Rating', rotation = 'vertical', fontsize = 'xx-small')
ax.text(85.2, 107, 'Offensive Rating', rotation = 'horizontal', fontsize = 'xx-small')
ax.text(90.5, 93.3, 'Positive Teams', rotation = -35, fontsize = 'xx-small', c="lightgrey", weight='bold')
ax.text(90, 94.4, 'Negative Teams', rotation = -35, fontsize = 'xx-small', c="lightgrey", weight='bold')
ax.text(96, 92, 'Net +5', rotation = -35, fontsize = 'xx-small', c="lightgrey")
ax.text(100.8, 92, 'Net +10', rotation = -35, fontsize = 'xx-small', c="lightgrey")
ax.text(105.8, 92, 'Net +15', rotation = -35, fontsize = 'xx-small', c="lightgrey")
ax.text(90, 95.8, 'Net -5', rotation = -35, fontsize = 'xx-small', c="lightgrey")
ax.text(90, 101, 'Net -10', rotation = -35, fontsize = 'xx-small', c="lightgrey")
ax.text(91, 107, 'Net -15', rotation = -35, fontsize = 'xx-small', c="lightgrey")
ax.plot([85, 130], [85, 130], ls="--", c="lightgrey", linewidth=1)
ax.plot([85, 125], [90, 130], ls="--", c="lightgrey", linewidth=0.5)
ax.plot([85, 120], [95, 130], ls="--", c="lightgrey", linewidth=0.5)
ax.plot([85, 115], [100, 130], ls="--", c="lightgrey", linewidth=0.5)
ax.plot([90, 130], [85, 125], ls="--", c="lightgrey", linewidth=0.5)
ax.plot([95, 130], [85, 120], ls="--", c="lightgrey", linewidth=0.5)
ax.plot([100, 130], [85, 115], ls="--", c="lightgrey", linewidth=0.5)
ax.set(xlim=(85, 130), ylim=(130, 85))

for x0, y0, path in zip(x, y,paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)
    
# VIZ 2
params = ['ORTG', 'eFG%', 'TS%', 'AST/TO', 'TOV%', 'OREB%', 'DRTG', 'DREB%', 'BLK%', 'STL%', 'NetRTG', 'Pace_team', 'FTA_rate', 'Valuation_team']
#Calculations
result2['ORTG'] = result2['Points_team'] / result2['Pace_team'] * 100
result2['DRTG'] = result2['Points_opp'] / result2['Pace_opp'] * 100
result2['eFG%'] = ((result2['FieldGoalsMade2_team'] + result2['FieldGoalsMade3_team']*1.5) / (result2['FieldGoalsAttempted2_team'] + result2['FieldGoalsAttempted3_team'])) * 100
result2['TS%'] = (result2['Points_team'] / (2*((result2['FieldGoalsAttempted2_team'] + result2['FieldGoalsAttempted3_team']) + 0.44*result2['FreeThrowsAttempted_team']))) * 100
result2['AST/TO'] = result2['Assistances_team'] / result2['Turnovers_team']
result2['TOV%'] = (result2['Turnovers_team'] / result2['Pace_team']) * 100
result2['OREB%'] = (result2['OffensiveRebounds_team'] / (result2['OffensiveRebounds_team'] + result2['DefensiveRebounds_opp'])) * 100
result2['DREB%'] = (result2['DefensiveRebounds_team'] / (result2['DefensiveRebounds_team'] + result2['OffensiveRebounds_opp'])) * 100
result2['BLK%'] = (result2['BlocksFavour_team'] / result2['FieldGoalsAttempted2_opp']) * 100
result2['STL%'] = (result2['Steals_team'] / result2['Pace_opp']) * 100
result2['NetRTG'] = result2['ORTG'] - result2['DRTG']
result2['FTA_rate'] = (result2['FreeThrowsAttempted_team'] / (result2['FieldGoalsAttempted2_team'] + result2['FieldGoalsAttempted3_team'])) * 100

pizza = (result2[['Team_name', 'ORTG', 'eFG%', 'TS%', 'AST/TO', 'TOV%', 'OREB%', 'DRTG', 'DREB%', 'BLK%', 'STL%', 'NetRTG', 'Pace_team', 'FTA_rate', 'Valuation_team']])
pizza = pizza.fillna(0)

team = pizza.loc[pizza['Team_name'] == teamselection].reset_index()
team = list(team.loc[0])
team = team[2:]

values = []
for x in range(len(params)):
    values.append(math.floor(stats.percentileofscore(pizza[params[x]], team[x])))
    
 # color for the slices and text
slice_colors = ["#1A78CF"] * 3 + ["#FF9300"] * 3 + ["#D70232"] * 4 + ["grey"] * 4
box_colors = ["white"] * 14
box_font_colors = ["#252528"] * 14
text_colors = ["black"] * 14

# instantiate PyPizza class
baker = PyPizza(
    params=params,  # list of parameters
    background_color="#FFFFFF",  # background color
    straight_line_color="white",  # color for straight lines
    straight_line_lw=1,  # linewidth for straight lines
    last_circle_lw=0,  # linewidth of last circle
    other_circle_lw=0,  # linewidth for other circles
    inner_circle_size=0  # size of inner circle
)

# plot pizza
fig1, ax = baker.make_pizza(
    values,  # list of values
    figsize=(8, 8.5),  # adjust figsize according to your need
    color_blank_space="same",  # use same color to fill blank space
    slice_colors=slice_colors,  # color for individual slices
    value_colors=box_colors,  # color for the value-text
    value_bck_colors=box_font_colors,  # color for the blank spaces
    blank_alpha=0.4,# alpha for blank-space colors
    kwargs_slices=dict(
        edgecolor="#212124", zorder=2, linewidth=1
    ),  # values to be used when plotting slices
    kwargs_params=dict(
        color="black", fontsize=11,
        va="center"
    ),  # values to be used when adding parameter
    kwargs_values=dict(
        color="black", fontsize=11,
        zorder=3,
        bbox=dict(
            edgecolor="black", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )  # values to be used when adding parameter-values
)

#Legend
# add text
fig1.text(0.5, 0.97, teamselection, size=20, color="#000000", ha="center")
fig1.text(
    0.1, 0.925, "Attacking", size=10, color="#000000"
)
fig1.text(
    0.1, 0.900, "Possession", size=10, color="#000000"
)
fig1.text(
    0.1, 0.875, "Defending", size=10, color="#000000"
)
fig1.text(
    0.1, 0.850, "Other", size=10, color="#000000"
)

# add rectangles
fig1.patches.extend([
    plt.Rectangle(
        (0.06, 0.922), 0.025, 0.021, fill=True, color="#1a78cf",
        transform=fig1.transFigure, figure=fig1
    ),
    plt.Rectangle(
        (0.06, 0.897), 0.025, 0.021, fill=True, color="#ff9300",
        transform=fig1.transFigure, figure=fig1
    ),
    plt.Rectangle(
        (0.06, 0.872), 0.025, 0.021, fill=True, color="#d70232",
        transform=fig1.transFigure, figure=fig1
    ),
    plt.Rectangle(
        (0.06, 0.847), 0.025, 0.021, fill=True, color="grey",
        transform=fig1.transFigure, figure=fig1
    ),
])

# VIZ 4
def draw_court(ax=None, color='black', lw=1, outer_lines=True):
    """
    FIBA basketball court dimensions:
    https://www.msfsports.com.au/basketball-court-dimensions/
    It seems like the Euroleauge API returns the shooting positions
    in resolution of 1cm x 1cm.
    """
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 45.72cm so it has a radius 45.72/2 cms
    hoop = Circle((0, 0), radius=45.72 / 2, linewidth=lw, color=color,
                  fill=False)

    # Create backboard
    backboard = Rectangle((-90, -157.5 + 120), 180, -1, linewidth=lw,
                          color=color)

    # The paint
    # Create the outer box of the paint
    outer_box = Rectangle((-490 / 2, -157.5), 490, 580, linewidth=lw,
                          color=color, fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-360 / 2, -157.5), 360, 580, linewidth=lw,
                          color=color, fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 580 - 157.5), 360, 360, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 580 - 157.5), 360, 360, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 2 * 125, 2 * 125, theta1=0, theta2=180,
                     linewidth=lw, color=color)

    # Three point line
    # Create the side 3pt lines
    corner_three_a = Rectangle((-750 + 90, -157.5), 0, 305, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((750 - 90, -157.5), 0, 305, linewidth=lw,
                               color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 2 * 675, 2 * 675, theta1=12, theta2=167.5,
                    linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box,
                      restricted, top_free_throw, bottom_free_throw,
                      corner_three_a, corner_three_b, three_arc]

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def plot_scatter(made, miss, title=None):
    """
    Scatter plot of made and missed shots
    """
    plt.figure()
    draw_court()
    plt.plot(miss['COORD_X'], miss['COORD_Y'], 'o', color='red', label='Missed', alpha=0.6, markeredgecolor='black', markersize=4)
    plt.plot(made['COORD_X'], made['COORD_Y'], 'o', label='Made', color='green', alpha=0.6, markeredgecolor='black', markersize=4)
    plt.legend(fontsize="x-small", frameon=False, alignment="left")
    plt.xlim([-800, 800])
    plt.ylim([-155, 1300])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return

# split the home and away teams, their made and missed shots
df_shots = df_shots[df_shots['Round'].between(begin, end)]
df_shots['TEAM'] = df_shots['TEAM'].str.strip()  # team id contains trailing white space
df_shots['ID_PLAYER'] = df_shots['ID_PLAYER'].str.strip()  # player id contains trailing white space
home_df = df_shots[df_shots['TEAM'] == teamselection]
fg_made_home_df = home_df[home_df['ID_ACTION'].isin(['2FGM', '3FGM'])]
fg_miss_home_df = home_df[home_df['ID_ACTION'].isin(['2FGA', '3FGA'])]

# scatter shot chart of PAOs
fig2 = plot_scatter(fg_made_home_df, fg_miss_home_df, title=teamselection)

#Stats avancées
result3 = (result2[['Team_name', 'ORTG', 'DRTG', 'NetRTG', 'eFG%', 'TS%', 'AST/TO', 'OREB%', 'DREB%', 'TOV%', 'BLK%', 'STL%', 'FTA_rate', 'Pace_team', 'Valuation_team']])
result3 = result3.set_index('Team_name')

#Display
row1_col1, row1_col2 = st.columns(2)
    
with row1_col1:
    st.header('Efficiency Landscape')
    st.pyplot(fig)
    st.header('Percentiles')
    st.pyplot(fig1)
    
with row1_col2:
    st.header('Advanced Stats')
    st.dataframe(result3)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.header('Shot Chart')
    st.pyplot(fig2)
    
#Chose what I wanna see in database (Brut, Moy comme LNB.FR / Advanced, Basic stats comme WNBA.COM)
#Pizza Plot, DRTG dans l'autre sens. Avoir un high DRTG n'est pas bien. Voir si c'est le cas d'autre stats.
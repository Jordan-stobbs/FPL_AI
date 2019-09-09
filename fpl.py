import sklearn
from sklearn.model_selection import train_test_split
import sqlite3
from sqlite3 import Error
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
from pulp import *
import re

"""
TODO see reddit post
TODO Deal with new teams?

Have tried seperating into defenders, keepers etc, did worse
"""


def predict(seasonAllRequiredFields, seasonAll, element_type, lastSeasonRequiredFields,lastSeason):
    
    print(list(seasonAllRequiredFields.columns.values))

    if element_type != 1 and element_type != 5:
        seasonAllRequiredFields = seasonAllRequiredFields.drop(['penalties_saved','saves'], axis ='columns')
        lastSeasonRequiredFields = lastSeasonRequiredFields.drop(['penalties_saved','saves'], axis ='columns')
    else:
        seasonAllRequiredFields = seasonAllRequiredFields.drop(['goals_scored'], axis ='columns')
        lastSeasonRequiredFields = lastSeasonRequiredFields.drop(['goals_scored'], axis ='columns')
    if(element_type == 4):
        seasonAllRequiredFields = seasonAllRequiredFields.drop(['goals_conceded','clean_sheets'], axis ='columns')
        lastSeasonRequiredFields = lastSeasonRequiredFields.drop(['goals_conceded','clean_sheets'], axis ='columns')
    
    if element_type == 5:
        season1718md = seasonAll
    else:
        season1718md = seasonAll[seasonAll.element_type==element_type]
        seasonAllRequiredFields = seasonAllRequiredFields[seasonAllRequiredFields.element_type==element_type]
    seasonAllRequiredFields.drop(['element_type'], axis='columns')
    season1718md.drop(['element_type'], axis='columns')
    x = seasonAllRequiredFields.iloc[:,0:-1]
    y = seasonAllRequiredFields.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size=0.8, random_state=12421)


    md = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=100000).fit(X_train, y_train)
    prediction = md.predict(X_test)
        
    score = round(md.score(X_test, y_test), 4)
    print(score)
    prediction = md.predict(x)
    prediction2 = md.predict(lastSeasonRequiredFields)
    season1718md['prediction'] = prediction
    lastSeason['prediction'] = prediction2
    lastSeason.to_csv('predictions.csv')
    return lastSeason
"""
Code adapted from
https://medium.com/ml-everything/using-python-and-linear-programming-to-optimize-fantasy-football-picks-dc9d1229db81
"""
def summary(prob):
    div = '---------------------------------------\n'
    print("Variables:\n")
    score = str(prob.objective)
    constraints = [str(const) for const in prob.constraints.values()]
    for v in prob.variables():
        score = score.replace(v.name, str(v.varValue))
        constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
        if v.varValue != 0:
            print(v.name, "=", v.varValue)
    print(div)
    print("Constraints:")
    for constraint in constraints:
        constraint_pretty = " + ".join(re.findall("[0-9\.]*\*1.0", constraint))
        if constraint_pretty != "":
            print("{} = {}".format(constraint_pretty, eval(constraint_pretty)))
    print(div)
    print("Score:")
    score_pretty = " + ".join(re.findall("[0-9\.]+\*1.0", score))
    print("{} = {}".format(str(score_pretty), eval(score)))

"""
Code adapted from
https://medium.com/ml-everything/using-python-and-linear-programming-to-optimize-fantasy-football-picks-dc9d1229db81
"""
def optimalTeam(df):
    """
    currently picking players from liverpool and man city only (probably for good reason, need to pass teams in)
    """
    costs = {}
    points = {}
    teams = {}
    COST_CAP = 835
    positionsAvailable = {
        1:1,
        2:4,
        3:5,
        4:1
    }
    teamsAvailable = {
        1:3,
        2:3,
        3:3,
        4:3,
        5:3,
        6:3,
        7:3,
        8:3,
        9:3,
        10:3,
        11:3,
        12:3,
        13:3,
        14:3,
        15:3,
        16:3,
        17:3,
        18:3,
        19:3,
        20:3
    }
    df = df.drop(df[df.minutes<=1000].index)
    print(df)
    dfCurrent = pd.read_csv("data/2019-20/players_raw.csv")
    dfCurrent = dfCurrent[['first_name','second_name','now_cost','element_type']]
    dfUse = pd.merge(df, dfCurrent,on=['first_name','second_name'],suffixes = ['_old','_new'], how = 'inner' )
    # get points, position, cost, name
    pointsCol = dfUse['prediction']
    costCol = dfUse['now_cost_new']
    positionCol = dfUse['element_type_new']
    fnames = dfUse['first_name']
    lnames = dfUse['second_name']

    players = dfUse[["prediction","now_cost_new","element_type_new","first_name","second_name",'team']]
    displayNames = []
    for fName, lName in zip(dfUse['first_name'],dfUse['second_name']):
        displayNames.append(fName + " " + lName)
    players["displayNames"]=displayNames

    for pos in players.element_type_new.unique():
        # for each unique position
        new_available_pos = players[players.element_type_new == pos]
        # new variable having only peopel of that position
        debug = new_available_pos[["displayNames","now_cost_new"]].set_index("displayNames") #two rows display name is title
        debug2 = debug.to_dict() #translates to K:name V:cost
        debug3 = debug2.values() # everything in the dictionary
        debug4 = list(debug3)[0] #just lookks like the dictionary 
        cost = list(new_available_pos[["displayNames","now_cost_new"]].set_index("displayNames").to_dict().values())[0]
        point = list(new_available_pos[["displayNames","prediction"]].set_index("displayNames").to_dict().values())[0]
        #team = list(new_available_pos[["displayNames","team"]].set_index("displayNames").to_dict().values())[0]
        costs[pos] = cost
        points[pos] = point
        #teams[pos] = team
    _vars = {k: LpVariable.dict(str(k), v, cat="Binary") for k, v in points.items()} #k number v array 
    #name, team
    # for tem in players.team.unique():
    #     available_team = players[players.team == tem]
    #     team =list(available_team[["displayNames","team"]].set_index("displayNames").to_dict().values())[0] 
    #     teams[tem] = team
    # _teams = {k: LpVariable.dict(str(k), v, cat="Binary") for k, v in teams.items()} #k name v iterator
    #learn how to use lpVars

    prob = LpProblem("Fantasy", LpMaximize)
    rewards = []
    costs2 = []
    position_constraints = []# Setting up the reward
    for k, v in _vars.items():
        costs2 += lpSum([costs[k][i] * _vars[k][i] for i in v])
        rewards += lpSum([points[k][i] * _vars[k][i] for i in v]) #v is name to nameso its poitns * new name
        prob += lpSum([_vars[k][i] for i in v]) == positionsAvailable[k]
    
    # for k,v in _teams.items():
    #     for name in _teams[k]:
    #         teamPlayer = players[players.displayNames==name]
    #         pos = teamPlayer["element_type_new"][0]
    #         newStr = _teams[k][name][2:]
    #         newStr = pos+"_"+newStr
    #         _teams[k][name] = newStr

    # for k, v in _teams.items():
        
    #     print(k)
    #     print(v)
    #     prob += lpSum(_teams[k][i] for i in v) <= teamsAvailable[k]
        #for team in teams
         # for name in team
         # for nam in name 
         # rename nam to whatever it is above probably do this by removing first 2 characers searching position usiong players[players.displayNames =newNam] then newPlayers[team] thens tring coincat team _ new nam 
    
    #prob += lpSum([teams[]])
    prob += lpSum(rewards)
    prob += lpSum(costs2) <= COST_CAP

    prob.solve()
    summary(prob)

def main():
    #change to for loop at some point
    season16 = pd.read_csv("data/2016-17/players_raw.csv")
    season17 = pd.read_csv("data/2017-18/players_raw.csv")
    season18 = pd.read_csv("data/2018-19/players_raw.csv")

    season18Short = season18[['first_name','second_name','total_points']]
    season17Short = season17[['first_name','second_name','total_points']]

    season1617 = pd.merge(season16, season17Short,on=['first_name','second_name'],suffixes = ['_old','_new'], how = 'inner' )
    season1718 = pd.merge(season17, season18Short,on=['first_name','second_name'],suffixes = ['_old','_new'], how = 'inner' )

    seasonAll = season1617.append(season1718)
    print(seasonAll.head())

    print(seasonAll)
    seasonAllRequiredFields = seasonAll[['assists','bonus','bps','clean_sheets','dreamteam_count','element_type','goals_conceded','goals_scored','minutes','points_per_game','red_cards',
    'selected_by_percent','total_points_old','yellow_cards','penalties_saved','saves','total_points_new']]
    print(seasonAllRequiredFields)
    lastSeason = season18[['assists','bonus','bps','clean_sheets','dreamteam_count','element_type','goals_conceded','goals_scored','minutes','points_per_game','red_cards',
    'selected_by_percent','total_points','yellow_cards','penalties_saved','saves']]
    seasonAllRequiredFields['goals_scored'] = (seasonAllRequiredFields['goals_scored']/seasonAllRequiredFields['minutes']).fillna(0)
    seasonAllRequiredFields['goals_conceded'] = (seasonAllRequiredFields['goals_conceded']/seasonAllRequiredFields['minutes']).fillna(1)
    seasonAllRequiredFields['goals_scored'] .replace([np.inf, -np.inf],0)
    seasonAllRequiredFields['goals_conceded'] .replace([np.inf, -np.inf],70)

    # gk = predict(seasonAllRequiredFields, seasonAll,1,lastSeason,season18)
    # md = predict(seasonAllRequiredFields, seasonAll,3,lastSeason,season18)
    # df2 = predict(seasonAllRequiredFields, seasonAll,2,lastSeason,season18)
    # st = predict(seasonAllRequiredFields, seasonAll,4,lastSeason,season18)
    # st['goals_conceeded'] = 0
    # st['clean_sheets'] = 0
    # all2 = md.append(df2.append(st))
    # all2['penalties_saved'] = 0
    # all2['saves'] = 0
    # gk['goals_scored'] = 0
    # all2 = all2.append(gk)
    all2 = predict(seasonAllRequiredFields, seasonAll,5,lastSeason,season18)
    print(list(all2.columns.values))
    optimalTeam(all2)
    all2.to_csv('testall.csv')

if(__name__ == "__main__"):
    main()
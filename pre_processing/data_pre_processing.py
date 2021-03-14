import csv
import numpy as np
import glob
import pandas as pd


absolute_path_to_data = "C:\\Users\\Sara\\Desktop\\master\\atp_matches_prediction\\data"

def readATPMatches(dirname):
    """Reads ATP matches"""
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for file in allFiles:
        df = pd.read_csv(file,
                         index_col=None,
                         header=0)
        container.append(df)
    matches = pd.concat(container)
    return matches


def readFMatches(dirname):
    """Reads ITF future matches"""
    allFiles = glob.glob(dirname + "/atp_matches_futures_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for file in allFiles:
        df = pd.read_csv(file,
                         index_col=None,
                         header=0)
        container.append(df)
    matches = pd.concat(container)
    return matches


def readChall_QATPMatches(dirname):
    """reads Challenger level + ATP Q matches"""
    allFiles = glob.glob(dirname + "/atp_matches_qual_chall_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for file in allFiles:
        df = pd.read_csv(file,
                         index_col=None,
                         header=0)
        container.append(df)
    matches = pd.concat(container)
    return matches


def checkTourneyId(matches):
    """Sanity checking tourney_id"""
    numOfNan = matches[matches['tourney_id'].isnull()].shape[0]
    print("Sanity checking tourney_id")
    print(numOfNan)
    matches.dropna(subset=['tourney_id'], inplace=True)
    return matches


def checkTourneyName(matches):
    """Sanity checking tourney_name"""
    numOfNan = matches[matches['tourney_name'].isnull()].shape[0]
    print("Sanity checking tourney_name")
    print(numOfNan)
    matches.dropna(subset=['tourney_name'], inplace=True)
    return matches


def checkSurface(matches):
    """Sanity checking surface"""
    numOfNan = matches[matches['surface'].isnull()].shape[0]
    print("Sanity checking surface")
    print(numOfNan)
    matches.dropna(subset=['surface'], inplace=True)
    return matches


def getSurfaceDict(matches):
    """Get surfaces"""
    surfaces = matches.groupby(['surface']).size().reset_index(
        name="surface_count").sort_values(by='surface_count', ascending=False)
    surfaces_dict = dict(surfaces.values)
    print("Surfaces")
    print(surfaces_dict)
    return surfaces_dict


def checkDrawSize(matches):
    """Sanity checking draw_size"""
    numOfNan = matches[matches['draw_size'].isnull()].shape[0]
    print("Sanity checking draw_size")
    print(numOfNan)
    matches.dropna(subset=['draw_size'], inplace=True)
    return matches


def getDrawSizeDict(matches):
    """Get draw_size"""
    draw_sizes = matches.groupby(['draw_size']).size().reset_index(
        name="draw_size_count").sort_values(by='draw_size', ascending=False)
    draw_size_dict = dict(draw_sizes.values)
    print("Draw_size")
    print(draw_size_dict)
    return draw_size_dict


def checkTourneyLevel(matches):
    """Sanity checking tourney_level"""
    numOfNan = matches[matches['tourney_level'].isnull()].shape[0]
    print("Sanity checking tourney_level")
    print(numOfNan)
    matches.dropna(subset=['tourney_level'], inplace=True)
    return matches


def getTourneyLevelDict(matches):
    """Get tourney_level"""
    tourney_levels = matches.groupby(['tourney_level']).size().reset_index(
        name="tourney_level_count").sort_values(by='tourney_level_count', ascending=False)
    tourney_level_dict = dict(tourney_levels.values)
    print("Tourney_levels")
    print(tourney_level_dict)
    return tourney_level_dict


def checkPlayerId(matches):
    """Sanity checking winner_id and loser_id"""
    numOfNan = matches[matches['winner_id'].isnull() | matches['loser_id'].isnull()].shape[0]
    print("Sanity checking winner_id and loser_id")
    print(numOfNan)
    matches.dropna(subset=['winner_id'], inplace=True)
    matches.dropna(subset=['loser_id'], inplace=True)
    return matches


def checkPlayerSeed(matches):
    """Sanity checking winner_seed and loser_seed"""
    matches[["winner_seed", "loser_seed"]] = matches[["winner_seed", "loser_seed"]].apply(pd.to_numeric, errors='coerce')
    numOfNan = matches[matches['winner_seed'].isnull() | matches['loser_seed'].isnull()].shape[0]
    print("Sanity checking winner_seed and loser_seed")
    print(numOfNan)
    """Fill NaN players seed with avg value"""
    winner_seed = pd.DataFrame(matches, columns=['winner_seed'])
    winner_seed.columns = ['players_seed']
    loser_seed = pd.DataFrame(matches, columns=['loser_seed'])
    loser_seed.columns = ['players_seed']
    players_seed = winner_seed.append(loser_seed)
    avg_player_seed = str(round(players_seed["players_seed"].mean(skipna=True)))

    matches['winner_seed'] = matches['winner_seed'].fillna(avg_player_seed)
    matches['loser_seed'] = matches['loser_seed'].fillna(avg_player_seed)

    """Fill NaN players seed with max value"""
    # max_player_seed = int(players_seed["players_seed"].max(skipna=True))
    #
    # matches['winner_seed'] = matches['winner_seed'].fillna(max_player_seed)
    # matches['loser_seed'] = matches['loser_seed'].fillna(max_player_seed)

    return matches


def getPlayersEntryDict(matches):
    """Get players_entry"""
    matches['winner_entry'] = matches['winner_entry'].fillna(" ")
    matches['loser_entry'] = matches['loser_entry'].fillna(" ")
    winner_entries = pd.DataFrame(matches, columns=['winner_entry'])
    winner_entries.columns = ['players_entry']
    loser_entries = pd.DataFrame(matches, columns=['loser_entry'])
    loser_entries.columns = ['players_entry']
    players_entries = winner_entries.append(loser_entries)
    players_entries = players_entries.groupby(['players_entry']).size().reset_index(
        name="players_entry_count").sort_values(by='players_entry_count', ascending=False)
    players_entry_dict = dict(players_entries.values)
    print("Players_entry")
    print(players_entry_dict)
    return players_entry_dict


def checkPlayersHand(matches):
    """Sanity checking winner_hand and loser_hand"""
    numOfNan = matches[matches['winner_hand'].isnull() | matches['loser_hand'].isnull()].shape[0]
    print("Sanity checking winner_hand and loser_hand")
    print(numOfNan)
    matches.dropna(subset=['winner_hand'], inplace=True)
    matches.dropna(subset=['loser_hand'], inplace=True)
    matches['winner_hand'].apply(lambda x: x.upper())
    matches['loser_hand'].apply(lambda x: x.upper())
    matches['winner_hand'] = np.where(matches['winner_hand'] == 'L', 1, matches['winner_hand'])
    matches['winner_hand'] = np.where(matches['winner_hand'] == 'R', 0, matches['winner_hand'])
    matches['loser_hand'] = np.where(matches['loser_hand'] == 'L', 1, matches['loser_hand'])
    matches['loser_hand'] = np.where(matches['loser_hand'] == 'R', 0, matches['loser_hand'])
    return matches


def checkPlayerHeight(matches):
    """Sanity checking winner_ht and loser_ht"""
    numOfNan = matches[matches['winner_ht'].isnull() | matches['loser_ht'].isnull()].shape[0]
    print("Sanity checking winner_ht and loser_ht")
    print(numOfNan)
    """Fill NaN players height with -1"""
    matches['winner_ht'] = matches['winner_ht'].fillna(-1)
    matches['loser_ht'] = matches['loser_ht'].fillna(-1)

    return matches


def checkPlayerAge(matches):
    """Sanity checking winner_age and loser_age"""
    numOfNan = matches[matches['winner_age'].isnull() | matches['loser_age'].isnull()].shape[0]
    print("Sanity checking winner_age and loser_age")
    print(numOfNan)
    """Fill NaN players age with avg value"""
    winner_age = pd.DataFrame(matches, columns=['winner_age'])
    winner_age.columns = ['players_age']
    loser_age = pd.DataFrame(matches, columns=['loser_age'])
    loser_age.columns = ['players_age']
    players_age = winner_age.append(loser_age)
    avg_player_age = str(round(players_age["players_age"].mean(skipna=True)))

    matches['winner_age'] = matches['winner_age'].fillna(avg_player_age)
    matches['loser_age'] = matches['loser_age'].fillna(avg_player_age)

    return matches


def checkPlayerRank(matches):
    """Sanity checking winner_rank and loser_rank"""
    numOfNan = matches[matches['winner_rank'].isnull() | matches['loser_rank'].isnull()].shape[0]
    print("Sanity checking winner_rank and loser_rank")
    print(numOfNan)
    """Fill NaN players rank with 2000 which represents really high rank"""
    matches['winner_rank'] = matches['winner_rank'].fillna(2000)
    matches['loser_rank'] = matches['loser_rank'].fillna(2000)

    return matches


def checkPlayerRankPoints(matches):
    """Sanity checking winner_rank_points and loser_rank_points"""
    numOfNan = matches[matches['winner_rank_points'].isnull() | matches['loser_rank_points'].isnull()].shape[0]
    print("Sanity checking winner_rank_points and loser_rank_points")
    print(numOfNan)
    """Fill NaN players rank points with 0"""
    matches['winner_rank_points'] = matches['winner_rank_points'].fillna(0)
    matches['loser_rank_points'] = matches['loser_rank_points'].fillna(0)

    return matches


def checkScore(matches):
    """Sanity checking score"""
    #matches['score'].apply(lambda x: x.lower())
    numOfNan = matches[matches['score'].isnull() | matches['score'] == "w/o"].shape[0]
    print("Sanity checking score")
    print(numOfNan)
    matches.dropna(subset=['score'], inplace=True)
    matches = matches[matches.score.str.lower() != "w/o"]

    return matches


def checkBestOf(matches):
    """Sanity checking best_of"""
    numOfNan = matches[matches['best_of'].isnull()].shape[0]
    print("Sanity checking best_of")
    print(numOfNan)
    """Fill Nan best_of with 5 for Grand slams and with 3 for others"""
    matches['best_of'] = np.where((matches['tourney_level'] == 'G') & (matches['best_of'].isnull()), 5, matches['best_of'])
    matches['best_of'] = np.where(matches['best_of'].isnull(), 3, matches['best_of'])

    return matches


def getRoundsDict(matches):
    """Get rounds"""
    rounds = matches.groupby(['round']).size().reset_index(
        name="rounds_count").sort_values(by='round', ascending=False)
    rounds_dict = dict(rounds.values)
    print("Rounds")
    print(rounds_dict)
    return rounds_dict


def checkMinutes(matches):
    """Sanity checking minutes"""
    numOfNan = matches[matches['minutes'].isnull()].shape[0]
    print("Sanity checking minutes")
    print(numOfNan)
    matches.dropna(subset=['minutes'], inplace=True)
    #think about filling nan values with avg
    return matches


def checkMatchStatistic(matches):
    """Sanity checking match statistic"""
    numOfNan = matches[matches['w_ace'].isnull() | matches['w_df'].isnull() | matches['w_svpt'].isnull() |
                       matches['w_1stIn'].isnull() | matches['w_1stWon'].isnull() | matches['w_2ndWon'].isnull() |
                       matches['w_SvGms'].isnull() | matches['w_bpSaved'].isnull() | matches['w_bpFaced'].isnull()].shape[0]
    numOfNan += matches[matches['l_ace'].isnull() | matches['l_df'].isnull() | matches['l_svpt'].isnull() |
                        matches['l_1stIn'].isnull() | matches['l_1stWon'].isnull() | matches['l_2ndWon'].isnull() |
                        matches['l_SvGms'].isnull() | matches['l_bpSaved'].isnull() | matches['l_bpFaced'].isnull()].shape[0]
    print("Sanity checking match statistic")
    print(numOfNan)

    matches.dropna(
        subset=['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced'],
        inplace=True)

    matches.dropna(
        subset=['l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'],
        inplace=True)

    return matches


def dropDavisCup(matches):
    """Delete Davis Cup matches"""
    numOfDavisCup = matches[matches['tourney_level'] == 'D'].shape[0]
    print("Delete Davis Cup matches")
    print(numOfDavisCup)
    matches = matches[matches.tourney_level != 'D']
    return matches


def getCleanData():
    atp_matches = readATPMatches(absolute_path_to_data)
    qa_matches = readChall_QATPMatches(absolute_path_to_data)
    f_matches = readFMatches(absolute_path_to_data)
    matches = pd.concat([atp_matches, qa_matches, f_matches])
    print(matches.shape[0])
    matches = checkTourneyId(matches)
    matches = checkTourneyName(matches)
    matches = checkSurface(matches)
    matches = checkDrawSize(matches)
    matches = checkTourneyLevel(matches)
    matches = dropDavisCup(matches)
    matches = checkPlayerId(matches)
    matches = checkPlayerSeed(matches)
    matches = checkPlayersHand(matches)
    matches = checkPlayerHeight(matches)
    matches = checkPlayerAge(matches)
    matches = checkPlayerRank(matches)
    matches = checkPlayerRankPoints(matches)
    matches = checkScore(matches)
    matches = checkBestOf(matches)
    matches = checkMinutes(matches)
    matches = checkMatchStatistic(matches)
    print("Number of matches")
    print(matches.shape[0])
    matches.to_csv("atp_matches_all.csv", index=False, encoding="utf-8-sig", header=True)


getCleanData()

from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

import os
decay_type = 1
half_life = 240
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
absolute_path_to_data = "C:\\Users\\Sara\\Desktop\\master\\atp_matches_prediction\\atp_matches_all.csv"
surfaces = {'Carpet': 1, 'Grass': 2, 'Clay': 3, 'Hard': 4}
df_atp = pd.read_csv(absolute_path_to_data)

z = 0
player_stats_tournament_sum = [dict() for z in range(12)]
player_stats_tournament_count = [dict() for z in range(12)]
player_stats_surface_sum = [dict() for z in range(12)]
player_stats_surface_count = [dict() for z in range(12)]
player_stats_overall_sum = [dict() for z in range(12)]
player_stats_overall_count = [dict() for z in range(12)]
tournament_form_won = dict()
tournament_form_total = dict()
overall_form = dict()
surface_form = dict()
recent_form = dict()
matches_won_lost_surface = dict()
matches_won_lost = dict()
head_to_head = dict()
head_to_head_surface = dict()
common_head_to_head = dict()
numOfRows = df_atp.shape[0]
player_elo = dict()
player_tournament_elo = dict()
player_surface_elo = dict()
numOfFeatures = int(23)
matches = np.zeros((2*numOfRows, numOfFeatures))


rolling_cols = [
    # 'player_overall_win_on_serve_per',
    # 'player_avg_return_per',
    # 'player_clutch_factor',
    # 'player_completeness',
    # 'player_serve_advantage',
    # 'player_game_win_ratio',
    # 'player_1stPer',
    # 'player_1stWonPer',
    # 'player_2ndWonPer',
    # 'player_bpSbpF',
    # 'player_acesVsDf',
    # 'player_df',
    # 'player_point_win_ratio',
]

surface_cols = [
    'player_overall_win_on_serve_per',
    'player_avg_return_per',
    'player_clutch_factor',
    'player_completeness',
    # 'player_serve_advantage',
    # 'player_game_win_ratio',
    'player_1stPer',
    'player_1stWonPer',
    'player_2ndWonPer',
    'player_bpSbpF',
    'player_acesVsDf',
    # 'player_df',
    # 'player_point_win_ratio',
]


feature_cols = [
    'player_age',
    'player_ht',
    'player_overall_win_on_serve_per',
    'player_avg_return_per',
    'player_clutch_factor',
    'player_completeness',
    # 'player_game_win_ratio',
    # 'player_1stPer',
    # 'player_1stWonPer',
    # 'player_2ndWonPer',
    'player_bpSbpF',
    'player_acesVsDf',
    # 'player_point_win_ratio',
    'player_serve_advantage',
    'player_rank',
    'player_log_rank',
    'player_rank_points',
    'surface_elo',
    '538elo',
    'per_of_matches_won_12_months',
    'total_matches_12_months',
    'tournament_form',
    'headToHead'
]


def data_cleaning(df):
    # Reorder columns
    df = df[['tourney_id', 'tourney_date', 'tourney_name', 'surface', 'tourney_level', 'match_num', 'winner_id',
             'winner_name',
             'winner_rank', 'winner_rank_points', 'winner_age', 'winner_ht', 'loser_id', 'loser_name',
             'loser_rank', 'loser_rank_points', 'loser_age', 'loser_ht', 'score', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved',
             'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']]

    # Renaming columns
    new_cols = [
        'tourney_id', 'tourney_date', 'tourney_name', 'surface', 'tourney_level', 'match_num', 'winner_id',
        'winner_name',
        'winner_rank', 'winner_rank_points', 'winner_age', 'winner_ht', 'loser_id', 'loser_name',
        'loser_rank', 'loser_rank_points', 'loser_age', 'loser_ht', 'score',
        'winner_ace', 'winner_df', 'winner_svpt', 'winner_1stIn', 'winner_1stWon', 'winner_2ndWon', 'winner_SvGms',
        'winner_bpSaved', 'winner_bpFaced', 'loser_ace', 'loser_df', 'loser_svpt', 'loser_1stIn', 'loser_1stWon',
        'loser_2ndWon', 'loser_SvGms', 'loser_bpSaved', 'loser_bpFaced']

    df.columns = new_cols
    # Parsing scores
    scores = df.loc[:, 'score'].str.split(' ')
    scores = scores.fillna(0)
    loser_total_games = []
    winner_total_games = []

    for index, value in scores.items():
        loser_game_score = 0
        winner_game_score = 0
        try:
            if value == 0 or value == ['W/O']:
                loser_total_games.append(loser_game_score)
                winner_total_games.append(winner_game_score)

            else:
                loser_game_score = 0
                winner_game_score = 0

                for set_ in value:
                    try:
                        text = re.match(r"(\d)\-(\d)", set_)
                        loser_game_score += int(text.group(2))
                        winner_game_score += int(text.group(1))
                    except:
                        pass
                loser_total_games.append(loser_game_score)
                winner_total_games.append(winner_game_score)
        except:
            print(index, value)

    df['tourney_date'] = df['tourney_date'].astype(str)
    df['tourney_date'] = df['tourney_date'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))

    df = df.assign(winner_total_games=pd.Series(winner_total_games))
    df = df.assign(loser_total_games=pd.Series(loser_total_games))
    df = df.assign(total_games=pd.Series(df['winner_total_games'] + df['loser_total_games']))
    df = df.assign(loser_RtGms=pd.Series(df['winner_SvGms']))
    df = df.assign(winner_RtGms=pd.Series(df['loser_SvGms']))
    df = df.assign(loser_bp=pd.Series(df['winner_bpFaced']))
    df = df.assign(winner_bp=pd.Series(df['loser_bpFaced']))

    df = df.assign(loser_bpWon=pd.Series(df['winner_bpFaced'] - df['winner_bpSaved']))
    df = df.assign(winner_bpWon=pd.Series(df['loser_bpFaced'] - df['loser_bpSaved']))

    df = df.assign(winner_1stPer=pd.Series(df['winner_1stIn']/df['winner_svpt']))
    df = df.assign(loser_1stPer=pd.Series(df['loser_1stIn']/df['loser_svpt']))
    df['winner_1stPer'].fillna(0, inplace=True)
    df['loser_1stPer'].fillna(0, inplace=True)
    df = df.assign(winner_1stWonPer=pd.Series(df['winner_1stWon'] / df['winner_1stIn']))
    df = df.assign(loser_1stWonPer=pd.Series(df['loser_1stWon'] / df['loser_1stIn']))
    df['winner_1stWonPer'].fillna(0, inplace=True)
    df['loser_1stWonPer'].fillna(0, inplace=True)
    df = df.assign(winner_2ndIn=pd.Series(df['winner_svpt'] - df['winner_1stIn'] - df['winner_df']))
    df = df.assign(loser_2ndIn=pd.Series(df['loser_svpt'] - df['loser_1stIn'] - df['loser_df']))

    df = df.assign(winner_2ndWonPer=pd.Series(df['winner_2ndWon'] / df['winner_2ndIn']))
    df = df.assign(loser_2ndWonPer=pd.Series(df['loser_2ndWon'] / df['loser_2ndIn']))
    df['winner_2ndWonPer'].fillna(0, inplace=True)
    df['loser_2ndWonPer'].fillna(0, inplace=True)
    df = df.assign(winner_1stRetWonPer=pd.Series((df['winner_1stIn'] - df['winner_1stWon']) / df['winner_1stIn']))
    df = df.assign(loser_1stRetWonPer=pd.Series((df['loser_1stIn'] - df['loser_1stWon']) / df['loser_1stIn']))
    df['winner_1stRetWonPer'].fillna(0, inplace=True)
    df['loser_1stRetWonPer'].fillna(0, inplace=True)
    df = df.assign(winner_2ndRetWonPer=pd.Series((df['winner_2ndIn'] - df['winner_2ndWon']) / df['winner_2ndIn']))
    df = df.assign(loser_2ndRetWonPer=pd.Series((df['loser_2ndIn'] - df['loser_2ndWon']) / df['loser_2ndIn']))
    df['winner_2ndRetWonPer'].fillna(0, inplace=True)
    df['loser_2ndRetWonPer'].fillna(0, inplace=True)
    # Imputing returns data so we can construct features

    df = df.assign(loser_rtpt=pd.Series(df['winner_svpt']))
    df = df.assign(winner_rtpt=pd.Series(df['loser_svpt']))
    # win on return
    df = df.assign(winner_rtptWon=pd.Series(df['loser_svpt'] - df['loser_1stWon'] - df['loser_2ndWon']))
    df = df.assign(loser_rtptWon=pd.Series(df['winner_svpt'] - df['winner_1stWon'] - df['winner_2ndWon']))
    df = df.assign(winner_svptWon=pd.Series(df['winner_1stWon'] + df['winner_2ndWon']))
    df = df.assign(loser_svptWon=pd.Series(df['loser_1stWon'] + df['loser_2ndWon']))
    df = df.assign(winner_total_points=pd.Series(df['winner_svptWon'] + df['winner_rtptWon']))
    df = df.assign(loser_total_points=pd.Series(df['loser_svptWon'] + df['loser_rtptWon']))
    df = df.assign(total_points=pd.Series(df['winner_total_points'] + df['loser_total_points']))

    # Dropping columns
    # cols_to_drop = [
    #     'tourney_level'
    # ]

    # df = df.drop(cols_to_drop, axis=1)

    return df


def convert_long(df):
    # Separating features into winner and loser so we can create rolling averages for each major tournament
    winner_cols = [col for col in df.columns if col.startswith('w')]
    loser_cols = [col for col in df.columns if col.startswith('l')]
    common_cols = [
        'tourney_id', 'tourney_name', 'tourney_level', 'tourney_date', 'surface', 'total_points', 'total_games', 'match_num']

    # Will also add opponent's rank
    df_winner = df[winner_cols + common_cols + ['loser_rank', 'loser_name', 'loser_id']]
    df_loser = df[loser_cols + common_cols + ['winner_rank', 'winner_name', 'winner_id']]

    df_winner = df_winner.assign(won=pd.Series(np.ones(numOfRows)))
    df_loser = df_loser.assign(won=pd.Series(np.zeros(numOfRows)))

    # Renaming columns
    df_winner.columns = [col.replace('winner', 'player').replace('loser', 'opponent') for col in df_winner.columns]
    df_loser.columns = df_winner.columns

    df_long = df_winner.append(df_loser, ignore_index=True)

    return df_long


def get_new_features(df):
    # Creating new features we can play around with, note that not all features may be used
    df = df.assign(player_serve_win_ratio=pd.Series((df['player_1stWon'] + df['player_2ndWon']) / (df['player_1stIn'] + df['player_2ndIn'] + df['player_df'])))
    df = df.assign(player_avg_return_per=pd.Series(df['player_rtptWon'] / df['player_rtpt']))
    df = df.assign(player_bp_per_game=pd.Series(df['player_bp'] / df['player_RtGms']))
    df = df.assign(player_bp_conversion_ratio=pd.Series(df['player_bpWon'] / df['player_bp']))

    df = df.assign(player_overall_win_on_serve_per=pd.Series(df['player_1stWonPer'] * df['player_1stPer'] + (1 - df['player_1stPer']) * df['player_2ndWonPer']))

    df = df.assign(player_completeness=pd.Series(df['player_overall_win_on_serve_per'] * df['player_avg_return_per']))
    df = df.assign(player_acesVsDf=pd.Series(df['player_ace'] / df['player_df']))
    df = df.assign(player_bpSbpF=pd.Series(df['player_bpSaved'] / df['player_bpFaced']))

    # Setting nans to zero for breakpoint conversion ratio
    df['player_serve_win_ratio'].fillna(0, inplace=True)
    df['player_avg_return_per'].fillna(0, inplace=True)
    df['player_bp_per_game'].fillna(0, inplace=True)
    df['player_bp_conversion_ratio'].fillna(0, inplace=True)
    df['player_overall_win_on_serve_per'].fillna(0, inplace=True)
    df['player_completeness'].fillna(0, inplace=True)
    df['player_acesVsDf'].fillna(0, inplace=True)
    df['player_bpSbpF'].fillna(0, inplace=True)

    df = df.assign(player_game_win_ratio=pd.Series(df['player_total_games'] / df['total_games']))
    df = df.assign(player_point_win_ratio=pd.Series(df['player_total_points'] / df['total_points']))

    df = df.assign(player_clutch_factor=pd.Series(df['player_game_win_ratio'] / df['player_point_win_ratio']))
    df = df.assign(player_log_rank=pd.Series(np.log(df['player_rank'])))
    df = df.assign(player_win_weight=pd.Series(df['won'] * np.exp(-df['opponent_rank'] / 100)))
    df['player_game_win_ratio'].fillna(0, inplace=True)
    df['player_point_win_ratio'].fillna(0, inplace=True)
    df['player_clutch_factor'].fillna(0, inplace=True)
    # Let's try weighting some of the features by the opponent's rank
    df = df.assign(player_game_win_ratio_weighted=pd.Series(df['player_game_win_ratio'] * np.exp((df['player_rank'] - df['opponent_rank']) / 500)))
    df = df.assign(player_point_win_ratio_weighted=pd.Series(df['player_point_win_ratio'] * np.exp((df['player_rank'] - df['opponent_rank']) / 500)))
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df


def setPlayersStatisticsFeature(df):

    for index in range(numOfRows):
        # matches[index][0] = str(df.loc[index, 'tourney_id'])
        # matches[index + numOfRows][0] = df.loc[index, 'tourney_id']
        # matches[index][1] = df.loc[index, 'tourney_date']
        # matches[index + numOfRows][1] = df.loc[index, 'tourney_date']

        winner_id = df.loc[index, 'player_id']
        loser_id = df.loc[index + numOfRows, 'player_id']
        winner_name = df.loc[index, 'player_name']
        loser_name = df.loc[index + numOfRows, 'player_name']
        surface = df.loc[index, 'surface']
        tournament_id = str(df.loc[index, 'tourney_id'])[5:]
        tourney_date = df.loc[index, 'tourney_date']
        tourney_level = df.loc[index, 'tourney_level']
        match_num = df.loc[index, 'match_num']
        # matches[index][2] = surface
        # matches[index + numOfRows][2] = surface
        #
        # matches[index][3] = winner_id
        # matches[index + numOfRows][3] = loser_id
        # matches[index][4] = winner_name
        # matches[index + numOfRows][4] = loser_name
        #
        # matches[index][5] = loser_id
        # matches[index + numOfRows][5] = winner_id
        # matches[index][6] = loser_name
        # matches[index + numOfRows][6] = winner_name

        feature_position = 0
        # Age feature
        matches[2 * index][feature_position] = df.loc[index, 'player_age'] - df.loc[index + numOfRows, 'player_age']
        matches[2 * index + 1][feature_position] = df.loc[index + numOfRows, 'player_age'] - df.loc[index, 'player_age']
        feature_position += 1

        # Height feature
        matches[2 * index][feature_position] = df.loc[index, 'player_ht'] - df.loc[index + numOfRows, 'player_ht']
        matches[2 * index + 1][feature_position] = df.loc[index + numOfRows, 'player_ht'] - df.loc[index, 'player_ht']
        feature_position += 1

        # Surface feature
        # matches[2 * index][feature_position] = surfaces[surface]
        # matches[2 * index + 1][feature_position] = surfaces[surface]
        # feature_position += 1

        winner_serv_per = 0
        loser_serv_per = 0
        winner_return_per = 0
        loser_return_per = 0
        for j, name in enumerate(surface_cols):
            if (winner_id, surface) not in player_stats_surface_count[j]:
                player_stats_surface_count[j][(winner_id, surface)] = 1
                player_stats_surface_sum[j][(winner_id, surface)] = df.loc[index, surface_cols[j]]
                winner_stat = 0
            else:
                winner_stat = player_stats_surface_sum[j][(winner_id, surface)] / player_stats_surface_count[j][
                    (winner_id, surface)]

                player_stats_surface_count[j][(winner_id, surface)] = player_stats_surface_count[j][
                                                                          (winner_id, surface)] + 1
                player_stats_surface_sum[j][(winner_id, surface)] = player_stats_surface_sum[j][(winner_id, surface)] + \
                                                                    df.loc[index, surface_cols[j]]

            if (loser_id, surface) not in player_stats_surface_count[j]:
                player_stats_surface_count[j][(loser_id, surface)] = 1
                player_stats_surface_sum[j][(loser_id, surface)] = df.loc[index + numOfRows, surface_cols[j]]
                loser_stat = 0
            else:
                loser_stat = player_stats_surface_sum[j][(loser_id, surface)] / player_stats_surface_count[j][
                    (loser_id, surface)]

                player_stats_surface_count[j][(loser_id, surface)] = player_stats_surface_count[j][
                                                                         (loser_id, surface)] + 1
                player_stats_surface_sum[j][(loser_id, surface)] = player_stats_surface_sum[j][(loser_id, surface)] + \
                                                                   df.loc[index + numOfRows, surface_cols[j]]
            if j == 0:
                winner_serv_per = winner_stat
                loser_serv_per = loser_stat
            if j == 1:
                winner_return_per = winner_stat
                loser_return_per = loser_stat
            matches[2 * index][feature_position] = winner_stat - loser_stat
            matches[2 * index + 1][feature_position] = loser_stat - winner_stat
            feature_position += 1
        winner_serve_adv = winner_serv_per - loser_return_per
        loser_serve_adv = loser_serv_per - winner_return_per
        matches[2 * index][feature_position] = winner_serve_adv - loser_serve_adv
        matches[2 * index + 1][feature_position] = loser_serve_adv - winner_serve_adv
        feature_position += 1

        for j, name in enumerate(rolling_cols):
            if winner_id not in player_stats_overall_count[j]:
                winner_stat = 0
                player_stats_overall_count[j][winner_id] = 1
                player_stats_overall_sum[j][winner_id] = df.loc[index, rolling_cols[j]]
            else:
                winner_stat = player_stats_overall_sum[j][winner_id] / player_stats_overall_count[j][winner_id]

                player_stats_overall_count[j][winner_id] = player_stats_overall_count[j][winner_id] + 1
                player_stats_overall_sum[j][winner_id] = player_stats_overall_sum[j][ winner_id] + df.loc[index, rolling_cols[j]]

            if loser_id not in player_stats_overall_count[j]:
                loser_stat = 0
                player_stats_overall_count[j][loser_id] = 1
                player_stats_overall_sum[j][loser_id] = df.loc[index + numOfRows, rolling_cols[j]]
            else:
                loser_stat = player_stats_overall_sum[j][loser_id] / player_stats_overall_count[j][loser_id]

                player_stats_overall_count[j][loser_id] = player_stats_overall_count[j][loser_id] + 1
                player_stats_overall_sum[j][loser_id] = player_stats_overall_sum[j][loser_id] + df.loc[index + numOfRows, rolling_cols[j]]

            if j == 0:
                winner_serv_per = winner_stat
                loser_serv_per = loser_stat
            if j == 1:
                winner_return_per = winner_stat
                loser_return_per = loser_stat

            matches[2 * index][feature_position] = winner_stat - loser_stat
            matches[2 * index + 1][feature_position] = loser_stat - winner_stat
            feature_position += 1

        #serv_adv
        winner_serve_adv = winner_serv_per - loser_return_per
        loser_serve_adv = loser_serv_per - winner_return_per
        matches[2 * index][feature_position] = winner_serve_adv - loser_serve_adv
        matches[2 * index + 1][feature_position] = loser_serve_adv - winner_serve_adv
        feature_position += 1

        matches[2*index][feature_position] = df.loc[index, 'player_rank'] - df.loc[index + numOfRows, 'player_rank']
        matches[2*index + 1][feature_position] = df.loc[index + numOfRows, 'player_rank'] - df.loc[index, 'player_rank']
        feature_position += 1

        matches[2*index][feature_position] = df.loc[index, 'player_log_rank'] - df.loc[index + numOfRows, 'player_log_rank']
        matches[2*index + 1][feature_position] = df.loc[index + numOfRows, 'player_log_rank'] - df.loc[index, 'player_log_rank']
        feature_position += 1

        matches[2*index][feature_position] = df.loc[index, 'player_rank_points'] - df.loc[index + numOfRows, 'player_rank_points']
        matches[2*index + 1][feature_position] = df.loc[index + numOfRows, 'player_rank_points'] - df.loc[index, 'player_rank_points']
        feature_position += 1

        winner_old_elo, loser_old_elo = calculateSurfaceElo(winner_id, loser_id, surface)
        matches[2*index][feature_position] = winner_old_elo - loser_old_elo
        matches[2*index + 1][feature_position] = loser_old_elo - winner_old_elo
        feature_position += 1

        winner_old_elo, loser_old_elo = calculateElo(winner_id, loser_id)
        matches[2*index][feature_position] = winner_old_elo - loser_old_elo
        matches[2*index + 1][feature_position] = loser_old_elo - winner_old_elo
        feature_position += 1

        # per of match won over last 12 months
        total_winner, winner_recent_form, total_loser, loser_recent_form = calculatePlayersRecentFrom(winner_id, loser_id, tourney_date, match_num)
        matches[2 * index][feature_position] = winner_recent_form - loser_recent_form
        matches[2 * index + 1][feature_position] = loser_recent_form - winner_recent_form
        feature_position += 1

        # total matches played in last 12 months
        matches[2 * index][feature_position] = total_winner - total_loser
        matches[2 * index + 1][feature_position] = total_loser - total_winner
        feature_position += 1

        # per of matches won in the same tournament
        winner_tourney_form, loser_tourney_form = setPlayersTournamentFormFeature(winner_id, loser_id, tournament_id)
        matches[2 * index][feature_position] = winner_tourney_form - loser_tourney_form
        matches[2 * index + 1][feature_position] = loser_tourney_form - winner_tourney_form
        feature_position += 1

        # winner_overall_perc, loser_overall_perc = setPlayersOverallWinLossFeature(winner_id, loser_id)
        # matches[2*index][feature_position] = winner_overall_perc - loser_overall_perc
        # matches[2*index + 1][feature_position] = loser_overall_perc - winner_overall_perc
        # feature_position += 1

        winner_score, loser_score = setPlayersHeadToHeadFeature(winner_id, loser_id)
        matches[2 * index][feature_position] = winner_score - loser_score
        matches[2 * index + 1][feature_position] = loser_score - winner_score
        feature_position += 1

        # winner_score, loser_score = setPlayersHeadToHeadSurfaceFeature(winner_id, loser_id, surface)
        # matches[2 * index][feature_position] = winner_score - loser_score
        # matches[2 * index + 1][feature_position] = loser_score - winner_score
        # feature_position += 1
        #
        # temp1, temp2, temp3, temp4 = setPlayersCommonOpponentHeadToHeadFeature(winner_id, loser_id)
        # matches[2 * index][feature_position] = temp1 - temp3
        # matches[2 * index + 1][feature_position] = temp3 - temp1
        # feature_position += 1
        #
        # matches[2 * index][feature_position] = temp2 - temp4
        # matches[2 * index + 1][feature_position] = temp4 - temp2
        # feature_position += 1

        matches[2*index][feature_position] = 1
        matches[2*index + 1][feature_position] = 0

    return feature_position


def calculateSurfaceElo(winner, loser, surface):

    if (winner, surface) not in player_surface_elo:
        winner_elos = []
        winner_matches = 0
        winner_old_elo = 0
    else:
        winner_elos = player_surface_elo[(winner, surface)]
        winner_matches = len(winner_elos)
        winner_old_elo = winner_elos[-1]

    if (loser, surface) not in player_surface_elo:
        loser_elos = []
        loser_matches = 0
        loser_old_elo = 0
    else:
        loser_elos = player_surface_elo[(loser, surface)]
        loser_matches = len(loser_elos)
        loser_old_elo = loser_elos[-1]

    pr_p1_win_elo = 1 / (1 + 10 ** ((loser_old_elo - winner_old_elo) / 400))
    pr_p2_win_elo = 1 / (1 + 10 ** ((winner_old_elo - loser_old_elo) / 400))

    winner_K = 250 / ((winner_matches + 5) ** 0.4)
    loser_K = 250 / ((loser_matches + 5) ** 0.4)

    winner_new_elo = winner_old_elo + winner_K * (1 - pr_p1_win_elo)
    loser_new_elo = loser_old_elo + loser_K * (0 - pr_p2_win_elo)

    winner_elos.append(winner_new_elo)
    loser_elos.append(loser_new_elo)

    player_surface_elo[(winner, surface)] = winner_elos
    player_surface_elo[(loser, surface)] = loser_elos

    return winner_old_elo, loser_old_elo


def calculateElo(winner, loser):

    if winner not in player_elo:
        winner_elos = []
        winner_matches = 0
        winner_old_elo = 0
    else:
        winner_elos = player_elo[winner]
        winner_matches = len(winner_elos)
        winner_old_elo = winner_elos[-1]

    if loser not in player_elo:
        loser_elos = []
        loser_matches = 0
        loser_old_elo = 0
    else:
        loser_elos = player_elo[loser]
        loser_matches = len(loser_elos)
        loser_old_elo = loser_elos[-1]

    pr_p1_win_elo = 1 / (1 + 10 ** ((loser_old_elo - winner_old_elo) / 400))
    pr_p2_win_elo = 1 / (1 + 10 ** ((winner_old_elo - loser_old_elo) / 400))

    winner_K = 250 / ((winner_matches + 5) ** 0.4)
    loser_K = 250 / ((loser_matches + 5) ** 0.4)

    winner_new_elo = winner_old_elo + winner_K * (1 - pr_p1_win_elo)
    loser_new_elo = loser_old_elo + loser_K * (0 - pr_p2_win_elo)

    winner_elos.append(winner_new_elo)
    loser_elos.append(loser_new_elo)

    player_elo[winner] = winner_elos
    player_elo[loser] = loser_elos

    return winner_old_elo, loser_old_elo


def setPlayersOverallFormFeature(winner, loser, tourney_date):
    if winner not in overall_form:
        overall_form[winner] = []

    if loser not in overall_form:
        overall_form[loser] = []

    winner_won_25 = 0

    loser_won_25 = 0

    for num, (won, date) in enumerate(reversed(overall_form[winner])):
        if num < 25:
            winner_won_25 += won
        else:
            break

    for num, (won, date) in enumerate(reversed(overall_form[loser])):
        if num < 25:
            loser_won_25 += won
        else:
            break

    overall_form[winner].append((1, tourney_date))
    overall_form[loser].append((0, tourney_date))

    return winner_won_25, loser_won_25


def setPlayersSurfaceFormFeature(winner, loser, current_date, match_num, surface):
    winner_recent_form = 0
    loser_recent_form = 0

    total_winner = 0
    total_loser = 0
    winner_wins = 0
    loser_wins = 0

    if (winner, surface) not in surface_form:
        surface_form[(winner, surface)] = []

    if (loser, surface) not in surface_form:
        surface_form[(loser, surface)] = []

    for num, (won, date, match) in enumerate(reversed(surface_form[(winner, surface)])):
        if 100 > (current_date - date).days >= 0 and match < match_num:
            winner_wins += won
            total_winner += 1
        else:
            break

    for num, (won, date, match) in enumerate(reversed(surface_form[(loser, surface)])):
        if 100 > (current_date - date).days >= 0 and match < match_num:
            loser_wins += won
            total_loser += 1
        else:
            break

    surface_form[(winner, surface)].append((1, current_date, match_num))
    surface_form[(loser, surface)].append((0, current_date, match_num))

    winner_recent_form = calculateWinPct(total_winner, winner_wins)
    loser_recent_form = calculateWinPct(total_loser, loser_wins)

    return winner_recent_form, loser_recent_form


def setPlayersOverallWinLossFeature(winner, loser):
    winner_won,  winner_total = (0, 0)
    loser_won, loser_total = (0, 0)

    if winner not in matches_won_lost:
        matches_won_lost[winner] = (0, 0)
        (winner_won, winner_total) = (0, 0)
    else:
        (won, total) = matches_won_lost[winner]
        matches_won_lost[winner] = (won + 1, total + 1)
        (winner_won, winner_total) = (won, total)

    if loser not in matches_won_lost:
        matches_won_lost[loser] = (0, 0)
        (loser_won, loser_total) = (0, 0)
    else:
        (won, total) = matches_won_lost[loser]
        matches_won_lost[loser] = (won, total + 1)
        (loser_won, loser_total) = (won, total)

    # if winner_total != 0:
    #     winner_perc = winner_won * 100 / winner_total
    # else:
    #     winner_perc = 0
    #
    # if loser_total != 0:
    #     loser_perc = loser_won * 100 / loser_total
    # else:
    #     loser_perc = 0

    winner_perc = calculateWinPct(winner_total, winner_won)
    loser_perc = calculateWinPct(loser_total, loser_won)

    return winner_perc, loser_perc


def setPlayersHeadToHeadFeature(winner, loser):
    if (winner, loser) not in head_to_head:
        (winner_score, loser_score) = (0, 0)
        head_to_head[(winner, loser)] = (1, 0)
        head_to_head[(loser, winner)] = (0, 1)
    else:
        (winner_score, loser_score) = head_to_head[(winner, loser)]
        head_to_head[(winner, loser)] = (winner_score + 1, loser_score)
        head_to_head[(loser, winner)] = (loser_score, winner_score + 1)

    return winner_score, loser_score


def calculateWinPct(x, y):
    '''
    Find a player's win %
    x = number of matches
    y = number of wins
    if x = 0, return 0
    '''
    if x == 0:
        return 0
    else:
        return y/x


def calculatePlayersRecentFrom(winner, loser, current_date, match_num):
    total_winner = 0
    total_loser = 0
    winner_wins = 0
    loser_wins = 0

    if winner not in overall_form:
        overall_form[winner] = []

    if loser not in overall_form:
        overall_form[loser] = []

    for num, (won, date, match) in enumerate(reversed(overall_form[winner])):
        if 365 > (current_date - date).days >= 0 and match < match_num:
            winner_wins += won
            total_winner += 1
        else:
            break

    for num, (won, date, match) in enumerate(reversed(overall_form[loser])):
        if 365 > (current_date - date).days >= 0 and match < match_num:
            loser_wins += won
            total_loser += 1
        else:
            break

    overall_form[winner].append((1, current_date, match_num))
    overall_form[loser].append((0, current_date, match_num))

    winner_recent_form = calculateWinPct(total_winner, winner_wins)
    loser_recent_form = calculateWinPct(total_loser, loser_wins)
    return total_winner, winner_recent_form, total_loser, loser_recent_form


def setPlayersHeadToHeadSurfaceFeature(winner, loser, surface):
    if (winner, loser, surface) not in head_to_head_surface:
        (winner_score, loser_score) = (0, 0)
        head_to_head_surface[(winner, loser, surface)] = (1, 0)
        head_to_head_surface[(loser, winner, surface)] = (0, 1)
    else:
        (winner_score, loser_score) = head_to_head_surface[(winner, loser, surface)]
        head_to_head_surface[(winner, loser, surface)] = (winner_score + 1, loser_score)
        head_to_head_surface[(loser, winner, surface)] = (loser_score, winner_score + 1)

    return winner_score, loser_score


def setPlayersCommonOpponentHeadToHeadFeature(winner, loser):
    if winner not in common_head_to_head:
        common_head_to_head[winner] = {loser: (0, 0)}

    if loser not in common_head_to_head:
        common_head_to_head[loser] = {winner: (0, 0)}

    if loser not in common_head_to_head[winner]:
        common_head_to_head[winner][loser] = (0, 0)

    if winner not in common_head_to_head[loser]:
        common_head_to_head[loser][winner] = (0, 0)

    (x11, y11) = common_head_to_head[winner][loser]
    (x22, y22) = common_head_to_head[loser][winner]
    common_head_to_head[winner][loser] = (x11 + 1, y11)
    common_head_to_head[loser][winner] = (x22, y22 + 1)

    new_head_to_head_winner = (0, 0)
    new_head_to_head_loser = (0, 0)

    for player_id in common_head_to_head[winner]:
        if player_id in common_head_to_head[loser]:
            (temp1, temp2) = common_head_to_head[winner][player_id]
            t1, t2 = new_head_to_head_winner
            new_head_to_head_winner = t1 + temp1, t2 + temp2

    for player_id in common_head_to_head[loser]:
        if player_id in common_head_to_head[winner]:
            (temp1, temp2) = common_head_to_head[loser][player_id]
            t1, t2 = new_head_to_head_loser
            new_head_to_head_loser = t1 + temp1, t2 + temp2

    temp1, temp2 = new_head_to_head_winner
    temp3, temp4 = new_head_to_head_loser

    temp5 = 0
    temp6 = 0

    if (temp1 + temp2) != 0:
        temp5 = ((temp1 * 100) / (temp1 + temp2))

    if (temp3 + temp4) != 0:
        temp6 = ((temp3 * 100) / (temp3 + temp4))

    return temp1, temp5, temp3, temp6


def setPlayersTournamentFormFeature(winner, loser, tournament):
    if (winner, tournament) not in tournament_form_won:
        tournament_form_won[(winner, tournament)] = (1, 1)
        (winner_won, winner_total) = (0, 0)
    else:
        (won, total) = tournament_form_won[(winner, tournament)]
        tournament_form_won[(winner, tournament)] = (won + 1, total + 1)
        (winner_won, winner_total) = (won, total)

    if (loser, tournament) not in tournament_form_won:
        tournament_form_won[(loser, tournament)] = (0, 1)
        (loser_won, loser_total) = (0, 0)
    else:
        (won, total) = tournament_form_won[(loser, tournament)]
        tournament_form_won[(loser, tournament)] = (won, total + 1)
        (loser_won, loser_total) = (won, total)

    winner_perc = calculateWinPct(winner_total, winner_won)
    loser_perc = calculateWinPct(loser_total, loser_won)

    return winner_perc, loser_perc

def getXSet():
    x_Set = matches[:, :(numOfFeatures - 1)]
    return x_Set


def getYSet():
    y_Set = matches[:, -1]
    return y_Set


df_atp = data_cleaning(df_atp)
df_atp_long = convert_long(df_atp)
df_atp_long = get_new_features(df_atp_long)
df_atp_long.to_csv("newset.csv", index=False, encoding="utf-8-sig", header=True)
feature_position = setPlayersStatisticsFeature(df_atp_long)
np.savetxt("foo.csv", matches, delimiter=",")


x = getXSet()
y = getYSet()

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scaled feature
x_after_min_max_scaler = min_max_scaler.fit_transform(x)

""" Standardisation """

Standardisation = preprocessing.StandardScaler()

# Scaled feature
x_after_Standardisation = Standardisation.fit_transform(x)


np.savetxt("training_data.csv", x_after_min_max_scaler, delimiter=",")
X_train, X_test, Y_train, Y_test = train_test_split(x_after_min_max_scaler, y, test_size=0.2, random_state=42)


kfold = model_selection.KFold(n_splits=10, random_state=100,  shuffle=True)
model_kfold = LogisticRegression(multi_class='ovr', random_state=0, solver='liblinear', penalty='l2')
results_kfold = model_selection.cross_val_score(model_kfold, x_after_min_max_scaler, y, cv=kfold)
print("KFold Logistic Regression Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))

model = LogisticRegression(multi_class='ovr', random_state=0, solver='liblinear', penalty='l2').fit(X_train, Y_train)
importance = model.coef_
for i, v in enumerate(importance):
    print("Feature Logistic Regression:", v)

y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_pred))

logit = LogisticRegression(solver='liblinear')

gs_params = {'C': 10**np.linspace(-2, 2, 26)}
gs = GridSearchCV(logit, gs_params, cv=5)

gs.fit(X_train, Y_train)

log_best = gs.best_estimator_
# for i, v in enumerate(gs.feature_importances_):
#     print("Feature Imrpoved Logistic Regression Accuracy:", v)
# print(log_best.score(X_train, Y_train))
print("Imrpoved Logistic Regression Accuracy:", log_best.score(X_test, Y_test))

# gbc = GradientBoostingClassifier(max_features = 'sqrt')
#
# gs_params = {'n_estimators': [500],
#              'max_depth': [6],
#              'subsample': [1.0],
#              'learning_rate': [0.02]}
#
# gs = GridSearchCV(gbc, gs_params, cv=5)
#
# gs.fit(X_train, Y_train)
#
# gbc_best = gs.best_estimator_
#
# print(gbc_best)
# # print("GradientBoostingClassifier train Accuracy:", gbc_best.score(X_train, Y_train))
# print("GradientBoostingClassifier test Accuracy:", gbc_best.score(X_test, Y_test))

# xgb_model = XGBRegressor()
# xgb_model = XGBRegressor(
#     objective="reg:squarederror",
#     n_estimators=500,
#     learning_rate=0.02,
#     max_depth=6
# )
#
# eval_set = [(X_test, Y_test)]
# xgb_model.fit(X_train,
#           Y_train,
#           eval_set=eval_set,
#           eval_metric="auc",
#           early_stopping_rounds=20)

# param_tuning = {
#         'learning_rate': [0.02],
#         'max_depth': [6],
#         'subsample': [1.0],
#         'n_estimators': [500],
#         'objective': ['reg:squarederror']
#     }

# gs = GridSearchCV(estimator=xgb_model,
#                   param_grid=param_tuning,
#                   cv=5,
#                   n_jobs=-1,
#                   verbose=1)
#
# gs.fit(X_train, Y_train)
#
# gbc_best = gs.best_estimator_
#
# print(gbc_best)
# print("XGBRegressor train Accuracy:", gbc_best.score(X_train, Y_train))
# print("XGBRegressor test Accuracy:", xgb_model.score(X_test, Y_test))

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=500,
    learning_rate=0.02,
    max_depth=6
)
# model = XGBClassifier()
#
# param_tuning = {
#         'learning_rate': [0.02],
#         'max_depth': [6],
#         'subsample': [1.0],
#         'n_estimators': [500],
#         'objective': ['binary:logistic']
#     }

# gs = GridSearchCV(estimator=model,
#                   param_grid=param_tuning,
#                   cv=5,
#                   n_jobs=-1,
#                   verbose=1)
#
# gs.fit(X_train, Y_train)
#
# gbc_best = gs.best_estimator_
#
# print(gbc_best)
# print("XGBClassifier train Accuracy:", gbc_best.score(X_train, Y_train))
# print("XGBClassifier test Accuracy:", gbc_best.score(X_test, Y_test))
eval_set = [(X_test, Y_test)]
model.fit(X_train,
          Y_train,
          eval_set=eval_set,
          eval_metric="auc",
          early_stopping_rounds=20)

y_pred = model.predict(X_test)
print("XGBClassifierAccuracy:", accuracy_score(Y_test, y_pred))
for i, v in enumerate(model.feature_importances_):
    print("Feature XGBClassifierAccuracy:", v)
# model = XGBClassifier()
# model.fit(X_train, Y_train)

# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]

# # evaluate predictions
# accuracy = accuracy_score(Y_test, predictions)
# y_pred = model.predict(X_test)
# print("XGBClassifier Accuracy:", accuracy_score(Y_test, y_pred))
# predictions = model.predict(X_test)
# for input, prediction, test in zip(X_test, predictions, Y_test):
#     if prediction != test:
#         print(input, 'has been classified as ', prediction, 'and should be ', test)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
y_pred = lda.predict(X_test)
print("LinearDiscriminantAnalysis Accuracy:", accuracy_score(Y_test, y_pred))

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred = gnb.predict(X_test)
print("GaussianNB Accuracy:", accuracy_score(Y_test, y_pred))

clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6)

clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)
print("Random Forest Classifier Accuracy:", accuracy_score(Y_test, y_pred))

# kfold = model_selection.KFold(n_splits=10, random_state=100,  shuffle=True)
# model_kfold = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=6)
# results_kfold = model_selection.cross_val_score(model_kfold, x, y, cv=kfold)
# print("Random Forest Classifier KFold Accuracy: %.2f%%" % (results_kfold.mean() * 100.0))

svclassifier = SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, x_after_min_max_scaler, y, cv=5)
print(scores)
svclassifier.fit(X_train, Y_train)
y_pred = svclassifier.predict(X_test)
# print(confusion_matrix(Y_test,y_pred))
# # print(classification_report(Y_test,y_pred))
print("SVC Accuracy:", accuracy_score(Y_test, y_pred))
print("SVC Precision Score:", precision_score(Y_test, y_pred))

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, Y_train)
# # logistic regression - penaltystr, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
#
y_pred = clf.predict(X_test)
clf.fit(X_train, Y_train)
# print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
print("SVC rbf kernel Accuracy:", accuracy_score(Y_test, y_pred))
print("SVC Precision Score:", precision_score(Y_test, y_pred))


clf = SVC(kernel='poly', degree=3, gamma=2)
clf.fit(X_train, Y_train)
# # logistic regression - penaltystr, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
#
y_pred = clf.predict(X_test)
# print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
print("SVC poly Accuracy:", accuracy_score(Y_test, y_pred))
print("SVC Precision Score:", precision_score(Y_test, y_pred))
# # Initialize the constructor
# model = Sequential()
#
# # Add an input layer
# model.add(Dense(12, activation='relu', input_shape=(11,)))
#
# # Add one hidden layer
# model.add(Dense(8, activation='relu'))
#
# # Add an output layer
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=1)
#
# y_pred = model.predict(X_test)
# print("Accuracy:",accuracy_score(Y_test, y_pred))
# score = model.evaluate(X_test, Y_test,verbose=1)
#
# print(score)
#
# confusion_matrix(Y_test, y_pred)
#
# # Precision
# precision_score(Y_test, y_pred)
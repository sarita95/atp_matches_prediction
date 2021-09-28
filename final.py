from datetime import datetime
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
absolute_path_to_data = "C:\\Users\\Sara\\Desktop\\master\\atp_matches_prediction\\atp_matches_all.csv"


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
    df = df.assign(loser_1stRetWonPer=pd.Series((df['winner_1stIn'] - df['winner_1stWon']) / df['winner_1stIn']))
    df = df.assign(winner_1stRetWonPer=pd.Series((df['loser_1stIn'] - df['loser_1stWon']) / df['loser_1stIn']))
    df['winner_1stRetWonPer'].fillna(0, inplace=True)
    df['loser_1stRetWonPer'].fillna(0, inplace=True)
    df = df.assign(loser_2ndRetWonPer=pd.Series((df['winner_2ndIn'] - df['winner_2ndWon']) / df['winner_2ndIn']))
    df = df.assign(winner_2ndRetWonPer=pd.Series((df['loser_2ndIn'] - df['loser_2ndWon']) / df['loser_2ndIn']))
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

    return df


def convert_long(df, numOfRows):
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


def calculateFeatureImportance(allFeatures, features, algorithm):
    features = np.absolute(features)
    feat_imp = sorted(list(zip(allFeatures, features)), key=lambda x: -x[1])

    feat_imp_df = pd.DataFrame({'feature': [x[0] for x in feat_imp],
                                'importance': [x[1] for x in feat_imp]})

    sns.barplot(data=feat_imp_df, x='feature', y='importance')

    plt.xticks(rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title('Feature importances (' + algorithm + ')', size=21)
    plt.savefig('featureImportance.png', bbox_inches='tight', dpi=100)


def logisticRegression(X_train, X_test, Y_train, Y_test, allFeatures):
    model = LogisticRegression(multi_class='ovr', random_state=0, solver='liblinear', penalty='l2').fit(X_train,
                                                                                                        Y_train)
    y_pred = model.predict(X_test)
    calculateFeatureImportance(allFeatures, model.coef_[0], 'Logistic Regression')

    return accuracy_score(Y_test, y_pred)


def kFoldLogisticRegression(X_train, X_test, Y_train, Y_test, allFeatures):
    logit = LogisticRegression(solver='liblinear')

    gs_params = {'C': 10 ** np.linspace(-2, 2, 26)}
    gs = GridSearchCV(logit, gs_params, cv=5)

    gs.fit(X_train, Y_train)

    log_best = gs.best_estimator_

    calculateFeatureImportance(allFeatures, log_best.coef_[0], 'KFold Logistic Regression')

    return log_best.score(X_test, Y_test)


def xgbClassifier(X_train, X_test, Y_train, Y_test, allFeatures):
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6
    )

    eval_set = [(X_test, Y_test)]
    model.fit(X_train,
              Y_train,
              eval_set=eval_set,
              eval_metric="auc",
              early_stopping_rounds=20)

    y_pred = model.predict(X_test)

    calculateFeatureImportance(allFeatures, model.feature_importances_, 'XGBoost Classifier')
    score = accuracy_score(Y_test, y_pred)
    return score


def linearDiscriminantAnalysis(X_train, X_test, Y_train, Y_test, allFeatures):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    y_pred = lda.predict(X_test)

    calculateFeatureImportance(allFeatures, lda.coef_[0], 'Linear Discriminant Analysis')
    return accuracy_score(Y_test, y_pred)


def gaussianNB(X_train, X_test, Y_train, Y_test, allFeatures):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_test)
    imps = permutation_importance(gnb, X_test, Y_test)
    importances = imps.importances_mean
    calculateFeatureImportance(allFeatures, importances, 'GaussianNB')

    return accuracy_score(Y_test, y_pred)


def randomForestClassifier(X_train, X_test, Y_train, Y_test, allFeatures):
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6)

    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    calculateFeatureImportance(allFeatures, clf.feature_importances_, 'Random Forest Classifier ')

    return accuracy_score(Y_test, y_pred)


def svclassifierAlg(X_train, X_test, Y_train, Y_test, allFeatures):
    svclassifier = SVC(kernel='linear', C=1, random_state=42)
    svclassifier.fit(X_train, Y_train)
    y_pred = svclassifier.predict(X_test)
    calculateFeatureImportance(allFeatures, svclassifier.coef_[0], 'SVC')

    return accuracy_score(Y_test, y_pred)


class FeaturesEngineering:
    def __init__(self, dirname, numOfRollingFeatures, numOfAllFeatures, rollingCols, featureColls):
        self.dirname = dirname
        self.numOfRollingFeatures = numOfRollingFeatures
        self.featureColls = featureColls
        z = 0
        self.rollingCols = rollingCols
        self.player_stats_surface_sum = [dict() for z in range(numOfRollingFeatures)]
        self.player_stats_surface_count = [dict() for z in range(numOfRollingFeatures)]
        self.tournament_form_won = dict()
        self.overall_form = dict()
        self.recent_form = dict()
        self.matches_won_lost = dict()
        self.head_to_head = dict()
        self.numOfRows = 0
        self.player_elo = dict()
        self.player_tournament_elo = dict()
        self.player_surface_elo = dict()
        self.numOfFeatures = numOfAllFeatures
        self.matches = np.zeros((0, 0))

    def createDfAndFeatures(self):
        df_atp = pd.read_csv(self.dirname)
        self.numOfRows = df_atp.shape[0]
        self.matches = np.zeros((2 * self.numOfRows, self.numOfFeatures))
        df_atp = data_cleaning(df_atp)
        df_atp_long = convert_long(df_atp, self.numOfRows)
        df_atp_long = get_new_features(df_atp_long)
        self.setPlayersStatisticsFeature(df_atp_long)

    def setPlayersStatisticsFeature(self, df):
        for index in range(self.numOfRows):
            winner_id = df.loc[index, 'player_id']
            loser_id = df.loc[index + self.numOfRows, 'player_id']
            surface = df.loc[index, 'surface']
            tournament_id = str(df.loc[index, 'tourney_id'])[5:]
            tourney_date = df.loc[index, 'tourney_date']
            match_num = df.loc[index, 'match_num']

            feature_position = 0

            winner_serv_per = 0
            loser_serv_per = 0
            winner_return_per = 0
            loser_return_per = 0
            for j, name in enumerate(self.rollingCols):
                if (winner_id, surface) not in self.player_stats_surface_count[j]:
                    self.player_stats_surface_count[j][(winner_id, surface)] = 1
                    self.player_stats_surface_sum[j][(winner_id, surface)] = df.loc[index, self.rollingCols[j]]
                    winner_stat = 0
                else:
                    winner_stat = self.player_stats_surface_sum[j][(winner_id, surface)] / self.player_stats_surface_count[j][
                        (winner_id, surface)]

                    self.player_stats_surface_count[j][(winner_id, surface)] = self.player_stats_surface_count[j][
                                                                              (winner_id, surface)] + 1
                    self.player_stats_surface_sum[j][(winner_id, surface)] = self.player_stats_surface_sum[j][(winner_id, surface)] + \
                                                                        df.loc[index, self.rollingCols[j]]

                if (loser_id, surface) not in self.player_stats_surface_count[j]:
                    self.player_stats_surface_count[j][(loser_id, surface)] = 1
                    self.player_stats_surface_sum[j][(loser_id, surface)] = df.loc[index + self.numOfRows, self.rollingCols[j]]
                    loser_stat = 0
                else:
                    loser_stat = self.player_stats_surface_sum[j][(loser_id, surface)] / self.player_stats_surface_count[j][
                        (loser_id, surface)]

                    self.player_stats_surface_count[j][(loser_id, surface)] = self.player_stats_surface_count[j][
                                                                             (loser_id, surface)] + 1
                    self.player_stats_surface_sum[j][(loser_id, surface)] = self.player_stats_surface_sum[j][(loser_id, surface)] + \
                                                                       df.loc[index + self.numOfRows, self.rollingCols[j]]
                if name == 'player_overall_win_on_serve_per':
                    winner_serv_per = winner_stat
                    loser_serv_per = loser_stat
                if name == 'player_avg_return_per':
                    winner_return_per = winner_stat
                    loser_return_per = loser_stat
                self.matches[2 * index][feature_position] = winner_stat - loser_stat
                self.matches[2 * index + 1][feature_position] = loser_stat - winner_stat
                feature_position += 1

            if 'player_serve_advantage' in self.featureColls:
                winner_serve_adv = winner_serv_per - loser_return_per
                loser_serve_adv = loser_serv_per - winner_return_per
                self.matches[2 * index][feature_position] = winner_serve_adv - loser_serve_adv
                self.matches[2 * index + 1][feature_position] = loser_serve_adv - winner_serve_adv
                feature_position += 1

            if 'player_age' in self.featureColls:
                # Age feature
                self.matches[2 * index][feature_position] = df.loc[index, 'player_age'] - df.loc[index + self.numOfRows, 'player_age']
                self.matches[2 * index + 1][feature_position] = df.loc[index + self.numOfRows, 'player_age'] - df.loc[index, 'player_age']
                feature_position += 1

            if 'player_ht' in self.featureColls:
                # Height feature
                self.matches[2 * index][feature_position] = df.loc[index, 'player_ht'] - df.loc[index + self.numOfRows, 'player_ht']
                self.matches[2 * index + 1][feature_position] = df.loc[index + self.numOfRows, 'player_ht'] - df.loc[index, 'player_ht']
                feature_position += 1

            if 'player_rank' in self.featureColls:
                self.matches[2*index][feature_position] = df.loc[index, 'player_rank'] - df.loc[index + self.numOfRows, 'player_rank']
                self.matches[2*index + 1][feature_position] = df.loc[index + self.numOfRows, 'player_rank'] - df.loc[index, 'player_rank']
                feature_position += 1

            if 'player_log_rank' in self.featureColls:
                self.matches[2*index][feature_position] = df.loc[index, 'player_log_rank'] - df.loc[index + self.numOfRows, 'player_log_rank']
                self.matches[2*index + 1][feature_position] = df.loc[index + self.numOfRows, 'player_log_rank'] - df.loc[index, 'player_log_rank']
                feature_position += 1

            if 'player_rank_points' in self.featureColls:
                self.matches[2*index][feature_position] = df.loc[index, 'player_rank_points'] - df.loc[index + self.numOfRows, 'player_rank_points']
                self.matches[2*index + 1][feature_position] = df.loc[index + self.numOfRows, 'player_rank_points'] - df.loc[index, 'player_rank_points']
                feature_position += 1

            if 'surface_elo' in self.featureColls:
                winner_old_elo, loser_old_elo = self.calculateSurfaceElo(winner_id, loser_id, surface)
                self.matches[2*index][feature_position] = winner_old_elo - loser_old_elo
                self.matches[2*index + 1][feature_position] = loser_old_elo - winner_old_elo
                feature_position += 1

            if '538elo' in self.featureColls:
                winner_old_elo, loser_old_elo = self.calculateElo(winner_id, loser_id)
                self.matches[2*index][feature_position] = winner_old_elo - loser_old_elo
                self.matches[2*index + 1][feature_position] = loser_old_elo - winner_old_elo
                feature_position += 1

            total_winner, winner_recent_form, total_loser, loser_recent_form = self.calculatePlayersRecentFrom( winner_id, loser_id, tourney_date, match_num)
            if 'per_of_matches_won_12_months' in self.featureColls:
                # per of match won over last 12 months
                self.matches[2 * index][feature_position] = winner_recent_form - loser_recent_form
                self.matches[2 * index + 1][feature_position] = loser_recent_form - winner_recent_form
                feature_position += 1

            if 'total_matches_12_months' in self.featureColls:
                # total matches played in last 12 months
                self.matches[2 * index][feature_position] = total_winner - total_loser
                self.matches[2 * index + 1][feature_position] = total_loser - total_winner
                feature_position += 1

            if 'tournament_form' in self.featureColls:
                # per of matches won in the same tournament
                winner_tourney_form, loser_tourney_form = self.setPlayersTournamentFormFeature(winner_id, loser_id, tournament_id)
                self.matches[2 * index][feature_position] = winner_tourney_form - loser_tourney_form
                self.matches[2 * index + 1][feature_position] = loser_tourney_form - winner_tourney_form
                feature_position += 1

            if 'overallWinLoss' in self.featureColls:
                winner_overall_perc, loser_overall_perc = self.setPlayersOverallWinLossFeature(winner_id, loser_id)
                self.matches[2*index][feature_position] = winner_overall_perc - loser_overall_perc
                self.matches[2*index + 1][feature_position] = loser_overall_perc - winner_overall_perc
                feature_position += 1

            if 'headToHead' in self.featureColls:
                winner_score, loser_score = self.setPlayersHeadToHeadFeature(winner_id, loser_id)
                self.matches[2 * index][feature_position] = winner_score - loser_score
                self.matches[2 * index + 1][feature_position] = loser_score - winner_score
                feature_position += 1

            self.matches[2*index][feature_position] = 1
            self.matches[2*index + 1][feature_position] = 0

    def calculateSurfaceElo(self, winner, loser, surface):

        if (winner, surface) not in self.player_surface_elo:
            winner_elos = []
            winner_matches = 0
            winner_old_elo = 0
        else:
            winner_elos = self.player_surface_elo[(winner, surface)]
            winner_matches = len(winner_elos)
            winner_old_elo = winner_elos[-1]

        if (loser, surface) not in self.player_surface_elo:
            loser_elos = []
            loser_matches = 0
            loser_old_elo = 0
        else:
            loser_elos = self.player_surface_elo[(loser, surface)]
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

        self.player_surface_elo[(winner, surface)] = winner_elos
        self.player_surface_elo[(loser, surface)] = loser_elos

        return winner_old_elo, loser_old_elo

    def calculateElo(self, winner, loser):

        if winner not in self.player_elo:
            winner_elos = []
            winner_matches = 0
            winner_old_elo = 0
        else:
            winner_elos = self.player_elo[winner]
            winner_matches = len(winner_elos)
            winner_old_elo = winner_elos[-1]

        if loser not in self.player_elo:
            loser_elos = []
            loser_matches = 0
            loser_old_elo = 0
        else:
            loser_elos = self.player_elo[loser]
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

        self.player_elo[winner] = winner_elos
        self.player_elo[loser] = loser_elos

        return winner_old_elo, loser_old_elo

    def setPlayersOverallWinLossFeature(self, winner, loser):
        winner_won,  winner_total = (0, 0)
        loser_won, loser_total = (0, 0)

        if winner not in self.matches_won_lost:
            self.matches_won_lost[winner] = (0, 0)
            (winner_won, winner_total) = (0, 0)
        else:
            (won, total) = self.matches_won_lost[winner]
            self.matches_won_lost[winner] = (won + 1, total + 1)
            (winner_won, winner_total) = (won, total)

        if loser not in self.matches_won_lost:
            self.matches_won_lost[loser] = (0, 0)
            (loser_won, loser_total) = (0, 0)
        else:
            (won, total) = self.matches_won_lost[loser]
            self.matches_won_lost[loser] = (won, total + 1)
            (loser_won, loser_total) = (won, total)

        winner_perc = calculateWinPct(winner_total, winner_won)
        loser_perc = calculateWinPct(loser_total, loser_won)

        return winner_perc, loser_perc

    def setPlayersHeadToHeadFeature(self, winner, loser):
        if (winner, loser) not in self.head_to_head:
            (winner_score, loser_score) = (0, 0)
            self.head_to_head[(winner, loser)] = (1, 0)
            self.head_to_head[(loser, winner)] = (0, 1)
        else:
            (winner_score, loser_score) = self.head_to_head[(winner, loser)]
            self.head_to_head[(winner, loser)] = (winner_score + 1, loser_score)
            self.head_to_head[(loser, winner)] = (loser_score, winner_score + 1)

        return winner_score, loser_score

    def calculatePlayersRecentFrom(self, winner, loser, current_date, match_num):
        total_winner = 0
        total_loser = 0
        winner_wins = 0
        loser_wins = 0

        if winner not in self.overall_form:
            self.overall_form[winner] = []

        if loser not in self.overall_form:
            self.overall_form[loser] = []

        for num, (won, date, match) in enumerate(reversed(self.overall_form[winner])):
            if 365 > (current_date - date).days >= 0 and match < match_num:
                winner_wins += won
                total_winner += 1
            else:
                break

        for num, (won, date, match) in enumerate(reversed(self.overall_form[loser])):
            if 365 > (current_date - date).days >= 0 and match < match_num:
                loser_wins += won
                total_loser += 1
            else:
                break

        self.overall_form[winner].append((1, current_date, match_num))
        self.overall_form[loser].append((0, current_date, match_num))

        winner_recent_form = calculateWinPct(total_winner, winner_wins)
        loser_recent_form = calculateWinPct(total_loser, loser_wins)
        return total_winner, winner_recent_form, total_loser, loser_recent_form

    def setPlayersTournamentFormFeature(self, winner, loser, tournament):
        if (winner, tournament) not in self.tournament_form_won:
            self.tournament_form_won[(winner, tournament)] = (1, 1)
            (winner_won, winner_total) = (0, 0)
        else:
            (won, total) = self.tournament_form_won[(winner, tournament)]
            self.tournament_form_won[(winner, tournament)] = (won + 1, total + 1)
            (winner_won, winner_total) = (won, total)

        if (loser, tournament) not in self.tournament_form_won:
            self.tournament_form_won[(loser, tournament)] = (0, 1)
            (loser_won, loser_total) = (0, 0)
        else:
            (won, total) = self.tournament_form_won[(loser, tournament)]
            self.tournament_form_won[(loser, tournament)] = (won, total + 1)
            (loser_won, loser_total) = (won, total)

        winner_perc = calculateWinPct(winner_total, winner_won)
        loser_perc = calculateWinPct(loser_total, loser_won)

        return winner_perc, loser_perc

    def getXSet(self):
        x_Set = self.matches[:, :(self.numOfFeatures - 1)]
        return x_Set

    def getYSet(self):
        y_Set = self.matches[:, -1]
        return y_Set

    def runAlgorithm(self, algorithm, test_size):
        x = self.getXSet()
        y = self.getYSet()
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

        # Scaled feature
        x_after_min_max_scaler = min_max_scaler.fit_transform(x)
        allFeatures = self.rollingCols + self.featureColls
        X_train, X_test, Y_train, Y_test = train_test_split(x_after_min_max_scaler, y, test_size=test_size/100, random_state=42)
        # trainSet = pd.DataFrame(data=X_train, index=None, columns=allFeatures)
        # trainSet['result'] = Y_train
        # trainSet = pd.DataFrame(data=X_train, index=None, columns=allFeatures)
        # trainSet['result'] = Y_train
        # trainSet.to_csv("trainSet.csv", index=False, encoding="utf-8-sig", header=True)
        # testSet.to_csv("testSet.csv", index=False, encoding="utf-8-sig", header=True)

        score = 0
        if algorithm == 0:
            score = logisticRegression(X_train, X_test, Y_train, Y_test, allFeatures)
        elif algorithm == 1:
            score = kFoldLogisticRegression(X_train, X_test, Y_train, Y_test, allFeatures)
        elif algorithm == 2:
            score = xgbClassifier(X_train, X_test, Y_train, Y_test, allFeatures)
        elif algorithm == 3:
            score = linearDiscriminantAnalysis(X_train, X_test, Y_train, Y_test, allFeatures)
        elif algorithm == 4:
            score = gaussianNB(X_train, X_test, Y_train, Y_test, allFeatures)
        elif algorithm == 5:
            score = randomForestClassifier(X_train, X_test, Y_train, Y_test, allFeatures)
        elif algorithm == 6:
            score = svclassifierAlg(X_train, X_test, Y_train, Y_test, allFeatures)

        return score


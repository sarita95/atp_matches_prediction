from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import svm

from pre_processing.data_pre_processing import DataProcessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

absolute_path_to_data = "C:\\Users\\Sara\\PycharmProjects\\tennis_atp"


def parseDate(dateNum):
    strDate = str(dateNum)
    year = int(strDate[0] + strDate[1] + strDate[2] + strDate[3])
    month = int(strDate[4] + strDate[5])
    date = int(strDate[6] + strDate[7])

    answer = 365 * year + 30 * month + date

    return answer


class FeaturesExtraction:
    def __init__(self, dirname):
        self.dataProcessing = DataProcessing(dirname)
        self.dataProcessing.getCleanData()
        self.matches = self.dataProcessing.matches
        self.numOfRows = self.matches.shape[0]
        self.numOfCols = self.matches.shape[1]
        self.numOfFeatures = int(73)
        self.playerOffset = int(36)
        self.Features = np.zeros((self.numOfRows, self.numOfFeatures))

        z = 0
        self.player_stats_overall_sum = [dict() for z in range(9)]
        self.player_stats_overall_count = [dict() for z in range(9)]
        self.tournament_form_won = dict()
        self.tournament_form_total = dict()

        self.overall_form = dict()
        self.surface_form = dict()
        self.matches_won_lost_surface = dict()
        self.matches_won_lost = dict()
        self.head_to_head = dict()
        self.head_to_head_surface = dict()
        self.common_head_to_head = dict()

    def getXSet(self):
        x_Set = self.Features[:, :(self.numOfFeatures - 1)]
        return x_Set

    def getYSet(self):
        y_Set = self.Features[:, -1]
        return y_Set

    def setPlayersEntryFeature(self, position):
        playersEntries_dict = self.dataProcessing.getPlayersEntryDict()
        for index, key in enumerate(playersEntries_dict):
            playersEntries_dict[key] = float(index)

        for index, row in self.trainSet.iterrows():
            self.X_train[2*index][position] = playersEntries_dict[' '] if row['winner_entry'] is np.nan else playersEntries_dict[row['winner_entry']]
            self.X_train[2*index + 1][position] = playersEntries_dict[' '] if row['loser_entry'] is np.nan else playersEntries_dict[row['loser_entry']]

        for index, row in self.testSet.iterrows():
            self.X_test[2*index][position] = playersEntries_dict[' '] if row['winner_entry'] is np.nan else playersEntries_dict[row['winner_entry']]
            self.X_test[2*index + 1][position] = playersEntries_dict[' '] if row['loser_entry'] is np.nan else playersEntries_dict[row['loser_entry']]

    def setResult(self, index):
        if index % 2 == 0:
            self.Features[index][self.numOfFeatures - 1] = 1
        else:
            self.Features[index][self.numOfFeatures - 1] = 0

    def setPlayersInfoFeature(self, feature_position, index):
        # 8-winner_seed, 9-winner_entry, 10-winner_name, 11-winner_hand, 12-winner_ht, 13-winner_ioc, 14-winner_age
        # 16-loser_seed, 17-loser_entry, 18-loser_name, 19-loser_hand, 20-loser_ht, 21-loser_ioc, 22-loser_age
        if index % 2 == 0:
            firstPlayerFeature = int(feature_position)
            secondPlayerFeature = int(feature_position + self.playerOffset)
        else:
            firstPlayerFeature = int(feature_position + self.playerOffset)
            secondPlayerFeature = int(feature_position)

        for j in range(8, 15):
            # exclude name and country and entry
            if j != 9 and j != 10 and j != 13:
                self.Features[index][firstPlayerFeature] = float(self.matches.iloc[index, j])
                self.Features[index][secondPlayerFeature] = float(self.matches.iloc[index, j+8])

                firstPlayerFeature += 1
                secondPlayerFeature += 1

        # rank and rank_points
        self.Features[index][firstPlayerFeature] = float(self.matches.iloc[index, 45])
        self.Features[index][secondPlayerFeature] = float(self.matches.iloc[index, 47])

        firstPlayerFeature += 1
        secondPlayerFeature += 1

        self.Features[index][firstPlayerFeature] = float(self.matches.iloc[index, 46])
        self.Features[index][secondPlayerFeature] = float(self.matches.iloc[index, 48])

        feature_position += 6
        return feature_position

    def setPlayersStatisticsFeature(self, feature_position, index):
        # 27-w_ace, 28-w_df, 29-w_svpt, 30-w_1stIn, 31-w_1stWon, 32-w_2ndWon, 33-w_SvGms, 34-w_bpSaved, 35-w_bpFaced
        # 36-l_ace, 37-l_df, 38-l_svpt, 39-l_1stIn, 40-l_1stWon, 41-l_2ndWon, 42-l_SvGms, 43-l_bpSaved, 44-l_bpFaced
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        for j in range(0, 9):
            if winner_id not in self.player_stats_overall_count[j]:
                self.Features[index][firstPlayerFeature] = 0

                self.player_stats_overall_count[j][winner_id] = 1
                self.player_stats_overall_sum[j][winner_id] = int(self.matches.iloc[index, j+27])
            else:
                self.Features[index][firstPlayerFeature] = self.player_stats_overall_sum[j][winner_id] / self.player_stats_overall_count[j][winner_id]

                self.player_stats_overall_count[j][winner_id] = self.player_stats_overall_count[j][winner_id] + 1
                self.player_stats_overall_sum[j][winner_id] = self.player_stats_overall_sum[j][winner_id] + int(self.matches.iloc[index, j+27])

            if loser_id not in self.player_stats_overall_count[j]:
                self.Features[index][secondPlayerFeature] = 0

                self.player_stats_overall_count[j][loser_id] = 1
                self.player_stats_overall_sum[j][loser_id] = int(self.matches.iloc[index, j + 36])
            else:
                self.Features[index][secondPlayerFeature] = self.player_stats_overall_sum[j][loser_id] / self.player_stats_overall_count[j][loser_id]

                self.player_stats_overall_count[j][loser_id] = self.player_stats_overall_count[j][loser_id] + 1
                self.player_stats_overall_sum[j][loser_id] = self.player_stats_overall_sum[j][loser_id] + int(self.matches.iloc[index, j + 36])

            firstPlayerFeature += 1
            secondPlayerFeature += 1

            # self.Features[index][firstPlayerFeature] = self.career_stats_winner_total[index][j]
            # self.Features[index][secondPlayerFeature] = self.career_stats_loser_total[index][j]
            # firstPlayerFeature += 1
            # secondPlayerFeature += 1

        feature_position += 9

        return feature_position

    def setPlayersTournamentFormFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]
        tournament_id = str(self.matches.iloc[index, 0])[5:]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        if (winner_id, tournament_id) not in self.tournament_form_won:
            self.tournament_form_won[(winner_id, tournament_id)] = 0
            self.tournament_form_total[(winner_id, tournament_id)] = 0

        if (loser_id, tournament_id) not in self.tournament_form_won:
            self.tournament_form_won[(loser_id, tournament_id)] = 0
            self.tournament_form_total[(loser_id, tournament_id)] = 0

        self.Features[index][firstPlayerFeature] = self.tournament_form_won[(winner_id, tournament_id)]
        self.Features[index][secondPlayerFeature] = self.tournament_form_won[(loser_id, tournament_id)]
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = self.tournament_form_total[(winner_id, tournament_id)]
        self.Features[index][secondPlayerFeature] = self.tournament_form_total[(loser_id, tournament_id)]
        firstPlayerFeature += 1
        secondPlayerFeature += 1

        temp11 = 1
        temp12 = 1

        if self.tournament_form_total[(winner_id, tournament_id)] != 0:
            temp11 = self.tournament_form_total[(winner_id, tournament_id)]
        if self.tournament_form_total[(loser_id, tournament_id)] != 0:
            temp12 = self.tournament_form_total[(loser_id, tournament_id)]

        self.Features[index][firstPlayerFeature] = self.tournament_form_won[(winner_id, tournament_id)] * 100 / temp11
        self.Features[index][secondPlayerFeature] = self.tournament_form_won[(loser_id, tournament_id)] * 100 / temp12

        self.tournament_form_won[(winner_id, tournament_id)] += 1

        self.tournament_form_total[(winner_id, tournament_id)] += 1
        self.tournament_form_total[(loser_id, tournament_id)] += 1

        feature_position += 3

        return feature_position

    def setPlayersOverallFormFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        if winner_id not in self.overall_form:
            self.overall_form[winner_id] = []

        if loser_id not in self.overall_form:
            self.overall_form[loser_id] = []

        winner_won_5 = 0
        winner_won_10 = 0
        winner_won_15 = 0
        winner_won_25 = 0

        loser_won_5 = 0
        loser_won_10 = 0
        loser_won_15 = 0
        loser_won_25 = 0

        for num, (won, date) in enumerate(reversed(self.overall_form[winner_id])):
            if num < 25:
                winner_won_25 += won
                if num < 15:
                    winner_won_15 += won
                    if num < 10:
                        winner_won_10 += won
                        if num < 5:
                            winner_won_5 += won
            else:
                break

        for num, (won, date) in enumerate(reversed(self.overall_form[loser_id])):
            if num < 25:
                loser_won_25 += won
                if num < 15:
                    loser_won_15 += won
                    if num < 10:
                        loser_won_10 += won
                        if num < 5:
                            loser_won_5 += won
            else:
                break

        self.Features[index][firstPlayerFeature] = winner_won_5
        self.Features[index][secondPlayerFeature] = loser_won_5
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won_10
        self.Features[index][secondPlayerFeature] = loser_won_10
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won_15
        self.Features[index][secondPlayerFeature] = loser_won_15
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won_25
        self.Features[index][secondPlayerFeature] = loser_won_25
        firstPlayerFeature += 1
        secondPlayerFeature += 1

        total_winner_month = 0
        total_loser_month = 0
        winner_won_month = 0
        loser_won_month = 0

        for num, (won, date) in enumerate(reversed(self.overall_form[winner_id])):
            if 30 > parseDate(self.matches.iloc[index, 5]) - parseDate(date) >= 0:
                winner_won_month += won
                total_winner_month += 1
            else:
                break

        for num, (won, date) in enumerate(reversed(self.overall_form[loser_id])):
            if 30 > parseDate(self.matches.iloc[index, 5]) - parseDate(date) >= 0:
                loser_won_month += won
                total_loser_month += 1
            else:
                break

        self.Features[index][firstPlayerFeature] = winner_won_month
        self.Features[index][secondPlayerFeature] = loser_won_month
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = total_winner_month
        self.Features[index][secondPlayerFeature] = total_loser_month

        self.overall_form[winner_id].append((1, int(self.matches.iloc[index, 5])))
        self.overall_form[loser_id].append((0, int(self.matches.iloc[index, 5])))

        feature_position += 6

        return feature_position

    def setPlayersSurfaceFormFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]
        surface = self.matches.iloc[index, 2]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        if (winner_id, surface) not in self.surface_form:
            self.surface_form[(winner_id, surface)] = []

        if (loser_id, surface) not in self.surface_form:
            self.surface_form[(loser_id, surface)] = []

        winner_won_5 = 0
        winner_won_10 = 0
        winner_won_15 = 0
        winner_won_25 = 0

        loser_won_5 = 0
        loser_won_10 = 0
        loser_won_15 = 0
        loser_won_25 = 0

        for num, won in enumerate(reversed(self.surface_form[(winner_id, surface)])):
            if num < 25:
                winner_won_25 += won
                if num < 15:
                    winner_won_15 += won
                    if num < 10:
                        winner_won_10 += won
                        if num < 5:
                            winner_won_5 += won
            else:
                break

        for num, won, in enumerate(reversed(self.surface_form[(loser_id, surface)])):
            if num < 25:
                loser_won_25 += won
                if num < 15:
                    loser_won_15 += won
                    if num < 10:
                        loser_won_10 += won
                        if num < 5:
                            loser_won_5 += won
            else:
                break

        self.Features[index][firstPlayerFeature] = winner_won_5
        self.Features[index][secondPlayerFeature] = loser_won_5
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won_10
        self.Features[index][secondPlayerFeature] = loser_won_10
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won_15
        self.Features[index][secondPlayerFeature] = loser_won_15
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won_25
        self.Features[index][secondPlayerFeature] = loser_won_25
        firstPlayerFeature += 1
        secondPlayerFeature += 1

        self.surface_form[(winner_id, surface)].append(1)
        self.surface_form[(loser_id, surface)].append(0)

        winner_won,  winner_total = (0, 0)
        loser_won, loser_total = (0, 0)

        if (winner_id, surface) not in self.matches_won_lost_surface:
            self.matches_won_lost_surface[(winner_id, surface)] = (0, 0)
            (winner_won, winner_total) = (0, 0)
        else:
            (won, total) = self.matches_won_lost_surface[(winner_id, surface)]
            self.matches_won_lost_surface[(winner_id, surface)] = (won + 1, total + 1)
            (winner_won, winner_total) = (won, total)

        if (loser_id, surface) not in self.matches_won_lost_surface:
            self.matches_won_lost_surface[(loser_id, surface)] = (0, 0)
            (loser_won, loser_total) = (0, 0)
        else:
            (won, total) = self.matches_won_lost_surface[(loser_id, surface)]
            self.matches_won_lost_surface[(loser_id, surface)] = (won, total + 1)
            (loser_won, loser_total) = (won, total)

        if winner_total != 0:
            winner_perc = winner_won * 100 / winner_total
        else:
            winner_perc = 0

        if loser_total != 0:
            loser_perc = loser_won * 100 / loser_total
        else:
            loser_perc = 0

        self.Features[index][firstPlayerFeature] = winner_perc
        self.Features[index][secondPlayerFeature] = loser_perc
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won
        self.Features[index][secondPlayerFeature] = loser_won

        feature_position += 6

        return feature_position

    def setPlayersOverallWinLossFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        winner_won,  winner_total = (0, 0)
        loser_won, loser_total = (0, 0)

        if winner_id not in self.matches_won_lost:
            self.matches_won_lost[winner_id] = (0, 0)
            (winner_won, winner_total) = (0, 0)
        else:
            (won, total) = self.matches_won_lost[winner_id]
            self.matches_won_lost[winner_id] = (won + 1, total + 1)
            (winner_won, winner_total) = (won, total)

        if loser_id not in self.matches_won_lost:
            self.matches_won_lost[loser_id] = (0, 0)
            (loser_won, loser_total) = (0, 0)
        else:
            (won, total) = self.matches_won_lost[loser_id]
            self.matches_won_lost[loser_id] = (won, total + 1)
            (loser_won, loser_total) = (won, total)

        if winner_total != 0:
            winner_perc = winner_won * 100 / winner_total
        else:
            winner_perc = 0

        if loser_total != 0:
            loser_perc = loser_won * 100 / loser_total
        else:
            loser_perc = 0

        self.Features[index][firstPlayerFeature] = winner_perc
        self.Features[index][secondPlayerFeature] = loser_perc
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = winner_won
        self.Features[index][secondPlayerFeature] = loser_won

        feature_position += 2

        return feature_position

    def setPlayersHeadToHeadFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        if (winner_id, loser_id) not in self.head_to_head:
            self.Features[index][firstPlayerFeature] = 0
            self.Features[index][secondPlayerFeature] = 0
            self.head_to_head[(winner_id, loser_id)] = (1, 0)
            self.head_to_head[(loser_id, winner_id)] = (0, 1)
        else:
            (winner_score, loser_score) = self.head_to_head[(winner_id, loser_id)]
            self.Features[index][firstPlayerFeature] = winner_score - loser_score
            self.Features[index][secondPlayerFeature] = loser_score - winner_score
            self.head_to_head[(winner_id, loser_id)] = (winner_score + 1, loser_score)
            self.head_to_head[(loser_id, winner_id)] = (loser_score, winner_score + 1)

        feature_position += 1

        return feature_position

    def setPlayersHeadToHeadSurfaceFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]
        surface = self.matches.iloc[index, 2]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        if (winner_id, loser_id, surface) not in self.head_to_head_surface:
            self.Features[index][firstPlayerFeature] = 0
            self.Features[index][secondPlayerFeature] = 0
            self.head_to_head_surface[(winner_id, loser_id, surface)] = (1, 0)
            self.head_to_head_surface[(loser_id, winner_id, surface)] = (0, 1)
        else:
            (winner_score, loser_score) = self.head_to_head_surface[(winner_id, loser_id, surface)]
            self.Features[index][firstPlayerFeature] = winner_score - loser_score
            self.Features[index][secondPlayerFeature] = loser_score - winner_score
            self.head_to_head_surface[(winner_id, loser_id, surface)] = (winner_score + 1, loser_score)
            self.head_to_head_surface[(loser_id, winner_id, surface)] = (loser_score, winner_score + 1)

        feature_position += 1

        return feature_position

    def setPlayersCommonOpponentHeadToHeadFeature(self, feature_position, index):
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]

        if index % 2 == 0:
            firstPlayerFeature = feature_position
            secondPlayerFeature = feature_position + self.playerOffset
        else:
            firstPlayerFeature = feature_position + self.playerOffset
            secondPlayerFeature = feature_position

        if winner_id not in self.common_head_to_head:
            self.common_head_to_head[winner_id] = {loser_id: (0, 0)}

        if loser_id not in self.common_head_to_head:
            self.common_head_to_head[loser_id] = {winner_id: (0, 0)}

        if loser_id not in self.common_head_to_head[winner_id]:
            self.common_head_to_head[winner_id][loser_id] = (0, 0)

        if winner_id not in self.common_head_to_head[loser_id]:
            self.common_head_to_head[loser_id][winner_id] = (0, 0)

        (x11, y11) = self.common_head_to_head[winner_id][loser_id]
        (x22, y22) = self.common_head_to_head[loser_id][winner_id]
        self.common_head_to_head[winner_id][loser_id] = (x11 + 1, y11)
        self.common_head_to_head[loser_id][winner_id] = (x22, y22 + 1)

        new_head_to_head_winner = (0, 0)
        new_head_to_head_loser = (0, 0)

        for player_id in self.common_head_to_head[winner_id]:
            if player_id in self.common_head_to_head[loser_id]:
                (temp1, temp2) = self.common_head_to_head[winner_id][player_id]
                t1, t2 = new_head_to_head_winner
                new_head_to_head_winner = t1 + temp1, t2 + temp2

        for player_id in self.common_head_to_head[loser_id]:
            if player_id in self.common_head_to_head[winner_id]:
                (temp1, temp2) = self.common_head_to_head[loser_id][player_id]
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

        self.Features[index][firstPlayerFeature] = temp1 - temp3
        self.Features[index][secondPlayerFeature] = temp3 - temp1
        firstPlayerFeature += 1
        secondPlayerFeature += 1
        self.Features[index][firstPlayerFeature] = temp5 - temp6
        self.Features[index][secondPlayerFeature] = temp6 - temp5

        feature_position += 2

        return feature_position

    def setAllFeatures(self):
        for index in range(self.numOfRows):
            feature_position = 0
            # set players information
            feature_position = self.setPlayersInfoFeature(feature_position, index)
            # set players current and overall statistics
            feature_position = self.setPlayersStatisticsFeature(feature_position, index)
            # set tournament form
            feature_position = self.setPlayersTournamentFormFeature(feature_position, index)
            # set overall form
            feature_position = self.setPlayersOverallFormFeature(feature_position, index)
            # set surface form
            feature_position = self.setPlayersSurfaceFormFeature(feature_position, index)
            # set overall win loss
            feature_position = self.setPlayersOverallWinLossFeature(feature_position, index)
            # set head to head
            feature_position = self.setPlayersHeadToHeadFeature(feature_position, index)
            # set head to head surface
            feature_position = self.setPlayersHeadToHeadSurfaceFeature(feature_position, index)
            # set common opponent head to head
            feature_position = self.setPlayersCommonOpponentHeadToHeadFeature(feature_position, index)
            self.setResult(index)

        np.random.shuffle(self.Features)

    def normalizeData(self):
        # copy of datasets
        x_set = self.getXSet()
        for index in range(int(x_set.shape[1]/2)):
            # fit on training data column
            player1_array = x_set[:, index]
            player2_array = x_set[:, (index + self.playerOffset)]
            data_array = np.concatenate(player1_array, player2_array)
            maxValue = np.max(data_array)
            minValue = np.min(data_array)
            if minValue == maxValue:
                print(index)

            for row in range(x_set.shape[0]):
                if minValue == maxValue:
                    x_set[row, index] = 1/data_array.shape[0]
                    x_set[row, (index + self.playerOffset)] = 1/data_array.shape[0]
                else:
                    x_set[row, index] = (x_set[row, index] - minValue)/(maxValue - minValue)
                    x_set[row, (index + self.playerOffset)] = (x_set[row, (index + self.playerOffset)] - minValue) / (maxValue - minValue)

        return x_set


features1 = FeaturesExtraction(absolute_path_to_data)
features1.setAllFeatures()

np.savetxt("training_data.csv", features1.getXSet(), delimiter=",")

x = features1.getXSet()
y = features1.getYSet()

X_train, X_test, Y_train, Y_test = train_test_split(x, y)

model = LogisticRegression(multi_class='ovr', random_state=0, solver='liblinear', penalty='l2').fit(X_train, Y_train)
importance = model.coef_
for i, v in enumerate(importance):
    print("Feature:", v)

# predictions = model.predict(X_test)
# for input, prediction, test in zip(X_test, predictions, Y_test):
#     if prediction != test:
#         print(input, 'has been classified as ', prediction, 'and should be ', test)

# np.savetxt("training_data.csv", X_train, delimiter=",")

print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))

# it was 59.5% with 9 features which were just stats,
# with ranking, seed, ranking points, basically values known pre-match which describe the player - improved to 62.5%
#62.654205607476634 - l1 penalty
# lbfgs - l2 - 62.57943925233645

# clf = svm.SVC(kernel='linear')
# clf.fit(features.X_train, features.Y_train)
#
# svm.SVC(kernel='linear')
# svm.SVC(kernel='rbf')
# svm.SVC(kernel='sigmoid')
# svm.SVC(kernel='poly')
# # logistic regression - penaltystr, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
#
# predictions_svm_svc = clf.predict(features.X_test)
#
# right = 0
#
# for i in range(2*features.testSetSize):
#     if features.Y_test[i] == predictions_svm_svc[i]:
#         right = right + 1
#
# print(2*features.testSetSize)
# print(right)
# print(right*100/(2*features.testSetSize))


import pandas as pd
import numpy as np
from pre_processing.data_pre_processing import DataProcessing
from sklearn.linear_model import LogisticRegression


absolute_path_to_data = "C:\\Users\\Sara\\PycharmProjects\\tennis_atp"


class FeaturesExtraction:
    def __init__(self, dirname):
        self.dataProcessing = DataProcessing(dirname)
        self.dataProcessing.getCleanData()
        self.matches = self.dataProcessing.matches
        self.numOfRows = self.matches.shape[0]
        self.numOfCols = self.matches.shape[1]
        self.testSetSize = self.getTestSetSize()
        self.X_inputs = np.zeros((2*(self.numOfRows - self.testSetSize), 30))
        self.Y_inputs = np.zeros((2 * (self.numOfRows - self.testSetSize)))
        self.X_prediction = np.zeros((2 * self.testSetSize, 30))
        self.Y_prediction = np.zeros((2 * self.testSetSize))
        self.career_stats_winner = np.zeros((self.numOfRows, 9))
        self.career_stats_loser = np.zeros((self.numOfRows, 9))
        self.career_stats_winner_total = np.zeros((self.numOfRows, 9))
        self.career_stats_loser_total = np.zeros((self.numOfRows, 9))

        x = 0
        self.player_stats_overall_sum = [dict() for x in range(9)]
        self.player_stats_overall_count = [dict() for x in range(9)]

    def getTestSetSize(self):
        count_test = 0
        for index, row in self.matches.iterrows():
            year = int(str(row['tourney_date'])[0:4])
            if year == 2019:
                count_test += 1
            elif year == 2020:
                count_test += 1
            elif year == 2021:
                count_test += 1

        return count_test

    def getTrainSet(self):
        #train_Set = self.dataProcessing.matches.sample(frac=0.8, random_state=1)
        train_Set = self.dataProcessing.matches.head(88412)
        return train_Set

    def getTestSet(self):
        test_Set = pd.concat([self.dataProcessing.matches, self.trainSet]).drop_duplicates(keep=False)
        return test_Set

    def setTourneyLevelFeature(self, position):
        tourneyLevels_dict = {'C': 1.0, 'A': 2.0, 'G': 4.0, 'M': 3.0, 'F': 0.0}

        for index, row in self.trainSet.iterrows():
            self.X_inputs[2*index][position] = tourneyLevels_dict[row['tourney_level']]
            self.X_inputs[2*index + 1][position] = tourneyLevels_dict[row['tourney_level']]

        for index, row in self.testSet.iterrows():
            self.X_prediction[2*index][position] = tourneyLevels_dict[row['tourney_level']]
            self.X_prediction[2*index + 1][position] = tourneyLevels_dict[row['tourney_level']]

    def setPlayersEntryFeature(self, position):
        playersEntries_dict = self.dataProcessing.getPlayersEntryDict()
        for index, key in enumerate(playersEntries_dict):
            playersEntries_dict[key] = float(index)

        for index, row in self.trainSet.iterrows():
            self.X_inputs[2*index][position] = playersEntries_dict[' '] if row['winner_entry'] is np.nan else playersEntries_dict[row['winner_entry']]
            self.X_inputs[2*index + 1][position] = playersEntries_dict[' '] if row['loser_entry'] is np.nan else playersEntries_dict[row['loser_entry']]

        for index, row in self.testSet.iterrows():
            self.X_prediction[2*index][position] = playersEntries_dict[' '] if row['winner_entry'] is np.nan else playersEntries_dict[row['winner_entry']]
            self.X_prediction[2*index + 1][position] = playersEntries_dict[' '] if row['loser_entry'] is np.nan else playersEntries_dict[row['loser_entry']]

    def setResult(self, index):
        if index < self.numOfRows - self.testSetSize:
            self.Y_inputs[2 * index] = 1
            self.Y_inputs[2 * index + 1] = 0
        else:
            row = int(2 * index - 2 * int(self.numOfRows) + 2 * int(self.testSetSize))
            self.Y_prediction[row] = 1
            self.Y_prediction[row + 1] = 0

    def setPlayersInfoFeature(self, feature_position, index):
        # 8-winner_seed, 9-winner_entry, 10-winner_name, 11-winner_hand, 12-winner_ht, 13-winner_ioc, 14-winner_age
        # 16-loser_seed, 17-loser_entry, 18-loser_name, 19-loser_hand, 20-loser_ht, 21-loser_ioc, 22-loser_age
        for j in range(8, 15):
            # exclude name and country
            if j != 9 and j != 10 and j != 13:
                # for train data
                if index < self.numOfRows - self.testSetSize:
                    self.X_inputs[2 * index][feature_position] = float(self.matches.iloc[index, j])
                    self.X_inputs[2 * index][feature_position + 6] = float(self.matches.iloc[index, j+8])
                    self.X_inputs[2 * index + 1][feature_position] = float(self.matches.iloc[index, j+8])
                    self.X_inputs[2 * index + 1][feature_position + 6] = float(self.matches.iloc[index, j])

                    feature_position += 1
                else:
                    # for test data
                    row = int(2 * index - 2 * int(self.numOfRows) + 2 * int(self.testSetSize))
                    self. X_prediction[row][feature_position] = float(self.matches.iloc[index, j])
                    self.X_prediction[row][feature_position + 6] = float(self.matches.iloc[index, j+8])
                    self.X_prediction[row + 1][feature_position] = float(self.matches.iloc[index, j+8])
                    self.X_prediction[row + 1][feature_position + 6] = float(self.matches.iloc[index, j])

                    feature_position += 1

        # rank and rank_points
        if index < self.numOfRows - self.testSetSize:
            self.X_inputs[2 * index][feature_position] = float(self.matches.iloc[index, 45])
            self.X_inputs[2 * index][feature_position + 6] = float(self.matches.iloc[index, 47])
            self.X_inputs[2 * index + 1][feature_position] = float(self.matches.iloc[index, 47])
            self.X_inputs[2 * index + 1][feature_position + 6] = float(self.matches.iloc[index, 45])

            feature_position += 1

            self.X_inputs[2 * index][feature_position] = float(self.matches.iloc[index, 46])
            self.X_inputs[2 * index][feature_position + 6] = float(self.matches.iloc[index, 48])
            self.X_inputs[2 * index + 1][feature_position] = float(self.matches.iloc[index, 48])
            self.X_inputs[2 * index + 1][feature_position + 6] = float(self.matches.iloc[index, 46])

            feature_position += 1
        else:
            # for test data
            row = int(2 * index - 2 * int(self.numOfRows) + 2 * int(self.testSetSize))
            self.X_prediction[row][feature_position] = float(self.matches.iloc[index, 45])
            self.X_prediction[row][feature_position + 6] = float(self.matches.iloc[index, 47])
            self.X_prediction[row + 1][feature_position] = float(self.matches.iloc[index, 47])
            self.X_prediction[row + 1][feature_position + 6] = float(self.matches.iloc[index, 45])

            feature_position += 1

            self.X_prediction[row][feature_position] = float(self.matches.iloc[index, 46])
            self.X_prediction[row][feature_position + 6] = float(self.matches.iloc[index, 48])
            self.X_prediction[row + 1][feature_position] = float(self.matches.iloc[index, 48])
            self.X_prediction[row + 1][feature_position + 6] = float(self.matches.iloc[index, 46])

            feature_position += 1

        feature_position += 6
        return feature_position

    def setPlayersStatisticsFeature(self, feature_position, index):
        # 27-w_ace, 28-w_df, 29-w_svpt, 30-w_1stIn, 31-w_1stWon, 32-w_2ndWon, 33-w_SvGms, 34-w_bpSaved, 35-w_bpFaced
        # 36-l_ace, 37-l_df, 38-l_svpt, 39-l_1stIn, 40-l_1stWon, 41-l_2ndWon, 42-l_SvGms, 43-l_bpSaved, 44-l_bpFaced
        winner_id = self.matches.iloc[index, 7]
        loser_id = self.matches.iloc[index, 15]
        for j in range(0, 9):
            if winner_id not in self.player_stats_overall_count[j]:
                self.career_stats_winner[index][j] = 0
                self.career_stats_winner_total[index][j] = 0

                self.player_stats_overall_count[j][winner_id] = 1
                self.player_stats_overall_sum[j][winner_id] = int(self.matches.iloc[index, j+27])
            else:
                self.career_stats_winner[index][j] = self.player_stats_overall_sum[j][winner_id] / self.player_stats_overall_count[j][winner_id]
                self.career_stats_winner_total[index][j] = self.player_stats_overall_sum[j][winner_id]

                self.player_stats_overall_count[j][winner_id] = self.player_stats_overall_count[j][winner_id] + 1
                self.player_stats_overall_sum[j][winner_id] = self.player_stats_overall_sum[j][winner_id] + int(self.matches.iloc[index, j+27])

            if loser_id not in self.player_stats_overall_count:
                self.career_stats_loser[index][j] = 0
                self.career_stats_loser_total[index][j] = 0

                self.player_stats_overall_count[j][loser_id] = 1
                self.player_stats_overall_sum[j][loser_id] = int(self.matches.iloc[index, j + 36])
            else:
                self.career_stats_loser[index][j] = self.player_stats_overall_sum[j][loser_id] / self.player_stats_overall_count[loser_id][j]
                self.career_stats_loser_total[index][j] = self.player_stats_overall_sum[j][loser_id]

                self.player_stats_overall_count[j][loser_id] = self.player_stats_overall_count[j][loser_id] + 1
                self.player_stats_overall_sum[j][loser_id] = self.player_stats_overall_sum[j][loser_id] + int(self.matches.iloc[index, j + 36])

            if index < self.numOfRows - self.testSetSize:
                self.X_inputs[2 * index][feature_position] = self.career_stats_winner[index][j] - self.career_stats_loser[index][j]
                self.X_inputs[2 * index + 1][feature_position] = self.career_stats_loser[index][j] - self.career_stats_winner[index][j]
                self.X_inputs[2 * index][feature_position + 9] = self.career_stats_winner_total[index][j] - self.career_stats_loser_total[index][j]
                self.X_inputs[2 * index + 1][feature_position + 9] = self.career_stats_loser_total[index][j] - self.career_stats_winner_total[index][j]
                feature_position += 1
            else:
                row = int(2 * index - 2 * int(self.numOfRows) + 2 * int(self.testSetSize))
                self.X_prediction[row][feature_position] = self.career_stats_winner[index][j] - self.career_stats_loser[index][j]
                self.X_prediction[row + 1][feature_position] = self.career_stats_loser[index][j] - self.career_stats_winner[index][j]
                self.X_prediction[row][feature_position + 9] = self.career_stats_winner_total[index][j] - self.career_stats_loser_total[index][j]
                self.X_prediction[row + 1][feature_position + 9] = self.career_stats_loser_total[index][j] - self.career_stats_winner_total[index][j]
                feature_position += 1

        feature_position += 9

        return feature_position

    def setAllFeatures(self):
        for index in range(self.numOfRows):
            feature_position = 0
            # set players information
            feature_position = self.setPlayersInfoFeature(feature_position, index)
            # set players current and overall statistics
            feature_position = self.setPlayersStatisticsFeature(feature_position, index)
            self.setResult(index)

    def getXInputs(self):
        return self.X_inputs

    def getYInputs(self):
        return self.Y_inputs

    def getXPrediction(self):
        return self.X_prediction

    def getYPrediction(self):
        return self.Y_prediction


features = FeaturesExtraction(absolute_path_to_data)
features.setAllFeatures()

clf = LogisticRegression(multi_class='ovr', random_state=0, solver='liblinear', penalty='l2').fit(features.X_inputs, features.Y_inputs)
importance = clf.coef_
for i, v in enumerate(importance):
    print("Feature:", v)
training_prediction = clf.predict_proba(features.X_prediction)

np.savetxt("training_data.csv", features.X_inputs, delimiter=",")

total = features.testSetSize
right = 0

print(training_prediction.shape)

for i in range(total):
    (a, b) = training_prediction[2*i]
    (c, d) = training_prediction[2*i+1]

    if a < b and c > d:
        predicted = 1
    else:
        if a > b and c < d:
            predicted = 0
        else:
            if a + d < b + c:
                predicted = 1
            else:
                predicted = 0

    if features.Y_prediction[2*i] == predicted:
        right = right + 1

print(total)
print(right)
print(right*100/total)

training_prediction_1 = clf.predict(features.X_inputs)

total = 2*(features.numOfRows - features.testSetSize)
right = 0

for i in range(2*(features.numOfRows - features.testSetSize)):
    if features.Y_inputs[i] == training_prediction_1[i]:
        right = right + 1

print(total)
print(right)
print(right*100/total)



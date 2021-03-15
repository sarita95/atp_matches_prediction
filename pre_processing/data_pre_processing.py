import numpy as np
import glob
import pandas as pd


absolute_path_to_data = "C:\\Users\\Sara\\Desktop\\master\\atp_matches_prediction\\data"


class DataProcessing:
    def __init__(self, dirname):
        self.dirname = dirname
        self.matches = self.getAllMatches()

    def readATPMatches(self):
        """Reads ATP matches"""
        allFiles = glob.glob(self.dirname + "/atp_matches_" + "????.csv")
        #matches = pd.DataFrame()
        container = list()
        for file in allFiles:
            df = pd.read_csv(file,
                             index_col=None,
                             header=0)
            container.append(df)
        atpMatches = pd.concat(container)
        return atpMatches

    def readFMatches(self):
        """Reads ITF future matches"""
        allFiles = glob.glob(self.dirname + "/atp_matches_futures_" + "????.csv")
        #atches = pd.DataFrame()
        container = list()
        for file in allFiles:
            df = pd.read_csv(file,
                             index_col=None,
                             header=0)
            container.append(df)
        fMatches = pd.concat(container)
        return fMatches

    def readChall_QATPMatches(self):
        """reads Challenger level + ATP Q matches"""
        allFiles = glob.glob(self.dirname + "/atp_matches_qual_chall_" + "????.csv")
        #matches = pd.DataFrame()
        container = list()
        for file in allFiles:
            df = pd.read_csv(file,
                             index_col=None,
                             header=0)
            container.append(df)
        qaMatches = pd.concat(container)
        return qaMatches

    def checkTourneyId(self):
        """Sanity checking tourney_id"""
        numOfNan = self.matches[self.matches['tourney_id'].isnull()].shape[0]
        print("Sanity checking tourney_id: " + str(numOfNan))
        self.matches.dropna(subset=['tourney_id'], inplace=True)

    def checkTourneyName(self):
        """Sanity checking tourney_name"""
        numOfNan = self.matches[self.matches['tourney_name'].isnull()].shape[0]
        print("Sanity checking tourney_name: " + str(numOfNan))
        self.matches.dropna(subset=['tourney_name'], inplace=True)

    def checkSurface(self):
        """Sanity checking surface"""
        numOfNan = self.matches[self.matches['surface'].isnull()].shape[0]
        print("Sanity checking surface: " + str(numOfNan))
        self.matches.dropna(subset=['surface'], inplace=True)

    def getSurfaceDict(self):
        """Get surfaces"""
        surfaces = self.matches.groupby(['surface']).size().reset_index(
            name="surface_count").sort_values(by='surface_count', ascending=False)
        surfaces_dict = dict(surfaces.values)
        print("Surfaces: ")
        print(surfaces_dict)
        return surfaces_dict

    def checkDrawSize(self):
        """Sanity checking draw_size"""
        numOfNan = self.matches[self.matches['draw_size'].isnull()].shape[0]
        print("Sanity checking draw_size: " + str(numOfNan))
        self.matches.dropna(subset=['draw_size'], inplace=True)

    def getDrawSizeDict(self):
        """Get draw_size"""
        draw_sizes = self.matches.groupby(['draw_size']).size().reset_index(
            name="draw_size_count").sort_values(by='draw_size', ascending=False)
        draw_size_dict = dict(draw_sizes.values)
        print("Draw_size: ")
        print(draw_size_dict)
        return draw_size_dict

    def checkTourneyLevel(self):
        """Sanity checking tourney_level"""
        numOfNan = self.matches[self.matches['tourney_level'].isnull()].shape[0]
        print("Sanity checking tourney_level: " + str(numOfNan))
        self.matches.dropna(subset=['tourney_level'], inplace=True)

    def getTourneyLevelDict(self):
        """Get tourney_level"""
        tourney_levels = self.matches.groupby(['tourney_level']).size().reset_index(
            name="tourney_level_count").sort_values(by='tourney_level_count', ascending=False)
        tourney_level_dict = dict(tourney_levels.values)
        print("Tourney_levels: ")
        print(tourney_level_dict)
        return tourney_level_dict

    def checkPlayerId(self):
        """Sanity checking winner_id and loser_id"""
        numOfNan = self.matches[self.matches['winner_id'].isnull() | self.matches['loser_id'].isnull()].shape[0]
        print("Sanity checking winner_id and loser_id: " + str(numOfNan))
        self.matches.dropna(subset=['winner_id'], inplace=True)
        self.matches.dropna(subset=['loser_id'], inplace=True)

    def checkPlayerSeed(self):
        """Sanity checking winner_seed and loser_seed"""
        self.matches[["winner_seed", "loser_seed"]] = self.matches[["winner_seed", "loser_seed"]].apply(pd.to_numeric, errors='coerce')
        numOfNan = self.matches[self.matches['winner_seed'].isnull() | self.matches['loser_seed'].isnull()].shape[0]
        print("Sanity checking winner_seed and loser_seed: " + str(numOfNan))

        """Fill NaN players seed with avg value"""
        winner_seed = pd.DataFrame(self.matches, columns=['winner_seed'])
        winner_seed.columns = ['players_seed']
        loser_seed = pd.DataFrame(self.matches, columns=['loser_seed'])
        loser_seed.columns = ['players_seed']
        players_seed = winner_seed.append(loser_seed)
        avg_player_seed = str(round(players_seed["players_seed"].mean(skipna=True)))

        self.matches['winner_seed'] = self.matches['winner_seed'].fillna(avg_player_seed)
        self.matches['loser_seed'] = self.matches['loser_seed'].fillna(avg_player_seed)

        """Fill NaN players seed with max value"""
        # max_player_seed = int(players_seed["players_seed"].max(skipna=True))
        #
        # self.matches['winner_seed'] = self.matches['winner_seed'].fillna(max_player_seed)
        # self.matches['loser_seed'] = self.matches['loser_seed'].fillna(max_player_seed)

    def getPlayersEntryDict(self):
        """Get players_entry"""
        self.matches['winner_entry'] = self.matches['winner_entry'].fillna(" ")
        self.matches['loser_entry'] = self.matches['loser_entry'].fillna(" ")
        winner_entries = pd.DataFrame(self.matches, columns=['winner_entry'])
        winner_entries.columns = ['players_entry']
        loser_entries = pd.DataFrame(self.matches, columns=['loser_entry'])
        loser_entries.columns = ['players_entry']
        players_entries = winner_entries.append(loser_entries)
        players_entries = players_entries.groupby(['players_entry']).size().reset_index(
            name="players_entry_count").sort_values(by='players_entry_count', ascending=False)
        players_entry_dict = dict(players_entries.values)
        print("Players_entry:")
        print(players_entry_dict)
        return players_entry_dict

    def checkPlayersHand(self):
        """Sanity checking winner_hand and loser_hand"""
        numOfNan = self.matches[self.matches['winner_hand'].isnull() | self.matches['loser_hand'].isnull()].shape[0]
        print("Sanity checking winner_hand and loser_hand: " + str(numOfNan))
        self.matches.dropna(subset=['winner_hand'], inplace=True)
        self.matches.dropna(subset=['loser_hand'], inplace=True)
        self.matches['winner_hand'].apply(lambda x: x.upper())
        self.matches['loser_hand'].apply(lambda x: x.upper())
        self.matches['winner_hand'] = np.where(self.matches['winner_hand'] == 'L', 1, self.matches['winner_hand'])
        self.matches['winner_hand'] = np.where(self.matches['winner_hand'] == 'R', 0, self.matches['winner_hand'])
        self.matches['loser_hand'] = np.where(self.matches['loser_hand'] == 'L', 1, self.matches['loser_hand'])
        self.matches['loser_hand'] = np.where(self.matches['loser_hand'] == 'R', 0, self.matches['loser_hand'])

    def checkPlayerHeight(self):
        """Sanity checking winner_ht and loser_ht"""
        numOfNan = self.matches[self.matches['winner_ht'].isnull() | self.matches['loser_ht'].isnull()].shape[0]
        print("Sanity checking winner_ht and loser_ht: " + str(numOfNan))
        """Fill NaN players height with -1"""
        self.matches['winner_ht'] = self.matches['winner_ht'].fillna(-1)
        self.matches['loser_ht'] = self.matches['loser_ht'].fillna(-1)

    def checkPlayerAge(self):
        """Sanity checking winner_age and loser_age"""
        numOfNan = self.matches[self.matches['winner_age'].isnull() | self.matches['loser_age'].isnull()].shape[0]
        print("Sanity checking winner_age and loser_age: " + str(numOfNan))

        """Fill NaN players age with avg value"""
        winner_age = pd.DataFrame(self.matches, columns=['winner_age'])
        winner_age.columns = ['players_age']
        loser_age = pd.DataFrame(self.matches, columns=['loser_age'])
        loser_age.columns = ['players_age']
        players_age = winner_age.append(loser_age)
        avg_player_age = str(round(players_age["players_age"].mean(skipna=True)))

        self.matches['winner_age'] = self.matches['winner_age'].fillna(avg_player_age)
        self.matches['loser_age'] = self.matches['loser_age'].fillna(avg_player_age)

    def checkPlayerRank(self):
        """Sanity checking winner_rank and loser_rank"""
        numOfNan = self.matches[self.matches['winner_rank'].isnull() | self.matches['loser_rank'].isnull()].shape[0]
        print("Sanity checking winner_rank and loser_rank: " + str(numOfNan))

        """Fill NaN players rank with 2000 which represents really high rank"""
        self.matches['winner_rank'] = self.matches['winner_rank'].fillna(2000)
        self.matches['loser_rank'] = self.matches['loser_rank'].fillna(2000)

    def checkPlayerRankPoints(self):
        """Sanity checking winner_rank_points and loser_rank_points"""
        numOfNan = self.matches[self.matches['winner_rank_points'].isnull() | self.matches['loser_rank_points'].isnull()].shape[0]
        print("Sanity checking winner_rank_points and loser_rank_points: " + str(numOfNan))

        """Fill NaN players rank points with 0"""
        self.matches['winner_rank_points'] = self.matches['winner_rank_points'].fillna(0)
        self.matches['loser_rank_points'] = self.matches['loser_rank_points'].fillna(0)

    def checkScore(self):
        """Sanity checking score"""
        #self.matches['score'].apply(lambda x: x.lower())
        numOfNan = self.matches[self.matches['score'].isnull() | self.matches['score'] == "w/o"].shape[0]
        print("Sanity checking score: " + str(numOfNan))
        self.matches.dropna(subset=['score'], inplace=True)
        self.matches = self.matches[self.matches.score.str.lower() != "w/o"]

    def checkBestOf(self):
        """Sanity checking best_of"""
        numOfNan = self.matches[self.matches['best_of'].isnull()].shape[0]
        print("Sanity checking best_of: " + str(numOfNan))

        """Fill Nan best_of with 5 for Grand slams and with 3 for others"""
        self.matches['best_of'] = np.where((self.matches['tourney_level'] == 'G') & (self.matches['best_of'].isnull()), 5, self.matches['best_of'])
        self.matches['best_of'] = np.where(self.matches['best_of'].isnull(), 3, self.matches['best_of'])

    def getRoundsDict(self):
        """Get rounds"""
        rounds = self.matches.groupby(['round']).size().reset_index(
            name="rounds_count").sort_values(by='round', ascending=False)
        rounds_dict = dict(rounds.values)
        print("Rounds: ")
        print(rounds_dict)
        return rounds_dict

    def checkMinutes(self):
        """Sanity checking minutes"""
        numOfNan = self.matches[self.matches['minutes'].isnull()].shape[0]
        print("Sanity checking minutes: " + str(numOfNan))
        self.matches.dropna(subset=['minutes'], inplace=True)
        #think about filling nan values with avg

    def checkMatchStatistic(self):
        """Sanity checking match statistic"""
        numOfNan = self.matches[self.matches['w_ace'].isnull() | self.matches['w_df'].isnull() |
                                self.matches['w_svpt'].isnull() | self.matches['w_1stIn'].isnull() |
                                self.matches['w_1stWon'].isnull() | self.matches['w_2ndWon'].isnull() |
                                self.matches['w_SvGms'].isnull() | self.matches['w_bpSaved'].isnull() |
                                self.matches['w_bpFaced'].isnull()].shape[0]

        numOfNan += self.matches[self.matches['l_ace'].isnull() | self.matches['l_df'].isnull() |
                                 self.matches['l_svpt'].isnull() | self.matches['l_1stIn'].isnull() |
                                 self.matches['l_1stWon'].isnull() | self.matches['l_2ndWon'].isnull() |
                                 self.matches['l_SvGms'].isnull() | self.matches['l_bpSaved'].isnull() |
                                 self.matches['l_bpFaced'].isnull()].shape[0]

        print("Sanity checking match statistic: " + str(numOfNan))

        self.matches.dropna(
            subset=['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced'],
            inplace=True)

        self.matches.dropna(
            subset=['l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'],
            inplace=True)

    def dropDavisCup(self):
        """Delete Davis Cup matches"""
        numOfDavisCup = self.matches[self.matches['tourney_level'] == 'D'].shape[0]
        print("Delete Davis Cup matches: " + str(numOfDavisCup))
        self.matches = self.matches[self.matches.tourney_level != 'D']

    def getAllMatches(self):
        atp_matches = self.readATPMatches()
        qa_matches = self.readChall_QATPMatches()
        f_matches = self.readFMatches()
        allMatches = pd.concat([atp_matches, qa_matches, f_matches])
        return allMatches

    def getCleanData(self):
        print("Number of all matches: " + str(self.matches.shape[0]))
        self.checkTourneyId()
        self.checkTourneyName()
        self.checkSurface()
        self.checkDrawSize()
        self.checkTourneyLevel()
        self.dropDavisCup()
        self.checkPlayerId()
        self.checkPlayerSeed()
        self.checkPlayersHand()
        self.checkPlayerHeight()
        self.checkPlayerAge()
        self.checkPlayerRank()
        self.checkPlayerRankPoints()
        self.checkScore()
        self.checkBestOf()
        self.checkMinutes()
        self.checkMatchStatistic()
        print("Number of matches after cleaning: " + str(self.matches.shape[0]))

    def getMatches(self):
        return self.matches

    def writeMatches(self):
        self.matches.to_csv("atp_matches_all.csv", index=False, encoding="utf-8-sig", header=True)


data_processing = DataProcessing(absolute_path_to_data)
data_processing.getCleanData()
data_processing.writeMatches()
data_processing.getDrawSizeDict()
data_processing.getPlayersEntryDict()
data_processing.getRoundsDict()
data_processing.getSurfaceDict()
data_processing.getTourneyLevelDict()



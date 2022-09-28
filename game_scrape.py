import concurrent.futures
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sqlite3
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import threading
import numpy as np
from sqlalchemy import create_engine
import os
import shutil
import glob
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import stanfordnlp
import stanza
from datetime import datetime
# nltk.download('punkt')

#thread_storage = threading.local()


class MatchData:

    def __init__(self):
        self.soup = None
        self.response = None
        self.df_lineup = None

        self.df = pd.DataFrame(columns=[
            'match_id',
            'home_team',
            'away_team',
            'competition',
            'round',
            'season',
            'type',
            'time',
            'text',
        ])
        self.data_df = pd.DataFrame(columns=['match_id',
                                        'home_team',
                                        'away_team',
                                        'home_lineup',
                                        'away_lineup',
                                        'league_season',
                                        'type',
                                        'time',
                                        'text',
                                        'date'
                                        ])
        with open('matches_w_comments.pkl', 'rb') as p:
            self.sites = pickle.load(p)

    def get_session(self):

        if not hasattr(thread_storage, "session"):
            thread_storage.session = requests.Session()
        return thread_storage.session

    def parse_page(self):

        url = "https://www.espn.com/soccer/commentary?gameId={}".format(self.match_id)
        session = self.get_session()
        self.response = session.get(url, timeout=15)
        html = '{}.html'.format(self.match_id)

        with open('match_data/{}'.format(html), 'wb') as fh:
            fh.write(self.response.content)
        time.sleep(0.2)

    def find_matches_w_comments(self):

        count = 0
        matches_w_comments = []
        for match in self.sites:
            print(count)
            with open("match_data/{}.html".format(match)) as fp:
                self.soup = BeautifulSoup(fp, 'html.parser')
                comments = self.soup.find(attrs={'data-id': re.compile("^comment")})
                if comments is not None:
                    matches_w_comments.append(match)
            count += 1
        with open('matches_w_comments.pkl', 'wb') as f:
            pickle.dump(matches_w_comments, f)

    def get_match_events(self, match_id):
        self.match_id = match_id
        game_df = pd.DataFrame(columns=['match_id',
                                        'home_team',
                                        'away_team',
                                        'competition',
                                        'round',
                                        'type',
                                        'time',
                                        'text',
                                        'date'
                                        ])
        with open("match_data/{}.html".format(self.match_id)) as fp:
            self.soup = BeautifulSoup(fp, 'html.parser')

            try:
                league = self.soup.find(class_='game-details header').string.strip()
                date = datetime.strptime(self.soup.find('title').string.split('-', -1)[2].strip(),
                                         '%B %d, %Y').strftime('%Y-%m-%d')
                #date = self.soup.find(attrs={'property': 'og:title'})['content']

                if ',' in league:
                    comp = league.split(',')
                    competition = comp[0]
                    round = comp[1]

                else:
                    competition = league
                    round = 'None'
                # the home team is listed as away in espn.com
                home_team = self.soup.find(class_='competitors sm-score').find(class_=re.compile("^team away")).find(
                    class_='long-name').string.strip()
                away_team = self.soup.find(id='gamepackage-matchup-wrap--soccer').find(class_=re.compile("^team home")).find(
                    class_='long-name').string.strip()
                comments = self.soup.find_all(attrs={'data-id': re.compile("^comment")})
                # home_goals = self.soup.find_all(class_=re.compile("^game-details footer soccer post"))[0] \
                #     .find(class_='goal icon-font-before icon-soccer-ball-before icon-soccerball').find_all('span')

                for i in range(len(comments)):
                    game_df.loc[i, 'type'] = comments[i]['data-type']
                    time = comments[i].find(class_='time-stamp').string

                    if "+" in time:
                        additional_time = time.split("+")
                        main_time = int(additional_time[0][:-1])
                        stoppage_time = int(additional_time[1][:-1]) / 100
                        game_df.loc[i, 'time'] = main_time + stoppage_time

                    elif time == "-":
                        game_df.loc[i, 'time'] = 0

                    else:
                        game_df.loc[i, 'time'] = int(time[:-1])

                    if type(comments[i].find(class_='game-details').string) != type(None):
                        game_df.loc[i, 'text'] = comments[i].find(class_='game-details').string.strip()

                    else:
                        game_df.loc[i, 'text'] = comments[i].find(class_='game-details').string

                game_df['match_id'] = int(self.match_id)
                game_df['home_team'] = home_team
                game_df['away_team'] = away_team
                game_df['competition'] = competition
                game_df['round'] = round
                game_df['date'] = date
                # for h_a in ['home', 'away']:
                #     game_df['{}_score_TOP'.format(h_a)] = (
                #                 (game_df['type'] == 'goal') & (game_df['team.id'] == team)).astype(
                #         int).cumsum().shift(1)
            except:
                print(self.match_id)
        return game_df

    def get_formations(self, match_id):
        self.match_id = match_id
        lineup_df = pd.DataFrame(columns=[
            'match_id',
            'home_players',
            'away_players',
            'home_starter_sub',
            'away_starter_sub'
        ])
        with open("match_data/{}.html".format(self.match_id)) as fp:
            self.soup = BeautifulSoup(fp, 'html.parser')
            try:
                for num, home_away in enumerate(['home','away']):
                    team_formation_holder = self.soup.find_all(attrs={'data-behavior': 'table_accordion'})[
                        num].find_all(
                        attrs={'colspan': '3'})
                    team_players_holder = [[name.string.strip() for name in team_formation_holder[i].find_all('a')]
                                           for i in range(0, 11)]
                    team_players = [i for sublist in team_players_holder for i in sublist]
                    team_starters_holder = [[name.string.strip() for name in team_formation_holder[i].find('a')]
                                            for i in range(0, 11)]
                    team_starters = [i for sublist in team_starters_holder for i in sublist]
                    team_substitutes = [i for i in team_players if i not in team_starters]
                    lineup_df['{}_players'.format(home_away)] = pd.Series(team_players)
                    lineup_df['{}_starter_sub'.format(home_away)] = np.where(lineup_df['{}_players'.format(home_away)]
                                                                             .isin(team_starters), 'starter','substitute')
                lineup_df['match_id'] = self.match_id
            except:
                print(self.match_id)
            return lineup_df

    def get_score_and_goals(self, match_id):
        self.match_id = match_id
        goal_df = pd.DataFrame(columns=['match_id',
                                        'away_team',
                                        'home_team',
                                        'goal_scorer',
                                        'goal_time',
                                        'scoring_team',
                                        'away_score',
                                        'home_score'
                                        ])
        with open("match_data/{}.html".format(self.match_id)) as fp:
            self.soup = BeautifulSoup(fp, 'html.parser')
            # Define the first row of goal_count so that the score at the beginning of the game is always 0-0
            try:
                goal_count = 0
                goal_df.loc[0, 'goal_scorer'] = np.nan
                goal_df.loc[0, 'scoring_team'] = np.nan
                goal_df.loc[0, 'goal_time'] = 0
                goal_df.loc[0, 'away_score'] = 0
                goal_df.loc[0, 'home_score'] = 0
                goal_df['match_id'] = self.match_id

                # Create for loop to get soup for away and home team
                for mul, team in enumerate(['home', 'away']):
                    # Find the goal time and taker for each team
                    goal_soup = self.soup.find_all(class_=re.compile('^game-details footer soccer'))[0].find_all(
                        class_=re.compile("^team "))[mul].find(attrs={'data-event-type': 'goal'})
                    # If there are goals, go into if, otherwise just skip
                    if goal_soup is not None:
                        # Create for loop to go through each goal event
                        for i, string in enumerate(goal_soup.stripped_strings):
                            # The soup alternates between scorer and goal time, so if even, its a scorer, odd is goal time
                            if i % 2 == 0:
                                # Fill each cell with the scorer ("string") and the scorer ("team")
                                goal_df.loc[goal_count + 1, 'goal_scorer'] = string
                                goal_df.loc[goal_count + 1, 'scoring_team'] = team
                            # If someone scores > 1 goal, the goals are listed in all the same string, so first look for those
                            elif "," in string:
                                # First get rid of every character except for #s, +s, and commas, then split on comma
                                time_list = re.sub("[^0-9+,]", "", string).split(',')
                                # Start a counter to add to cell loc b/c otherwise time with have no goal scorer
                                multi_goal_count = 0
                                # For each time in a multi-goal list
                                for time in time_list:
                                    # Look for + for additional time
                                    if "+" in time:
                                        # Split an added time goal into a list of its regular time and its added time
                                        additional_time = time.split("+")
                                        # Make the regular time into a int
                                        main_time = int(additional_time[0])
                                        # Make the stoppage time into an int / 100
                                        stoppage_time = int(additional_time[1]) / 100
                                        # Make the goal time the main time + 1/100th of the stoppage time to ensure no mistiming
                                        goal_df.loc[goal_count + 1, 'goal_time'] = main_time + stoppage_time

                                        # Look for the goals in multi-goal count > 1
                                        if multi_goal_count >= 1:
                                            # Take the goalscorer from the cell above and bring it into this cell
                                            goal_df.loc[goal_count + 1, 'goal_scorer'] = goal_df.loc[
                                                goal_count, 'goal_scorer']

                                            # Take the scoring team from cell above and bring it into this cell
                                            goal_df.loc[goal_count + 1, 'scoring_team'] = goal_df.loc[
                                                goal_count, 'scoring_team']

                                        # Increase goal counts
                                        multi_goal_count += 1
                                        goal_count += 1

                                    # If goal wasn't scored in stoppage time
                                    else:
                                        # Do the same as above to bring scorer and team down to this cell
                                        goal_df.loc[goal_count + 1, 'goal_time'] = int(time)
                                        if multi_goal_count >= 1:
                                            goal_df.loc[goal_count + 1, 'goal_scorer'] = goal_df.loc[
                                                goal_count, 'goal_scorer']
                                            goal_df.loc[goal_count + 1, 'scoring_team'] = goal_df.loc[
                                                goal_count, 'scoring_team']
                                        multi_goal_count += 1
                                        goal_count += 1

                            # If it is a single goal scored in stoppage time
                            elif "+" in string:
                                additional_time = re.sub("[^0-9+]", "", string).split("+")
                                main_time = int(additional_time[0])
                                stoppage_time = int(additional_time[1]) / 100
                                goal_df.loc[goal_count + 1, 'goal_time'] = main_time + stoppage_time
                                goal_count += 1

                            # It is just a single goal
                            else:
                                goal_df.loc[goal_count + 1, 'goal_time'] = float(re.sub("[^0-9]", "", string))
                                goal_count += 1

                # Set match_id
                goal_df['match_id'] = self.match_id

                # Put in the away and home team names
                goal_df['home_team'] = self.soup.find(class_='competitors sm-score').find_all('span', class_='long-name')[
                    0].string.strip()
                goal_df['away_team'] = self.soup.find(class_='competitors sm-score').find_all('span', class_='long-name')[
                    1].string.strip()

                # Sort goals by the time they were scored
                goal_df = goal_df.sort_values('goal_time')

                # Define the score as the cumulative sum of all prior rows in column + 1
                goal_df['home_score'] = np.where(goal_df['scoring_team'] == 'home', 1, 0).cumsum()
                goal_df['away_score'] = np.where(goal_df['scoring_team'] == 'away', 1, 0).cumsum()
            except:
                print(self.match_id)
            return goal_df

    def combine_data(self, func):

        with multiprocessing.Pool(processes=12) as pool:
            pool_map = pool.map(func, self.sites)
        df = pd.concat(pool_map, axis=0)
        return df

    def create_sql(self, data, db_name):

        engine = create_engine('sqlite:///soccer_statistics.db', echo=False)
        with engine.begin() as connection:
            data = data.to_sql('{}'.format(db_name), con=connection, if_exists='replace', chunksize=10000, method='multi')

    def listdir_nohidden(self, path):
        file =  glob.glob(os.path.join(path, '*'))
        game_ids = [int(x[11:-5]) for x in file]
        game_ids.sort()

        with open('parsed_gameids.pkl', 'wb') as f:
            pickle.dump(game_ids, f)

    def tempo(self):
        counter = 0
        engine = create_engine('sqlite:///penalty.db', echo=False)
        for match_id in self.sites[109000:]:
            with open("match_data/{}.html".format(match_id)) as fp:
                soup = BeautifulSoup(fp, 'html.parser')
                game_df = pd.DataFrame(columns=['type',
                                                'time',
                                                'text',
                                                ])

                league = soup.find(class_='game-details header').string.strip()
                date = soup.find(attrs={'property': 'og:title'})['content']
                league = re.split("\s", league, 1)
                print(match_id, counter)
                print(league)
                counter += 1
                if len(league) > 1:
                    season = league[0]
                    if ',' in league[1]:
                        comp = league[1].split(',')
                        competition = comp[0]
                        round = comp[1]

                    else:
                        competition = league[1]
                        round = 'None'
                else:
                    season = 'None'
                    competition = league[0]

                home_team = soup.find(class_='competitors sm-score').find(class_=re.compile("^team away")).find(
                    class_='long-name').string
                away_team = soup.find(id='gamepackage-matchup-wrap--soccer').find(class_=re.compile("^team home")).find(
                    class_='long-name').string
                comments = soup.find_all(attrs={'data-id': re.compile("^comment")})

                for i in range(len(comments)):
                    game_df.loc[i, 'type'] = comments[i]['data-type']
                    time = comments[i].find(class_='time-stamp').string

                    if "+" in time:
                        additional_time = time.split("+")
                        main_time = int(additional_time[0][:-1])
                        stoppage_time = int(additional_time[1][:-1]) / 100
                        game_df.loc[i, 'time'] = main_time + stoppage_time

                    elif time == "-":
                        game_df.loc[i, 'time'] = 0

                    else:
                        try:
                            game_df.loc[i, 'time'] = int(time[:-1])
                        except ValueError:
                            pass

                    if not comments[i].find(class_='game-details'):
                        game_df.loc[i, 'text'] = comments[i].find(class_='game-details').string.strip()

                    else:
                        game_df.loc[i, 'text'] = comments[i].find(class_='game-details').string
                game_df['match_id'] = match_id
                game_df['home_team'] = home_team
                game_df['away_team'] = away_team
                game_df['competition'] = competition
                game_df['round'] = round
                game_df['season'] = season

                if not game_df.empty:
                    self.df = pd.concat([self.df, game_df])

            with engine.begin() as connection:
                data = self.df.to_sql('{}'.format('match_comments1'), con=connection, if_exists='append', chunksize=5000,
                                   method='multi')

    def read_sql_table(self, query):

        engine = create_engine('sqlite:///soccer_statistics.db', echo=False)
        df = pd.read_sql_query(query, con=engine)
        #df.to_csv('penalties.csv')
        return df

    def nlp_match_commentary(self):
        '''
        Function that takes the match commentary and returns a dataframe with the shot location, outcome, taker, and team
        :return: dataframe
        '''

        df['text'] = df['text'].str.strip()

        # Define function to find certain attributes
        def find_attribute(attributes, text):
            '''
            :return: Each attribute in a line of commentary
            '''
            return next((x for x in attributes if x in text), "")

        # Define the possible penalty foot, location, and outcomes
        feet = ['left foot', 'right foot']
        locations = ['bottom left', 'bottom right', 'top right', 'top left', 'center', 'centre', 'post', 'crossbar',
                     'miss',
                     'lower left', 'lower right', 'upper right', 'upper left', 'middle']
        outcomes = ['convert', 'save', 'miss', 'post', 'bar', 'goal', 'scores']

        var_dict = {'foot': feet,
                    'location': locations,
                    'event_outcome': outcomes
                    }
        # Find the words or phrases in each line of commentary that match the feet, locations, and outcomes, and  store the result
        for key, value in var_dict.items():
            df['{}'.format(key)] = df.apply(lambda x: find_attribute(value, x['text']), axis=1)

        df.reset_index(inplace=True)

        # Replace common grammatical error in data to make full sentence

        target_phrases = [',  right footed shot ', ',  left footed shot ']
        true_phrases = [', his/her right footed shot is ', ', his/her left footed shot is ']

        target = df['text'].str.contains(target_phrases[0]), 'text'
        df.loc[target] = df.loc[target].str.replace(target_phrases[0], true_phrases[0])
        df.loc[target] = df.loc[target].str.replace(target_phrases[1], true_phrases[1])

        # Download Spacy and Tokenize
        nlp = stanza.Pipeline(lang='en', processors={
            'tokenize': 'spacy'})  # spaCy tokenizer is currently only allowed in English pipeline.

        def get_taker_and_team(text):
            '''
            Function parses text of each commentary and returns the taker and team
            :param text:
            :return:
            team = team doing the action
            taker = person doing the action
            :raises:
            IndexError: some commentaries do not contain the team doing the action. These are returned as NaN
            '''
            doc = nlp(text)
            try:
                taker_fw = [word.id for word in doc.sentences[-1].words if (word.deprel == 'nsubj')][0]
                taker = [word.text for word in doc.sentences[-1].words if
                         (word.deprel == 'nsubj' or word.head == taker_fw) and word.deprel != 'appos']
                taker = (' '.join(map(str, taker)))
                team_fw = [word.id for word in doc.sentences[-1].words if (word.deprel == 'appos')][0]
                team = [word.text for word in doc.sentences[-1].words if
                        (word.deprel == 'appos' or word.head == team_fw) and word.deprel != 'punct']
                team = (' '.join(map(str, team)))

            except IndexError:
                taker = np.NAN
                team = np.NAN

            return taker, team

        # Apply function to text
        df[['taker', 'team']] = pd.DataFrame(df['text'].progress_apply(get_taker_and_team).tolist(),
                                             index=df.index)

        # Simplify the types of outcomes
        true_outcome = {'convert': 'goal',
                        'goal': 'goal',
                        'miss': 'miss',
                        'post': 'miss',
                        'bar': 'miss',
                        'save': 'save'}

        true_loc = {'bottom left': 'bottom left',
                    'bottom right': 'bottom right',
                    'top right': 'top right',
                    'top left': 'top left',
                    'center': 'center',
                    'centre': 'center',
                    'post': 'miss',
                    'crossbar': 'miss',
                    'miss': 'miss',
                    'lower left': 'bottom left',
                    'lower right': 'bottom right',
                    'upper right': 'top right',
                    'upper left': 'top left',
                    'middle': 'center'
                    }
        # Change locations to common locations
        df['location'] = df['location'].map(true_loc)
        df['event_outcome'] = df['event_outcome'].map(true_outcome)

        return df

    def main(self):
        #get_sites = self.listdir_nohidden('match_data')
        #temp = self.find_matches_w_comments()
        combine_games = self.combine_data(self.get_match_events)
        print('nextadsglkf;sdlkjdflkgdkgkl;fg;ffjgkfgjf')
        #self.tempo()
        # qry = "SELECT * FROM match_comments1 WHERE type REGEXP '^penalty---'"
        # self.read_sql_table(qry)
        combine_lineup = self.combine_data(self.get_formations)
        create_lineup_database = self.create_sql(combine_lineup, 'lineup')
        get_scores = self.combine_data(self.get_score_and_goals)
        create_goals_database = self.create_sql(get_scores, 'scores')
        create_commentary_database = self.create_sql(combine_games, 'match_commentary')
        return combine_games


if __name__ == '__main__':

    # start_time = time.time()
    # pd.set_option('display.max_columns', None)
    # with ProcessPoolExecutor() as executor:
    #     start = time.time()
    #     futures = [executor.submit(MatchData().find_data, url) for url in sites]
    #     results = []
    #     for result in as_completed(futures):
    #         results.append(result)
    #         print(results)
    #
    # duration = time.time() - start_time
#     print(f"The program run for {duration} seconds")+
#
#     with ProcessPoolExecutor() as executor:
#         start = time.time()
#         futures = [executor.submit(MatchData().find_data, url) for url in sites]
#         for sit in sites:
#
#             ans = m.find_data()
    pd.set_option('display.max_columns', None)
    m = MatchData()
    print(m.main())
#     lineups = pd.concat([lineups, line], ignore_index=True)
# engine = create_engine('sqlite:///penalty.db', echo=False)
# df = pd.read_sql_table('match_commentary1')
# for h_a in ['home_team','away_team']:
#     data['{}_score_TOP'.format(h_a)] = ((data['outcome.name'] == 'Goal') & (data['team.id'] == team)).astype(
#         int).cumsum().shift(1)

pd.set_option('display.max_columns', None)


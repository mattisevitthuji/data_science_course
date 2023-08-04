import requests
import pandas as pd
from tqdm import tqdm


def get_all_games():
    url = "https://free-nba.p.rapidapi.com/games"
    final_df = pd.DataFrame()
    headers = {
        "X-RapidAPI-Key": "c04bef501cmsh91218f4c86b1b44p1e2f47jsne1978861b6f3",
        "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
    }
    page = 0
    while True:
        querystring = {"page": str(page),"per_page":"100"}
        page += 1
        response = requests.get(url, headers=headers, params=querystring)
        json_file = response.json()
        df = pd.json_normalize(json_file['data'])
        if(df.empty):
            break
        final_df = pd.concat([final_df, df]).reset_index(drop=True)
    final_df.to_pickle('games.pkl')
    final_df.to_csv('games.csv', index=False)
    games_df = pd.read_csv('games.csv')
    print("games data: ")
    print(games_df.head(5))
    return response.json()




def get_all_stats():
    url = "https://free-nba.p.rapidapi.com/stats"
    final_df = pd.DataFrame()
    headers = {
        "X-RapidAPI-Key": "c04bef501cmsh91218f4c86b1b44p1e2f47jsne1978861b6f3",
        "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
    }
    for i in tqdm(range(1000)):
        querystring = {"page": str(i), "per_page": "100"}
        response = requests.get(url, headers=headers, params=querystring)
        json_file = response.json()
        df = pd.json_normalize(json_file['data'])
        final_df = pd.concat([final_df, df]).reset_index(drop=True)
    final_df.to_pickle('stats.pkl')
    final_df.to_csv('stats.csv', index=False)
    final_df = pd.read_csv('stats.csv')
    print("stats data: ")
    print(final_df.head(5))


def sort_data_frames():
    game_data = pd.read_csv('games.csv')
    stats_data = pd.read_csv('stats.csv')
    game_data = game_data.sort_values(by=['id', 'date'])
    # Sort stats_data by 'game_id' and 'min'
    stats_data = stats_data.sort_values(by=['game.id', 'game.date'])
    print("games data: ")
    print(game_data.head(5))
    print("stats data: ")
    print(stats_data.head(5))
    # Save the sorted dataframes as CSV files
    game_data.to_csv('game_data_all.csv', index=False)
    stats_data.to_csv('stats_data_all.csv', index=False)


def merge_data():
    game_data = pd.read_csv('game_data_all.csv')
    stats_data = pd.read_csv('stats_data_all.csv')
    game_data = game_data.rename(columns={'id': 'game.id'})
    merged_data = pd.merge(game_data, stats_data, on='game.id', how='outer')
    merged_data.to_csv('merged_data.csv', index=False)
    print("stats data: ")
    print(merged_data.head(5))


def remove_pre_2019():
    game_data = pd.read_csv('merged_data.csv')
    game_data = game_data[game_data['season'] >= 2019]
    game_data.to_csv('merged_data.csv', index=False)
    print("pre 2019 data: ")
    print(game_data.head(5))


def sort_merged_data():
    merged_data = pd.read_csv('merged_data.csv')
    merged_data = merged_data.sort_values(by=['game.id', 'date'])
    merged_data.to_csv('merged_data.csv', index=False)
    print("sorted data: ")
    print(merged_data.head(5))


def rename_columns():
    new_column_names = {
        'ast': 'assists',
        'blk': 'blocks',
        'dreb': 'defensive_rebounds',
        'fg3_pct': 'three_point_pct',
        'fg3a': 'three_point_attempts',
        'fg3m': 'three_point_made',
        'fg_pct': 'field_goal_pct',
        'fga': 'field_goal_attempts',
        'fgm': 'field_goal_made',
        'ft_pct': 'free_throw_pct',
        'fta': 'free_throw_attempts',
        'ftm': 'free_throw_made',
        'min': 'minutes_played',
        'oreb': 'offensive_rebounds',
        'pf': 'personal_fouls',
        'pta': 'points',
        'rreb': 'rebounds',
        'stl': 'steals',
        'pts': 'points'
    }
    merged_data = pd.read_csv('merged_data.csv')
    merged_data = merged_data.rename(columns=new_column_names)
    merged_data.to_csv('merged_data.csv', index=False)
    print("removed cols data: ")
    print(merged_data.head(5))


def create_new_features():
    merged_data = pd.read_csv('merged_data.csv')
    # 'team_won’ which will represent whether the player’s team won or lost the game

    def players_team_won(row):
        return row['team.id'] == row['home_team.id'] if row['home_team_score'] > row['visitor_team_score'] else \
            row['team.id'] == row['visitor_team.id']

    merged_data['team_won'] = merged_data.apply(players_team_won, axis=1)

    # ‘abs_score_difference’ absolute score difference between the home and the visitors.
    merged_data['abs_score_difference'] = merged_data.apply(
        lambda row: abs(row['home_team_score'] - row['visitor_team_score']), axis=1)

    # ‘player_full_name’ and remove the separated columns.
    merged_data['player_full_name'] = merged_data['player.first_name'] + ' ' + merged_data['player.last_name']
    merged_data = merged_data.drop(columns=['player.first_name', 'player.last_name'])

    # ‘player_height_cm’. Convert the height to cm using ft × 30.48 + in × 2.54.
    merged_data['player_height_cm'] = merged_data.apply(
        lambda row: row['player.height_feet'] * 30.48 + row['player.height_inches'] * 2.54, axis=1)

    # ‘home_player_stats’ that indicate whether the player’s stats is for the home or visiting team (hint: look at
    # whether the ‘player_team_id’ is of the ‘home’ or ‘visiting’ team).
    merged_data['home_player_stats'] = merged_data.apply(lambda row: row['home_team.id'] == row['team.id'],
                                                         axis=1)
    # Extra features
    merged_data['same_division'] = merged_data.apply(lambda row: row['home_team.division'] ==
                                                                 row['visitor_team.division'], axis=1)
    def winning_team(row):
        return row['home_team.full_name'] if row['home_team_score'] > row['visitor_team_score'] else\
            row['visitor_team.full_name']
    merged_data['winning_team'] = merged_data.apply(winning_team, axis=1)
    def calc_point_per_minute(row):
        minute_float = float(str(row['minutes_played']).replace(':','.'))
        if minute_float != 0:
            return row['points'] / minute_float
        return 0
    merged_data['points_per_minute'] = merged_data.apply(calc_point_per_minute, axis=1 )
    merged_data.to_csv('merged_data.csv', index=False)
    print("new feature data: ")
    print(merged_data.head(5))


def remove_redundant_cols():
    redundant_cols = ['player','team.name', 'team.full_name','team.division', 'team.conference',
                      'team.city', 'team.abbreviation', 'team.id', 'player.team_id', 'game.date', 'team.city',
                      'game.home_team_score', 'game.period', 'game.postseason', 'game.season',
                      'game.status', 'game.time','game.visitor_team_id', 'game.visitor_team_score']
    merged_data = pd.read_csv('merged_data.csv')
    merged_data = merged_data.drop(columns=redundant_cols)
    print(merged_data.columns)
    merged_data.to_pickle('nba_data_preprocessed.pkl')
    merged_data.to_csv('nba_data_preprocessed.csv', index=False)
    print("non redundant cols data: ")
    print(merged_data.head(5))



def calculate_2019_winner():
    df = pd.read_csv('nba_data_preprocessed.csv')
    df = df[df['season'] == 2019]
    wins_by_team = df.groupby('winning_team')['game.id'].nunique()
    team_with_most_wins = wins_by_team.idxmax()
    print("The team with the most wins in 2019 was:", team_with_most_wins)
    winning_points_sum = df.drop_duplicates(subset="game.id"
                                    ).loc[df["home_team.full_name"] == team_with_most_wins, "home_team_score"].sum() + \
                 df.drop_duplicates(subset="game.id").loc[df["visitor_team.full_name"] == team_with_most_wins, "visitor_team_score"].sum()
    print(f"The sum of points scored by {team_with_most_wins} is: {winning_points_sum}")
    team_with_least_wins = wins_by_team.idxmin()
    print("The team with the least wins in 2019 was:", team_with_least_wins)
    losing_points_sum = df.drop_duplicates(subset="game.id"
                                    ).loc[df["home_team.full_name"] == team_with_least_wins, "home_team_score"].sum() + \
                 df.drop_duplicates(subset="game.id").loc[
                     df["visitor_team.full_name"] == team_with_least_wins, "visitor_team_score"].sum()
    print(f"The sum of points scored by {team_with_least_wins} is: {losing_points_sum}")


if __name__ == '__main__':
    # Q1-2
    get_all_stats()
    get_all_games()

    # Q3
    sort_data_frames()
    #
    # # Q4
    merge_data()
    #
    # #Q5
    remove_pre_2019()
    #
    # #Q6
    sort_merged_data()
    #
    # #Q7
    rename_columns()
    #
    # #Q8
    create_new_features()
    #
    # #Q9-10-11
    remove_redundant_cols()

    #Final questions
    # Q3
    calculate_2019_winner()
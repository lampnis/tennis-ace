import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())
print(df.columns.to_list())



# perform exploratory analysis here:

# plt.scatter(df['FirstServe'], df['Wins'])
# plt.text(0, 0, '1')
# plt.show()

# plt.scatter(df['DoubleFaults'], df['Losses'])
# plt.text(0, 0, '2', fontsize=36)
# plt.show()

# plt.scatter(df['DoubleFaults'], df['Wins'])
# plt.text(0, 0, '3', fontsize=36)
# plt.show()

# plt.scatter(df['BreakPointsFaced'], df['Wins'])
# plt.text(0, 0, '5', fontsize=36)
# plt.show()

# plt.scatter(df['BreakPointsOpportunities'], df['Wins'])
# plt.text(0, 0, '6', fontsize=36)
# plt.show()

def previous_ranking(frame):
    previous_rank = []
    for _, values in frame.iterrows():
        # print(values)
        name = values['Player']
        try:
            previous_year = values['Year'] - 1
            previous_rankler = df[(df['Player'] == name) & (df['Year'] == previous_year)]['Ranking']
            previous_rank.append(int(previous_rankler.astype(int).values))
        except ValueError:
            print('No rank, returning 1501')
            previous_rank.append(int(values['Ranking']))
        except TypeError:
            print('No rank, returning 1501')
            previous_rank.append(int(values['Ranking']))
        
    return previous_rank

df['RankingPrev'] = previous_ranking(df)
print(df.columns.to_list())
print(df.sort_values(by=['Player', 'Year']).tail(20))


## perform single feature linear regressions here:

ols = LinearRegression()

columns = ['Year', 'FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon', 'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon', 'TotalServicePointsWon']

win_dict = {}
for col in columns:
    X_train, X_test, y_train, y_test = train_test_split(
        df[[col]], df[['Wins']],
        train_size=0.8
    )
    doublefaults_wins = ols.fit(X_train, y_train)
    win_dict[col] = [doublefaults_wins.score(X_test, y_test)]

win_dict_sorted = dict(sorted(win_dict.items(), key=lambda item: item[1], reverse=True))
print(list(win_dict_sorted.items())[0:3])

winnings_dict = {}
for col in columns:
    X_train, X_test, y_train, y_test = train_test_split(
        df[[col]], df[['Winnings']],
        train_size=0.8
    )
    doublefaults_wins = ols.fit(X_train, y_train)
    winnings_dict[col] = [doublefaults_wins.score(X_test, y_test)]

winnings_dict_sorted = dict(sorted(winnings_dict.items(), key=lambda item: item[1], reverse=True))
print(list(winnings_dict_sorted.items())[0:3])




## perform multiple feature linear regressions here:
for n_features in range(len(win_dict_sorted.items())):

    try:

        top_three_wins = [item[0] for item in list(win_dict_sorted.items())[0:n_features]]

        X_train, X_test, y_train, y_test = train_test_split(
            df[top_three_wins], df[['Wins']],
            train_size=0.8
        )
        wins_MLR_model = ols.fit(X_train, y_train)
        print(f'MLR model score for wins w/ top {n_features} parameters from individual regressions:\nR^2 = {wins_MLR_model.score(X_test, y_test):.2f}')
    
    except ValueError as e:
        print(f'\n\n{e}\n\nNeed at least one feature!')
        continue

for n_features in range(len(winnings_dict_sorted.items())):
    try:

        top_three_winnings = [item[0] for item in list(winnings_dict_sorted.items())[0:3]]

        X_train, X_test, y_train, y_test = train_test_split(
            df[top_three_winnings], df[['Winnings']],
            train_size=0.8
        )
        winnings_MLR_model = ols.fit(X_train, y_train)
        print(f'MLR model score for winningss w/ top {n_features} parameters from individual regressions:\nR^2 = {winnings_MLR_model.score(X_test, y_test):.2f}')

    except ValueError as e:
        print(f'\n\n{e}\n\nNeed at least one feature!')
        continue
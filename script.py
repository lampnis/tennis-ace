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
# df['RankingPrev'] = df.apply()
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
    win_dict_sorted = dict(sorted(win_dict.items(), key=lambda item: item[1]))
    print(win_dict_sorted)

winnings_dict = {}
for col in columns:
    X_train, X_test, y_train, y_test = train_test_split(
        df[[col]], df[['Winnings']],
        train_size=0.8
    )
    doublefaults_wins = ols.fit(X_train, y_train)
    winnings_dict[col] = [doublefaults_wins.score(X_test, y_test)]
    winnings_dict_sorted = dict(sorted(winnings_dict.items(), key=lambda item: item[1]))
    print(winnings_dict_sorted)







## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:

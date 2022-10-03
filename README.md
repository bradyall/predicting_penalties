# predicting_penalties
Project to scrape soccer data from all matches on espn.com and use data to predict penalty locations

This project collects data from around 300,000 soccer matches on ESPN.com and uses this data to try to predict penalty locations. The primary data collected is lineups, commentary, goals, and scores. Natural Language Processing is used to transform each line of commentary about a penalty kick from the matches into data about who took the penalty kick, which team they played for, what foot they used, the direction that the ball was placed, and the outcome of the kick. Data was then collected on the situation surrounding the penalty kick, such as the score, whether the taker was on the home team or away time, the time in the match of the kick, and if they have taken a penalty before, to attempt to predict the most likely outcome for each penalty. 

The files are broken down as follows:

1. The game_scrape collects all of the data (with the help of the matches_w_comments file), creates a local SQL database, and stores the data in the     database. 
2. The text_to_data uses Natural Language Processing to transform the commentary into usable data
3. The machine_learning pulls the data from SQL, performs EDA to find trends in the data, and uses machine learning to predict the locations of the penalties

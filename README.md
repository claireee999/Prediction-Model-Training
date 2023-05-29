# Prediction-Model-Training
## Overview
The Ludum Dare competition is the competition where participants submit games they design in a certain amount of time for evaluation. This program trains a model to predict the overall ranking each game would get in the Ludum Dare competetion using quanlitative and qualitive data of each game. 
## Dataset Description
Ludum Dare is a well-known, long-running online game jam. It is essentially a competition in which contestants develop a game in 48 or 72 hours, ideally relating to a particular theme, and receive rankings (1-5 stars) from other contestants. There are two categories: the "compo" (developers work alone, using no premade assets, and have only 48 hours) and the "jam" (contestants may work in teams, using some premade assets, and may take up to 72 hours). Games receive an overall rating as well as ratings for individual categories (fun, theme, innovation, graphics, audio, humour, and mood). <br/>
This data set consists of data about all games entered in the last 9 iterations of the Ludum Dare competition, obtained via the public API of the official Ludum Dare site. Each entry is labeled according to its final average score in the "overall" category (from 1 to 5 stars, rounded to the nearest integer, or 0 if the game did not receive enough ratings to officially rank). This is a classification problem with 6 (unbalanced) classes. 
For each game, there are several numerical and categorical features available. <br/>
The training data consists of games entered in LD38 through 45, while the test data comes from LD46. LD46 was the largest Ludum Dare competition yet by a substantial margin, presumably due to a large influx of new participants on account of COVID-19 quarantine; as such, keep in mind that there may be some shift in the distribution between the training and test set.
## Run Program
`python3 main.py` <br/>
Install xgboost using `pip install xgboost` if needed.
## Result 
The program has a prediction accuracy of 96.4% on the test set.

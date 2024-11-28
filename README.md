# XYZ Ball Odds Prediction from Point Spread

## Table of Contents

1.  [Approach](#approach)
2.  [Cleaning](#cleaning)
3.  [Results](#results)
4.  [Version 2 Results](#version-2-results)
5.  [Final Submission](#final-submission)

## Approach

**Problem:** Given a point spread and league number determine the bookmaker prices for the home and away team in American or Decimal odds. Essentially, this function maps the expected score to money line odds.

My understanding of the dataset provided is that there are 209 games from 3 unique leagues where we are given the point spread from a US bookmaker and a point spread from a EU bookmaker for the same game. We are also given the moneyline for the home team and the away team in both American and Decimal Odds (most likely from the same bookmaker respectively). The data has some errors that we need to find and clean. From the data we also know a positive point spread means the home team is the favourite and a negative point spread means the away team is a favourite. Our goal is to find the relationship between point spread and the moneylines for the home and away team.

Initially, I tried using linear regression to model the data because of the intuitive linear relationship between point spread and moneyline odds. A greater point spread typically indicates that the bookmaker believes a team is more likely to win by a larger margin, and thus their moneyline odds should be higher. Then in terms of who the favourite/underdog is, the more positive the point spread is, the more negative the home team's moneyline will be and more positive the away team's money line will be (if we're using American Odds), and vice versa. However, the relationship is obviously not one to one. In other words, if the home team is -500, that doesn't automatically mean the away team is +500, however it should be somewhat linear. I think linear regression failed to catch the subtleties and nonlinearities in the data, which is shown by the larger error. Thus, I went to a Light Gradient Boosted Machine (LGBM) model to model this function.

I think LGBM was a good choice for this problem because it's designed to handle structured tabular data efficiently, which is exactly the format of this dataset. Its gradient-boosting approach iteratively builds an ensemble of decision trees, allowing it to focus on correcting errors from previous iterations and fine-tuning predictions. LGBM can handle missing or noisy data effectively, which is particularly important given the initial errors in the dataset that needed cleaning. Also it has the ability to incorporate regularization (e.g., L1/L2 penalties) which helps prevent overfitting, ensuring that the model remains generalizable even with the limited number of data points available.

At some point I thought of splitting the dataset into 12 models (home and away, underdog and favourite, for each of the 3 leagues). I decided to create just 2 models: one for predicting home team odds and another for predicting away team odds. This approach helps prevent overfitting, which is particularly important given the limited size of the dataset (209 games). Splitting the data into 12 subsets would leave each model with even fewer training points, increasing the risk of overfitting to noise and producing less generalizable results. Additionally, by training just two models, we ensure the system is more practical for predicting odds for new leagues. With this approach, the models can generalize across leagues, allowing them to immediately generate reasonable odds for leagues without historical data, instead of requiring years of data collection to produce reliable predictions.

## Cleaning

From inspecting the plots of the data and looking at the data myself. Below is the found 3 errors within the data and how I fixed it:

1.  League 2 does not have an EU point spread and therefor has no EU odds. **Solution:** We don't want to drop the row because we have limited data (209 points), and especially we would be dropping a whole league. The intuitive solution is to just use what we have (i.e. use point_spread_us and American odds) and copy them over to decimal. Another solution
2.  5 rows, specifically game_id's 30, 31, 32, 180, 181 have no American odds (or both homeprice_us and awayprice_us are listed as 999). **Solution:** Similarly to above, we use the knowledge we have: the homeprice_eu and awayprice_eu. Although the point spreads are slightly different and therefore the odds should be slightly different we know they should be very similar and generalize this for the model.
3.  The sneakiest of them all, when point_spread_us and point_spread_eu disagree on the favourite. We know a positive point spread means the home team is the favourite and the away team is the underdog (with the exception of small point spread where both teams are favourites). That is, the error is when point_spread_us < 0 and point_spread_eu > 0 (e.g. game_id 33: point_spread_us = -147.5 and point_spread_eu = 146.5). **Solution:** To determine which point spread is correct we have to look at the other fields. We look at the American Odds and the Decimal Odds and take whichever they both agree on is the favourite. If they agree the home team is the favourite then both point spreads should be positive, otherwise we should make them both negative.

These changes take place in `clean_data.py`.

## Results

After these changes the model trained well despite the lack of data. We measure the error of the model with multiple error metrics to get the whole picture of how the model is doing. We use Mean Absolute Error (MAE), Mean Squared Error(MSE), and Mean Absolute Percentage Error (MAPE). Below shows the results of only cleaning the data with techniques 1 and 2 to cleaning the data with techniques 1, 2, and 3 from `plot_graphs.py`:

Home Price US Model MAE: ~24 ⇒ 11.83 \
Home Price US Model MSE: ~1500 ⇒ 547.56 \
Home Price US Model MAPE: ~16% ⇒ 0.07% \
Away Price US Model MAE: ~25 ⇒ 10.67 \
Away Price US Model MSE: ~1500 ⇒ 283.29 \
Away Price US Model MAPE: ~17% ⇒ 0.07% \

This makes sense because mean squared error punishes large mistakes greater than mean absolute error. So before cleaning the data properly there were many outliers (from technique 3) that didn't follow the intuitive nature of positive point spread, home team favourite, away team underdog and negative point spread, away team favourite, home team underdog.

![image](/images/US_PSvsHomeOddsL1V1.png)
![image](/images/US_PSvsAwayOddsL1V1.png)
![image](/images/US_PSvsHomeOddsL2V1.png)
![image](/images/US_PSvsAwayOddsL1V1.png)
![image](/images/US_PSvsHomeOddsL3V1.png)
![image](/images/US_PSvsAwayOddsL1V1.png)

We can clearly see for point spreads around -200 to 200 the model is very accurate. It is even able to determine that both teams are favourites when the point spread is very small:

Predicted Odds: Point Spread = 0.5, League = 3 \
Home Price (US): -111.0; Away Price (US): -110.45

We can also tell why the mean squared error is so large: The extreme points, where we define an extreme point as a point when the point spread is greater than 200 or less than -200, the model flattens out instead of continuing on the curve. By inspecting the graphs we can let us also call extreme points where the money line is greater than +200 and -200. These large errors are the main contributors to the large mean squared error.

### Possible errors:

1. One reason I can think of is that these extreme points are not in the training dataset and if the model never sees that the odds can go that low or high, then it flattens outs. **Solution:** we force these data points in the training set.
2. Another thing that I thought was happening was that even if the extreme points were in training set, there's so little that they have no significance on the model. **Solution:** artifically increasing their representation, by duplicating the extreme points. This ensures when we do random training splits, enough extreme points are in the training set so the model treats them more significantly.

## Version 2 Results

Home Price US Model MAE: 8.54 \
Home Price US Model MSE: 348.06 \
Home Price US Model MAPE: 0.05% \
Away Price US Model MAE: 8.95 \
Away Price US Model MSE: 351.04 \
Away Price US Model MAPE: 0.06%

![image](/images/US_PSvsHomeOddsL1V2.png)
![image](/images/US_PSvsAwayOddsL1V2.png)
![image](/images/US_PSvsHomeOddsL2V2.png)
![image](/images/US_PSvsAwayOddsL1V2.png)
![image](/images/US_PSvsHomeOddsL3V2.png)
![image](/images/US_PSvsAwayOddsL1V2.png)

### Preventing Overfitting

At first when I saw these graphs, I certainly thought the model was overfitting to the training data. To verify these results I did 3 things:

1. Adjust the parameters of the model. More specifically, number of leaves and learning rate. Reducing the number of leaves from 144 to 20 limits the models complexity and controls overfitting. Using a learning rate less than 0.1 (this model uses 0.05) was another technique used to reduce overfitting
2. Early stopping. LightGBM models have a parameter `callbacks = [lgb.early_stopping(stopping_rounds=50)]` that stop training early when it detects overfitting
3. Cross-validation. Using sklearn's `cross_val_score` and measuring with MAE the results were ~2x: \
   Cross-Validation MAE (Home Price): 21.4 \
   Cross-Validation MAE (Away Price): 19.3 \
   which is good considering the size of the dataset and the range of the data (approximately -500 to 500).

## Final Submission

Satisfied with these results I exported the home and away model using LightGBM's native `save_model` function after training them and loading them in the `submission.py` where our `makeSpreads` function is. `makeSpreads(point_spread: float, league_id: int)` which takes `point_spread` a float representing the point spread the bookmaker has for this game and the `league_id`. `makeSpreads(point_spread: float, league_id: int)` returns a tuple containing 2 elements, the first being the moneyline for the home team in American Odds and the second being the moneyline for the away team in American Odds.

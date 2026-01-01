# Result of this Model Training

After training different models, like Linear Regression, Trees etc

## 1:

When used simple Linear Regression
Got RMSE to be **0.15700871843194814**

## 2:

Then used Random Forest Regressor
Got RMSE to be **0.1431334742252469**

I just improved.
0.157 → 0.143
That is a huge jump in tabular ML.
And this happened because tree models understand non-linear relationships.

## 3:

Then used Gradient Boosting Regressor
And brought down the RMSE to **0.12414960224517031**

This is elite-level tabular performance for a clean pipeline.

## 4:

Then used something, which was new for me

- Model Stacking (combining my best models into one super model)

Using it i got the lowest RMSE of **0.12261541194584222**

## 5:

After that use ExtraTrees Regressor and Light BGM Regressor, but the RMSE was higher

### Conclusion:

Gradient Boosting Regressor was the best and taking it up a notch was when used Model Stacking.

So the champion model was GradientBoostingRegressor (RMSE = 0.124)

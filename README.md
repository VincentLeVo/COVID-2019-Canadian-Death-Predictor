# COVID-2019-Canadian-Death-Predictor
Predicts death count of Canada for Covid for the next five days.


## What it does:
Participating in a Kaggle competition, our main goal was to ‘predict’ “forward in time” give days of COVID19 daily death counts for the entire country of Canada. An initial CoVid19 dataset for numerous countries from ECDC with the following features was provided: {country id, date, cases, deaths, cases 14 100k, cases 100k} from late December 2019 to early October 2020. 
Written in Python, the dataset was simplified to only preserve countries ‘similar’ to Canada and with machine learning, the least squares classifier and auto-regressive model was implemented to extrapolate the predicted Canadian death counts for specific future dates in October 2020. The output is the training error and testing error. 

## To Run:
1. Install Python 3.++
2. Cd in the /code directory 
3. 
```
python main -q 1
```

## Technologies Used:
* Python
* Pandas
* SkLearn

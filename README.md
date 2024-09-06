# Stock Price Prediction
https://docs.google.com/document/d/11CoE7TrdprRMbzgC2VPh9_E2TvYQ-ulywtujbtXgqqI/edit?pli=1

## Build a new conda environment from the requirements.txt

1. Install Anaconda3 from `https://www.anaconda.com`.
2. Make sure you have `pip` installed on your machine and upgraded to the latest version.
3. Navigate to stock_price_prediction/environments/
```shell
conda create env
```
4. Activate the environment.
```shell
conda activate stock_prediction
```
5. Navigate to stock_price_prediction/backend/ and upgrade local dev package.
```shell
pip install -e .
```
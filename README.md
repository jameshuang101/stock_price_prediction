# Stock Price Prediction

https://docs.google.com/document/d/11CoE7TrdprRMbzgC2VPh9_E2TvYQ-ulywtujbtXgqqI/edit?pli=1

## Build a new conda environment from the requirements.txt

1. Install Anaconda3 from `https://www.anaconda.com`.

2. Make sure you have `pip` installed on your machine and upgraded to the latest version.

3. Navigate to stock_price_prediction/environments/

4. Change the prefix in `environment.yml` to where you would like to save the environment. Recommended path is `~/anaconda3/envs/`.

5. Install the environment.

```shell
conda env create -f environment.yml
```

6. Activate the environment.

```shell
conda activate stock_prediction
```

7. Navigate to stock_price_prediction/backend/ and upgrade local dev package.

```shell
pip install -e .
```

## Authorize your device with the FRED API

1. Visit `https://fred.stlouisfed.org/docs/api/api_key.html` to retrieve or create a new FRED API key.

2. Create an empty file titled `api_keys.yml` in the root directory of the project `stock_price_prediction/`.

3. Add your FRED API key to the file as follows:

```yaml
fred_api_key: "YOUR_API_KEY_HERE"
```

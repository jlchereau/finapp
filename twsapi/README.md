# Interactive Brokers

Docs:
- https://www.interactivebrokers.com/campus/ibkr-api-page/

## IB Gateway

Download and install IB Gateway from https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

## TWS API

1) Download the stable version from from https://interactivebrokers.github.io/#
2) Install TWS API as explained in https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#unix-install

```shell
cd twsapi && unzip twsapi_macunix.<Major Version>.<Minor Version>.zip
cd ..
python -m venv .venv
source .venv/bin/activate
pip3 install -U pip
python3 -m pip install twsapi/IBJts/source/pythonclient
```

3) Confirm installation

```shell
python -m pip show ibapi
```

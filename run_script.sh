#!/bin/bash

while true
do
    nohup python3 volatility_strategy_binance_auto.py > output.log 2>&1
    if [ $? -eq 0 ]; then
        echo "Script executed successfully, exiting..."
        break
    else
        echo "Script failed, restarting in 15 minutes..."
        sleep 900
    fi
done

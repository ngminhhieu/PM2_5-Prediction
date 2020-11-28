#!/bin/bash

if [ ! -d "PM2_5-Prediction" ]
then
git clone -b ga https://github.com/ngminhhieu/PM2_5-Prediction
while [ ! -d "PM2_5-Prediction" ]
do
  sleep 2 # or less like 0.2
done
fi
rm -rf "PM2_5-Prediction/log/*"
cd "PM2_5-Prediction"
tmux new-session -d -s real

FIXED_SHUFFLE_WINDOW=( "fixed_false-backtest_0" "fixed_false-backtest_10" "fixed_false-backtest_20" "fixed_false-backtest_40" "fixed_false-backtest_60" "fixed_false-backtest_80" "fixed_false-backtest_100")
index=(0 1 2 3 4 5 6)
percentage_back_test=(0 10 20 40 60 80 100)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${FIXED_SHUFFLE_WINDOW[$i]} " ENTER
    sleep 0.5
done
sleep 1
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${FIXED_SHUFFLE_WINDOW[$i]} 'python main.py --percentage_back_test=${percentage_back_test[$i]} --fixed=false --tmp=$i' ENTER" ENTER
    sleep 1
done

FIXED_WINDOW=( "fixed_false-shuffle_false-backtest_0" "fixed_false-shuffle_false-backtest_10" "fixed_false-shuffle_false-backtest_20" "fixed_false-shuffle_false-backtest_40" "fixed_false-shuffle_false-backtest_60" "fixed_false-shuffle_false-backtest_80" "fixed_false-shuffle_false-backtest_100")
percentage_back_test=(0 10 20 40 60 80 100)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${FIXED_WINDOW[$i]} " ENTER
    sleep 0.5
done
sleep 1
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${FIXED_WINDOW[$i]} 'python main.py --percentage_back_test=${percentage_back_test[$i]} --fixed=false --shuffle=false --tmp=$i' ENTER" ENTER
    sleep 1
done

# End of outer loop.
exit 0

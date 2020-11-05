#!/bin/bash

if [ ! -d "PM2_5-Prediction" ]
then
git clone https://github.com/ngminhhieu/PM2_5-Prediction
while [ ! -d "PM2_5-Prediction" ]
do
  sleep 2 # or less like 0.2
done
fi
rm -rf "PM2_5-Prediction/log/*"
cd "PM2_5-Prediction"
git checkout ga
tmux new-session -d -s real

FIXED_SHUFFLE_WINDOW=( "shuffle_true-backtest_0" "shuffle_true-backtest_10" "shuffle_true-backtest_20" "shuffle_true-backtest_40" "shuffle_true-backtest_60" "shuffle_true-backtest_80" "shuffle_true-backtest_100")
index=(0 1 2 3 4 5 6)
percentage_back_test=(0 10 20 40 60 80 100)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${FIXED_SHUFFLE_WINDOW[$i]} " ENTER
done

for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${FIXED_SHUFFLE_WINDOW[$i]} 'python3 main.py --population=2 --percentage_back_test=${percentage_back_test[$i]}' ENTER" ENTER
done

FIXED_WINDOW=( "shuffle_false-backtest_0" "shuffle_false-backtest_10" "shuffle_false-backtest_20" "shuffle_false-backtest_40" "shuffle_false-backtest_60" "shuffle_false-backtest_80" "shuffle_false-backtest_100")
index=(0 1 2 3 4 5 6)
percentage_back_test=(0 10 20 40 60 80 100)
for i in "${index[@]}"
do
    tmux send-keys -t real "tmux new-window -n ${FIXED_WINDOW[$i]} " ENTER
done

for i in "${index[@]}"
do
    tmux send-keys -t real "tmux send-keys -t ${FIXED_WINDOW[$i]} 'python3 main.py --population=2 --percentage_back_test=${percentage_back_test[$i]} --shuffle_gen=false' ENTER" ENTER
done

# End of outer loop.
exit 0

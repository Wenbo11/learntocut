set -e

for seed in 1 2 3 
do
for step in 0.01
do
for std in 0.02
do
for type in attention
do
for i in {1..1000}
do

echo 'launching' ${i}

python main_cuts.py -nd 10 --seed $seed --step_size $step --delta_std $std --policy_type $type

done
done
done
done
done

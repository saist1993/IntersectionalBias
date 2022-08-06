echo "running methods over adult multi group dataset"

cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_titled_erm  -fairness_function equal_opportunity &> log1 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_titled_erm  -fairness_function equal_opportunity &> log2 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_titled_erm  -fairness_function equal_opportunity &> log3 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_titled_erm  -fairness_function equal_opportunity &> log4 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_titled_erm  -fairness_function equal_opportunity &> log5 &

cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_mixup  -fairness_function equal_opportunity &> log6 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_mixup  -fairness_function equal_opportunity &> log7 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_mixup  -fairness_function equal_opportunity &> log8 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_mixup  -fairness_function equal_opportunity &> log9 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method only_mixup  -fairness_function equal_opportunity &> log10 &

cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_mixup  -fairness_function equal_opportunity &> log6 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_mixup  -fairness_function equal_opportunity &> log7 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_mixup  -fairness_function equal_opportunity &> log8 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_mixup  -fairness_function equal_opportunity &> log9 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_mixup  -fairness_function equal_opportunity &> log10 &

wait




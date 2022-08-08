echo "running methods over adult multi group, adult, and twitter hate speech dataset"


cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log6a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log7a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log8a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log9a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log10a &



cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log6b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log7b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log8b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log9b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log10b &



cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log6c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log7c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log8c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log9c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 300 -method tilted_erm_with_fairness_loss  -fairness_function equal_opportunity &> log10c &

wait

echo "running methods over twitter hate speech dataset"

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log1 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log2 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log3 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log4 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log5 &

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log6 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log7 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log8 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log9 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log10 &

wait

echo "running methods over twitter hate speech dataset v1"

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log1a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log2a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log3a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log4a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log5a &

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log6a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log7a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log8a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log9a &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech_v1 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log10a &



wait

echo "running methods over twitter hate speech dataset v2"

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log1b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log2b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log3b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log4b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log5b &

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log6b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log7b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log8b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log9b &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech_v2 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log10b &



wait

echo "running methods over twitter hate speech dataset v3"

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log1c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log2c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log3c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log4c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log5c &

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log6c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log7c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log8c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log9c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name twitter_hate_speech_v3 -batch_size 1000 -model simple_non_linear -epochs 75 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_opportunity &> log10c &

echo "running first block"


cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method tilted_erm_with_mixup_only_one_group  -fairness_function equal_odds &> log6 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method tilted_erm_with_mixup_only_one_group  -fairness_function equal_odds &> log7 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method tilted_erm_with_mixup_only_one_group  -fairness_function equal_odds &> log8 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method tilted_erm_with_mixup_only_one_group  -fairness_function equal_odds &> log9 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method tilted_erm_with_mixup_only_one_group  -fairness_function equal_odds &> log10 &


cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -fairness_function equal_odds &> log1 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -fairness_function equal_odds &> log2 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -fairness_function equal_odds &> log3 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -fairness_function equal_odds &> log4 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained_with_fairness_loss -fairness_lambda_s 0.05 -fairness_lambda_e 0.5 -fairness_function equal_odds &> log5 &

cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -fairness_function equal_odds &> log6 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -fairness_function equal_odds &> log7 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -fairness_function equal_odds &> log8 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -fairness_function equal_odds &> log9 &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method unconstrained -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -fairness_function equal_odds &> log10 &


wait
echo "running second block"


cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_titled_erm  -fairness_function equal_odds &> log1c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_titled_erm  -fairness_function equal_odds &> log2c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_titled_erm  -fairness_function equal_odds &> log3c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_titled_erm  -fairness_function equal_odds &> log4c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_titled_erm  -fairness_function equal_odds &> log5c &

cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_mixup  -fairness_function equal_odds &> log6c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_mixup  -fairness_function equal_odds &> log7c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_mixup  -fairness_function equal_odds &> log8c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_mixup  -fairness_function equal_odds &> log9c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method only_mixup  -fairness_function equal_odds &> log10c &


cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_odds &> log6c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_odds &> log7c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_odds &> log8c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_odds &> log9c &
cd ~/codes/IntersectionalBias/src/; python hyperparam_runner.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 100 -method adversarial_group -fairness_lambda_s 0.0 -fairness_lambda_e 0.0 -adv_s 0.0 -adv_e 2.0 -fairness_function equal_odds &> log10c &


wait

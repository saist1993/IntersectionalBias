echo "first block"

cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_opportunity -version 1.0 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_odds -version 1.0 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_opportunity -version 1.0 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_odds -version 1.0 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_opportunity -version 1.0 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_odds -version 1.0 &> log6 &


cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_opportunity -version 2.0 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_odds -version 2.0 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_opportunity -version 2.0 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_odds -version 2.0 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_opportunity -version 2.0 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 50 -method $1  -fairness_function equal_odds -version 2.0 &> log6 &


wait
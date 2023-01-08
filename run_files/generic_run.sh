echo "first block"

cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log2 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log3 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log4 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log5 &


cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log7 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log8 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log9 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name adult_multi_group -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log10 &




cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log2 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log3 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log4 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log5 &


wait

echo "second block"

cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log7 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log8 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log9 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name twitter_hate_speech -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log10 &



cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log1 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log2 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log3 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log4 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_opportunity -use_dropout $2 &> log5 &


cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 10 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log6 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 20 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log7 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 30 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log8 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 40 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log9 &
cd /srv/storage/magnet@storage1.lille.grid5000.fr/gmaheshwari/IntersectionalBias/src/; python hyperparam_titled_erm.py -seeds 50 -dataset_name celeb_multigroup_v3 -batch_size 1000 -model simple_non_linear -epochs 80 -method $1  -fairness_function equal_odds -use_dropout $2 &> log10 &


wait
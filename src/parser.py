# Top level file which calls the parser.
import numpy as np
from texttable import Texttable
from parsing import basic_parser

fairness_function = 'equal_odds'
level_1_strategy_params = {'keep_last_k': 100.0}
level_2_strategy_params = {'relaxation_threshold': 0.02,
                           'fairness_function': fairness_function}
#
# print(basic_parser.get_best_run(dataset_name='adult_multi_group', method='unconstrained', model='simple_non_linear',
#                           seed=10, fairness_function=fairness_function,
#                           level_1_strategy='keep_last_k', level_1_strategy_params=level_1_strategy_params,
#                           level_2_strategy='relaxation_threshold', level_2_strategy_params=level_2_strategy_params))


def temp_table_generator():
    # methods = ['unconstrained', 'unconstrained_with_fairness_loss',
    #            'adversarial_group', 'adversarial_group_with_fairness_loss',
    #            'only_titled_erm', 'only_mixup', 'tilted_erm_with_mixup',
    #            'tilted_erm_with_fairness_loss']

    # methods = ['unconstrained', 'unconstrained_with_fairness_loss',
    #            'adversarial_group','fairgrad', 'only_titled_erm',
    #             'only_mixup', 'tilted_erm_with_mixup',
    #            'tilted_erm_with_mixup_only_one_group',
    #            'weighted_sample_erm',
    #            'only_mixup_based_on_distance',
    #            'tilted_erm_with_mixup_based_on_distance',
    #            'only_mixup_based_on_distance_and_augmentation', 'only_tilted_dro',
    #            'only_tilted_erm_with_mixup_augmentation_lambda_weights_v2'
    #            ]

    methods = ['unconstrained','only_titled_erm', 'only_mixup',
               'only_mixup_based_on_distance',
               'only_tilted_erm_with_mixup_augmentation_lambda_weights_v4',
               'tilted_erm_with_mixup_only_one_group',
               'tilted_erm_with_mixup_based_on_distance',
               'only_tilted_erm_with_weights_on_loss',
               'train_with_mixup_only_one_group_based_distance_v2',
               'train_with_mixup_only_one_group_based_distance_v3'
               ]

    # methods = [ 'unconstrained_with_fairness_loss', 'tilted_erm_with_mixup_only_one_group'
    #            ]



    # dataset_names = ['adult_multi_group']
    # dataset_names = ['twitter_hate_speech']
    dataset_names = ['celeb_multigroup_v3']
    models = ['simple_non_linear']
    seeds = [10,20,30,40,50]
    # seeds = [50]
    # fairness_function = 'equal_odds'
    fairness_function = 'equal_opportunity'
    k = 2

    level_1_strategy_params = {'keep_last_k': 100.0}
    level_2_strategy_params = {'relaxation_threshold': 0.02,
                               'fairness_function': fairness_function}

    rows = []
    average_rows = []
    rows.append(['method', 'balanced accuracy', 'fairness', 'confidence_interval', 'seed'])
    average_rows.append(['method', 'balanced accuracy', 'fairness'])

    for dataset in dataset_names:
        for model in models:
            for method in methods:
                rows_temp = []
                for seed in seeds:
                    result = basic_parser.get_best_run(dataset_name=dataset, method=method,
                                              model=model,
                                              seed=seed, fairness_function=fairness_function,
                                              level_1_strategy='keep_last_k',
                                              level_1_strategy_params=level_1_strategy_params,
                                              level_2_strategy='relaxation_threshold',
                                              level_2_strategy_params=level_2_strategy_params)
                    accuracy = result.test_epoch_metric.balanced_accuracy
                    fairness = result.test_epoch_metric.eps_fairness[fairness_function].intersectional_bootstrap[0]
                    confidence_interval = result.test_epoch_metric.eps_fairness[fairness_function].intersectional_bootstrap[1]
                    confidence_interval[0], confidence_interval[1] = round(confidence_interval[0], k),\
                                                                     round(confidence_interval[1], k)
                    rows_temp.append([method, round(accuracy,k), round(fairness,k), confidence_interval, seed])

                    if method == 'unconstrained_with_fairness_loss':
                        print(result.arguments)
                        print(result.test_epoch_metric.epoch_number)
                # average over seeds
                rows = rows + rows_temp
                average_accuracy = round(np.mean([r[1] for r in rows_temp]), k)
                average_accuracy_std = round(np.std([r[1] for r in rows_temp]), k)

                average_fairness = round(np.mean([r[2] for r in rows_temp]), k)
                average_fairness_std = round(np.std([r[2] for r in rows_temp]), k)

                average_rows.append([method, f"{average_accuracy} +/- {average_accuracy_std}", f"{average_fairness} +/- {average_fairness_std}"])


    t = Texttable()
    t.add_rows(rows)
    print(t.draw())

    t = Texttable()
    t.add_rows(average_rows)
    print(t.draw())

temp_table_generator()
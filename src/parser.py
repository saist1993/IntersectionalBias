# Top level file which calls the parser.
from pathlib import Path
from parsing import basic_parser

fairness_function = 'equal_opportunity'
level_1_strategy_params = {'keep_last_k': 100.0}
level_2_strategy_params = {'relaxation_threshold': 0.02,
                           'fairness_function': fairness_function}

print(basic_parser.get_best_run(dataset_name='adult_multi_group', method='unconstrained', model='simple_non_linear',
                          seed=10, fairness_function=fairness_function,
                          level_1_strategy='keep_last_k', level_1_strategy_params=level_1_strategy_params,
                          level_2_strategy='relaxation_threshold', level_2_strategy_params=level_2_strategy_params))
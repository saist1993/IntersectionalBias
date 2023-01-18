import os
from numpy import nan
from pathlib import Path
from metrics.calculate_epoch_metric import *

@dataclass
class BlockData:
    unique_id: str
    arguments: dict
    train_epoch_metric: EpochMetricTracker
    valid_epoch_metric: EpochMetricTracker
    test_epoch_metric: EpochMetricTracker


class BestCandidateMechanism:
    """
    The best candidate mechansim has two levels. But to understand this two levels, we need to understand data

    data = [ [blocks for run 1 - output of BasicLogParser.block_parser], [blocks for run 2], [blocks for run 3], [] ]

    Level 1 - Now [blocks for run 1] would have 50 blocks if there are 50 epochs ..
    So level 1 transforms this to say X epochs.
    This is usefule if we just want last 10% of the runs, or if we want last epoch .. we can also set it to none, if we
    don't care about it.

    Level 2 - [[pruned blocks for run 1], [pruned blocks for run 2], [pruned blocks for run 3], []] .. The objective
    here is to get final single result.
    """

    def __init__(self, data, level_1_strategy, level_1_strategy_params, level_2_strategy, level_2_strategy_params):
        self.data = data
        self.level_1_strategy = level_1_strategy
        self.level_1_strategy_params = level_1_strategy_params
        self.level_2_strategy = level_2_strategy
        self.level_2_strategy_params = level_2_strategy_params



    def keep_last_k(self):
        pruned_data = []
        for d in self.data:
            pruned_data.append(d[::-1][:int(len(d)*self.level_1_strategy_params['keep_last_k']*0.01)])
        self.data = pruned_data

    def relaxation_threshold(self):
        try:
           best_validation_accuracy = max([block.valid_epoch_metric.balanced_accuracy for blocks in self.data for block in blocks])
           print(best_validation_accuracy)
        except ValueError:
            print("here")
        all_blocks_flat = [block for blocks in self.data for block in blocks
                           if block.valid_epoch_metric.balanced_accuracy > best_validation_accuracy -
                           self.level_2_strategy_params['relaxation_threshold']]

        best_fairness_index = np.argmin([block.valid_epoch_metric.eps_fairness[self.level_2_strategy_params['fairness_function']].intersectional_bootstrap[0]
                                              for block in all_blocks_flat])

        return all_blocks_flat[best_fairness_index]

    def run(self):
        if self.level_1_strategy == 'keep_last_k':
            self.keep_last_k()
        if self.level_2_strategy == 'relaxation_threshold':
            block = self.relaxation_threshold()

        return block



class BasicLogParser:

    @staticmethod
    def block_parser(block):


        train_epoch_metric, valid_epoch_metric,  test_epoch_metric = [None, None, None]

        for b in block:
            if "train epoch metric:" in b:
                train_epoch_metric = eval(b.split("train epoch metric: ")[1]) # THIS IS NOT SAFE!
            if "test epoch metric:" in b:
                test_epoch_metric = eval(b.split("test epoch metric: ")[1])
            if "valid epoch metric:" in b:
                valid_epoch_metric = eval(b.split("valid epoch metric: ")[1])

        return [train_epoch_metric, valid_epoch_metric, test_epoch_metric]

    def core_parser(self, file_name:Path):
        print(f"file name received {file_name}")
        if 'kNpapCsDje9NtBjYUVK3yL.log' in str(file_name):
            print("here")
        with open(file_name, 'r') as f:
            lines = [line for line in f]
            unique_id = lines[0].split("unique id is:")[1]
            arguments = lines[1].split("arguemnts: ")[1]
        # if 'mixup_rg=10.0' not in arguments  and 'mixup_rg=20.0' not in arguments and 'mixup_rg=30.0' not in arguments and 'mixup_rg=40.0' not in arguments:
        #     return None
        blocks = []
        for l in lines[2:]:
            if "start of epoch block" in l:
                temp = []
            elif "end of epoch block" in l:
                blocks.append(temp)
            else:
                temp.append(l)

        parsed_blocks = []
        for b in blocks:
            block = self.block_parser(b)
            if None not in block:
                block = BlockData(
                    unique_id=unique_id,
                    arguments=arguments,
                    train_epoch_metric=block[0],
                    valid_epoch_metric=block[1],
                    test_epoch_metric=block[2]
                )
                parsed_blocks.append(block)

        return parsed_blocks



    def get_parsed_content(self, dataset_name, method, model, seed, fairness_function):
        # Step 1: Find the correct directory
        log_files_location = Path(f"../server_logs/logs/{dataset_name}/{method}/{model}/{seed}/{fairness_function}")
        all_log_files_names = log_files_location.glob('*')
        # all_log_files_content = [self.core_parser(file_name) for file_name in all_log_files_names]

        all_log_files_content = []
        for file_name in all_log_files_names:
            temp = self.core_parser(file_name)
            if temp:
                all_log_files_content.append(temp)

        return all_log_files_content


def get_best_run(dataset_name, method, model, seed, fairness_function,
                 level_1_strategy, level_1_strategy_params, level_2_strategy, level_2_strategy_params):
    basic_log_parser = BasicLogParser()
    all_log_files_content = basic_log_parser.get_parsed_content(dataset_name, method, model, seed, fairness_function)
    best_candidate_mechanism = BestCandidateMechanism(all_log_files_content,
                                                      level_1_strategy, level_1_strategy_params,
                                                      level_2_strategy, level_2_strategy_params )
    return best_candidate_mechanism.run()

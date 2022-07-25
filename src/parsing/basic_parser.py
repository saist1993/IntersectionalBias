from pathlib import Path
from metrics.calculate_epoch_metric import *


# arguments,
# unique id
# all epoch metrics
#
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
        with open(file_name, 'r') as f:
            lines = [line for line in f]
            unique_id = lines[0].split("unique id is:")[1]
            arguments = lines[1].split("arguemnts: ")[1]
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
                parsed_blocks.append(block)

        return [unique_id, arguments, parsed_blocks]
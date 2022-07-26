from main import RunnerArguments, runner
from typing import Optional, NamedTuple
import numpy as np
import argparse
import torch




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', '-seeds', nargs="*", help="seed", type=int, default=42)
    parser.add_argument('--dataset_name', '-dataset_name', help="dataset name", type=str, default='adult_multi_group')
    parser.add_argument('--batch_size', '-batch_size', help="batch size", type=int, default=1000)
    parser.add_argument('--model', '-model', help="simple_non_linear/simple_linear", type=str, default='simple_non_linear')
    parser.add_argument('--epochs', '-epochs', help="epochs", type=int, default=50)
    parser.add_argument('--save_model_as', '-save_model_as', help="save the model as", type=Optional[str], default=None)
    parser.add_argument('--method', '-method', help="unconstrained/adv ...", type=str, default='unconstrained')
    parser.add_argument('--optimizer_name', '-optimizer_name', help="adam/sgd", type=str, default='adam')
    parser.add_argument('--lr', '-lr', help="learning rate", type=float, default=0.001)
    parser.add_argument('--use_wandb', '-use_wandb', help="use wandb", type=bool, default=False)
    parser.add_argument('--adv_start', '-adv_s', help="start of adv scale for increment increase for ex: 0.1",
                        type=float, default=0.0)
    parser.add_argument('--adv_end', '-adv_e', help="end of adv scale 1.0", type=float, default=0.0)
    parser.add_argument('--fairness_lambda_start', '-fairness_lambda_s', help="start of fairnesslambda scale for increment increase for ex: 0.1",
                        type=float, default=0.0)
    parser.add_argument('--fairness_lambda_end', '-fairness_lambda_e', help="end of adv scale 1.0", type=float, default=0.0)
    parser.add_argument('--dataset_size', '-dataset_size', help="dataset_size", type=int, default=1000)
    parser.add_argument('--attribute_id', '-attribute_id', help="attribute_id", type=Optional[int], default=None)
    parser.add_argument('--log_file_name', '-log_file_name', help="log_file_name", type=Optional[str], default=None)
    parser.add_argument('--fairness_function', '-fairness_function', help="fairness_function", type=str, default='equal_opportunity')


    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    args = parser.parse_args()

    if type(args.seeds) is int:
        args.seeds = [args.seeds]

    adv_scales = [round(i, 2) for i in np.arange(args.adv_start, args.adv_end, 0.2)]
    fairness_scales = [round(i, 2) for i in np.arange(args.fairness_lambda_start, args.fairness_lambda_end, 0.05)]

    if not adv_scales:
        adv_scales = [0.0]
    if not fairness_scales:
        fairness_scales = [0.0]

    for seed in args.seeds:
        for adv_scale in adv_scales:
            for fairness_scale in fairness_scales:
                try:
                    print(f"*************************{seed}, {adv_scale}, {fairness_scale}*************************")
                    runner_arguments = RunnerArguments(
                        seed=seed,
                        dataset_name=args.dataset_name,  # twitter_hate_speech
                        batch_size=args.batch_size,
                        model=args.model,
                        epochs=args.epochs,
                        save_model_as=args.save_model_as,
                        method=args.method,  # unconstrained, adversarial_single
                        optimizer_name=args.optimizer_name,
                        lr=args.lr,
                        use_wandb=args.use_wandb,
                        adversarial_lambda=adv_scale,
                        dataset_size=args.dataset_size,
                        attribute_id=args.attribute_id,  # which attribute to care about!
                        fairness_lambda=fairness_scale,
                        log_file_name=args.log_file_name,
                        fairness_function=args.fairness_function
                    )
                    output = runner(runner_arguments=runner_arguments)
                except KeyboardInterrupt:
                    raise IOError

# generic runner: cd ~/codes/IntersectionalBias/src/; python hyperparam_runner -seeds 10 20 -dataset_name adult_multi_group -batch_size 64 -model simple_non_linear -epochs 5 -method unconstrained -fairness_lambda_start 0.05 -fairness_lambda_end 0.5 -fairness_function equal_opportunity

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
    parser.add_argument('--dataset_size', '-dataset_size', help="dataset_size", type=int, default=1000)
    parser.add_argument('--attribute_id', '-attribute_id', help="attribute_id", type=Optional[int], default=None)
    parser.add_argument('--log_file_name', '-log_file_name', help="log_file_name", type=Optional[str], default=None)
    parser.add_argument('--fairness_function', '-fairness_function', help="fairness_function", type=str, default='equal_opportunity')
    parser.add_argument('--version', '-version', help="version number", type=float, default=0.0)


    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    args = parser.parse_args()

    if type(args.seeds) is int:
        args.seeds = [args.seeds]

    if args.method == 'tilted_erm_with_mixup' or args.method == 'tilted_erm_with_mixup_only_one_group' \
            or args.method == 'tilted_erm_with_mixup_based_on_distance' \
            or args.method == 'train_with_mixup_only_one_group_based_distance_v2' \
            or args.method == 'train_with_mixup_only_one_group_based_distance_v3' \
            or args.method == 'only_mixup_based_on_distance_fid':
        if args.version == 0:
            titled_scales = [0.1, 1.0, 5.0, 10.0]
            mixup_scales = [0.3, 0.6, 0.9]
        else:
            titled_scales = [0.1, 1.0, 5.0, 10.0]
            mixup_scales = [args.version]

    if args.method == 'only_mixup' or args.method == 'only_mixup_with_loss_group' \
            or args.method == 'only_mixup_based_on_distance'\
            or args.method == 'only_mixup_based_on_distance_and_augmentation':
        titled_scales = [0.0]
        if args.version == 0:
            mixup_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif args.version == 1.0:
            mixup_scales = [0.1, 0.2, 0.3, 0.4]
        elif args.version == 2.0:
            mixup_scales = [0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            raise NotImplementedError


    if args.method == 'only_titled_erm' or args.method == 'only_tilted_erm_with_mixup_augmentation_lambda_weights'\
            or args.method == 'only_tilted_erm_with_mixup_augmentation_lambda_weights_v2'\
            or args.method == 'only_tilted_erm_with_mixup_augmentation_lambda_weights_v3'\
            or args.method == 'only_tilted_erm_with_mixup_augmentation_lambda_weights_v4'\
            or args.method == 'only_tilted_erm_with_weights_on_loss'\
            or args.method == 'only_titled_erm_with_mask'\
            or args.method in ['only_tilted_erm_generic', 'only_tilted_erm_with_mask_on_tpr',
                                    'only_tilted_erm_with_weighted_loss_via_global_weight',
                                    'only_tilted_erm_with_mask_on_tpr_and_weighted_loss_via_global_weight']:
        if args.version == 0:
            titled_scales = [0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 50.0]
            mixup_scales = [0.0]
        elif args.version == 1.0:
            titled_scales = [0.1, 1.0, 3.0, 5.0]
            mixup_scales = [0.0]
        elif args.version == 2.0:
            titled_scales = [8.0, 10.0, 50.0]
            mixup_scales = [0.0]
        else:
            raise NotImplementedError


    if args.method in ['only_tilted_dro', 'train_only_group_dro_with_augmentation_static_positive_and_negative_weights',
                       'train_only_group_dro_with_mixup_regularizer_super_group',
                       'train_only_group_dro_with_super_group']:
        mixup_scales = [0.0]
        if args.version == 0:
            titled_scales = [0.1, 0.5, 0.01, 0.05, 0.8, 0.3, 0.08, 0.03]
        elif args.version == 1.0:
            titled_scales = [0.1, 0.5, 0.01, 0.05]
        elif args.version == 2.0:
            titled_scales = [0.8, 0.3, 0.08, 0.03]
        else:
            raise NotImplementedError

    if args.method == 'tilted_erm_with_fairness_loss':
        if args.version == 0:
            titled_scales = [1.0, 5.0, 10.0]
            mixup_scales = [0.3, 0.6, 0.9]

    if args.method in ['train_only_group_dro_with_mixup_with_distance',
                                     'train_only_group_dro_with_mixup_with_random',
                                     'train_only_group_dro_with_mixup_with_random_with_weighted_sampling',
                                     'train_only_group_dro_with_mixup_with_distance_with_weighted_sampling']:
        titled_scales = [0.1, 0.5, 0.01, 0.05]
        mixup_scales = [1.0, 0.5]


    if args.method == 'train_only_group_dro' or args.method == 'train_only_group_dro_with_weighted_sampling':
        titled_scales = [0.05, 0.01, 0.1, 0.5]
        mixup_scales = [0.0]

    if args.method in ['simple_mixup_data_augmentation', 'lisa_based_mixup', 'lisa_based_mixup_with_mixup_regularizer',
                                     'lisa_based_mixup_with_distance',
                                     'lisa_based_mixup_with_mixup_regularizer_and_with_distance',
                       'take_2_train_lisa_based_mixup_with_mixup_regularizer']:
        titled_scales = [0.0]
        mixup_scales = [0.0]


    if args.method in ['train_only_group_dro_with_data_augmentation_via_mixup_super_group',
                                    'train_only_group_dro_with_data_augmentation_via_mixup_super_group_with_mixup_regularizer']:
        titled_scales = [0.01, 0.05, 0.1]
        mixup_scales = [0.25, 0.50, 0.75]


    for seed in args.seeds:
        for titled_scale in titled_scales:
            for mixup_scale in mixup_scales:
                try:
                    print(f"*************************{seed}, {titled_scale}, {mixup_scale}*************************")
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
                        adversarial_lambda=0.0,
                        dataset_size=args.dataset_size,
                        attribute_id=args.attribute_id,  # which attribute to care about!
                        fairness_lambda=0.0,
                        log_file_name=args.log_file_name,
                        fairness_function=args.fairness_function,
                        titled_t=titled_scale,
                        mixup_rg=mixup_scale
                    )
                    output = runner(runner_arguments=runner_arguments)
                except KeyboardInterrupt:
                    raise IOError

# generic runner: cd ~/codes/IntersectionalBias/src/; python hyperparam_runner -seeds 10 20 -dataset_name adult_multi_group -batch_size 64 -model simple_non_linear -epochs 5 -method unconstrained -fairness_lambda_start 0.05 -fairness_lambda_end 0.5 -fairness_function equal_opportunity

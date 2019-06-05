import argparse

def get_active_args() -> argparse.ArgumentParser:
    '''
    Create arg parse for active learning training containing options for
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = argparse.ArgumentParser(description='Build an active learning iterative pipeline')

    # Active Learning Pipeline Parameters
    parser.add_argument('--log_dir', type=str, default='logs/', help='the directory to log into')
    parser.add_argument('--model_name', type=str, default='active_learning_model', help='the name to give the model')
    parser.add_argument('--sample_strategy', type=str, default='sample', help='the method to sample the next points from')
    parser.add_argument('--debug', action='store_true', help='log debug information to console')
    parser.add_argument('--seed', type=int, default=None, help='seed the experiment')

    # dataset parameters
    parser.add_argument('--dataset', type=str, default='CADEC', help='the dataset to use {CONLL, CADEC}')
    parser.add_argument('--binary_class', type=str, default='ADR', help='the binary class to use for the dataset')
    parser.add_argument('--test', action='store_true', help='use the test set for evaluation')

    # hyper parameters
    parser.add_argument('--model_type', type=str, default='ELMo_bilstm_crf', help='the model type to use')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the hidden dimensions for the model')

    # optimizer config
    parser.add_argument('--opt_type', type=str, default='SGD', help='the optimizer to use')
    parser.add_argument('--opt_lr', type=float, default=0.01, help='the learning rate for the optimizer')
    parser.add_argument('--opt_weight_decay', type=float, default=1e-4, help='weight decay for optimizer')

    # weak data config
    parser.add_argument('--use_weak', action='store_true', help='use the weak set during training')
    parser.add_argument('--use_weak_fine_tune', action='store_true', help='use the weak fine tuning approach')
    parser.add_argument('--weak_weight', type=float, default=1.0, help='the weight to give to the weak set during training')
    parser.add_argument('--weak_function', nargs='+', default=['linear'], help='a list of the type of weak function to use')
    parser.add_argument('--weak_collator', type=str, default='union', help='the type of collator to use for the weak set')

    # training config
    parser.add_argument('--num_epochs', type=int, default=5, help='the number of epochs to run each iteration')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of training')
    parser.add_argument('--patience', type=int, default=5, help='patience parameter for training')

    # system config
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')
    parser.add_argument('--cached', action='store_true', help='rely on cached embeddings if possible')

    # Parser data loader options
    return parser
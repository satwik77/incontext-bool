import argparse
import json

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "conjunction",
    "teach_biconjunction",
    "mono_conjunction",
	"teach_conjunction",
	"nearest_neighbours",
    "disjunction",
    "sparse_disjunction",
    "sparse_thres",
    "parity",
    "sparse_parity",
    "int_halfspace",
    "majority",
    "dnf",
    "teach_dnf",
    "cnf",
]


MODELS = ['san', 'lstm', 'gpt', 'llama', 'gru', 'mysan', 'attclf', 'retnet', 'hyena', 'dss']
MODEL_LIST = ["gpt2", "gpt2-large", "gpt2-xl", 'lin_attn', 'llama7b']

def build_parser():
	parser = argparse.ArgumentParser(description='Run')

	# Miscellaneous
	parser.add_argument('-wandb', dest='wandb', action='store_true', help='Store wandb')
	parser.add_argument('-no-wandb', dest='wandb', action='store_false', help='Do not store wandb')
	parser.set_defaults(wandb=False)
	parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
	# parser.add_argument('-seed', type=int, default=1729, help='Default seed to set')

	parser.add_argument('-out_dir', type=str, default='./models/', help='outputs directory')
	parser.add_argument('-delete', dest='delete', action='store_true', help='delete model after run')
	parser.add_argument('-no-delete', dest='delete', action='store_false', help='Do not delete model after run')
	parser.set_defaults(delete=True)

	parser.add_argument('-test_run', dest='test_run', action='store_true', help='Test run')
	parser.add_argument('-no-test_run', dest='test_run', action='store_false', help='Not test run')
	parser.set_defaults(test_run=False)

	# Model
	parser.add_argument('-family', type=str, default='mysan', choices= MODELS,  help='Model Family')
	parser.add_argument('-model_name', type=str, default='gpt2', choices= MODEL_LIST,  help='Select Pretrained Model')
	parser.add_argument('-llama_weights_path', type=str, default='/home/', help='Outer directory where hf converted LLaMA weights are stored')
	parser.add_argument('-precision', type=str, default='full', choices= ['half', 'full'],  help='Select precision for llama weights')
	parser.add_argument('-n_positions', type=int, default=150, help='Maximum context length')
	parser.add_argument('-n_dims', type=int, default=28, help='input dimension')
	parser.add_argument('-n_embd', type=int, default=256, help='embedding dimension')
	parser.add_argument('-n_layer', type=int, default=6, help='number of layers')
	parser.add_argument('-n_head', type=int, default=8, help='number of heads')
	parser.add_argument('-order', type=int, default=3, help='Order: For Hyena')
	parser.add_argument('-freeze', type=int, default=0, choices= [0, 1, 2], help='0: no freeze, 1: freeze partial, 2: freeze all')

	# Task
	parser.add_argument('-task', type=str, default='conjunction', choices= TASK_LIST,  help='Task')
	# parser.add_argument('-task_kwargs', type=json.loads, default='{\"start_idx\": 20}', help='Task arguments')
	parser.add_argument('-task_kwargs', type=json.loads, default='{}', help='Task arguments')
	parser.add_argument('-num_tasks', type=int, default=0, help='number of tasks')
	parser.add_argument('-num_training_examples', type=int, default=0, help='number of training examples')
	parser.add_argument('-data', type=str, default='boolean', choices= ['gaussian', 'boolean'],  help='Data Type')
	parser.add_argument('-noise_rate', type=float, default=0.0, help='Noise rate during training')

	# Training
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('-learning_rate', type=float, default=0.0001, help='Learning rate')
	parser.add_argument('-train_steps', type=int, default=20001, help='number of train steps')
	parser.add_argument('-save_every_steps', type=int, default=1000, help='how often to checkpoint')
	parser.add_argument('-keep_every_steps', type=int, default=100000, help='permanent checkpoints')
	parser.add_argument('-resume_id', type=str, default='',  help='resume id')
	parser.add_argument('-analyze', dest='analyze', action='store_true', help='analyze')
	parser.add_argument('-no-analyze', dest='analyze', action='store_false', help='Do not  analyze')
	parser.set_defaults(analyze=False)

	# Curriculum
	parser.add_argument('-curriculum_dims_start', type=int, default=28, help='initial parameter')
	parser.add_argument('-curriculum_dims_end', type=int, default=28, help='limit of final value')
	parser.add_argument('-curriculum_dims_inc', type=int, default=1, help='how much to increment each time')
	parser.add_argument('-curriculum_dims_interval', type=int, default=2000, help='increment every how many steps')
	
	parser.add_argument('-curriculum_points_start', type=int, default=40, help='initial parameter')
	parser.add_argument('-curriculum_points_end', type=int, default=40, help='limit of final value')
	parser.add_argument('-curriculum_points_inc', type=int, default=5, help='how much to increment each time')
	parser.add_argument('-curriculum_points_interval', type=int, default=2000, help='increment every how many steps')

	# Prefix Scoring
	parser.add_argument('-prefix_score_train', dest='prefix_score_train', action='store_true', help='Calculate prefix scores during training')
	parser.add_argument('-no-prefix_score_train', dest='prefix_score_train', action='store_false', help='Do not calculate prefix scores during training')
	parser.set_defaults(prefix_score_train=False)
	parser.add_argument('-prefix_score_eval', dest='prefix_score_eval', action='store_true', help='Calculate prefix scores during evaluation')
	parser.add_argument('-no-prefix_score_eval', dest='prefix_score_eval', action='store_false', help='Do not calculate prefix scores during evaluation')
	parser.set_defaults(prefix_score_eval=False)
	parser.add_argument('-prefix_score_train_interval', type=int, default=500, help='calculate prefix score after every how many steps')
	parser.add_argument('-prefix_score_n_repeats', type=int, default=4, help='How many times to repeat sequence')
	parser.add_argument('-prefix_score_n_points', type=int, default=12, help='How many points in sequence')
	parser.add_argument('-prefix_score_bsize', type=int, default=100, help='Average score over how many examples - should not be more than bsize')

	# nearest neighbours Scoring
	parser.add_argument('-nn_score_train', dest='nn_score_train', action='store_true', help='Calculate nn scores during training')
	parser.add_argument('-no-nn_score_train', dest='nn_score_train', action='store_false', help='Do not calculate nn scores during training')
	parser.set_defaults(nn_score_train=False)
	parser.add_argument('-nn_score_eval', dest='nn_score_eval', action='store_true', help='Calculate nn scores during evaluation')
	parser.add_argument('-no-nn_score_eval', dest='nn_score_eval', action='store_false', help='Do not calculate nn scores during evaluation')
	parser.set_defaults(nn_score_eval=False)
	parser.add_argument('-nn_score_train_interval', type=int, default=500, help='calculate nn score after every how many steps')
	parser.add_argument('-nn_score_n_points', type=int, default=80, help='How many points in sequence')
	parser.add_argument('-nn_score_start_point', type=int, default=41, help='From which point to start nn scoring')
	parser.add_argument('-nn_score_bsize', type=int, default=100, help='Average score over how many examples - should not be more than bsize')
	parser.add_argument('-nn_score_num_neighbours', type=int, default=1, help='How many nearest neighbours to observe for attn weights?')

	# Wandb
	parser.add_argument('-project', type=str, default='in-context-test', help='wandb project name')
	parser.add_argument('-entity', type=str, default='name', help='wandb entity name')
	parser.add_argument('-notes', type=str, default='', help='wandb notes')
	parser.add_argument('-name', type=str, default='conj_test', help='run name')
	parser.add_argument('-log_every_steps', type=int, default=100, help='wandb log every how many steps')
	


	return parser
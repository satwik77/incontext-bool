# python -m src.train -wandb -family llama -model_name llama7b -freeze 1 -task conjunction -entity name -project in-context-Tune -name llama7b_1e6 -task_kwargs {} -train_steps 30000 -batch_size 32 -n_dims 28 -curriculum_dims_start 28 -curriculum_dims_end 28 -curriculum_points_start 80 -curriculum_points_end 80 -prefix_score_train -prefix_score_eval -nn_score_train -nn_score_eval -precision half -learning_rate 0.000001
import os
from random import randint
import uuid
import numpy as np
from tqdm import tqdm
import torch
import yaml
import pdb

from src.eval import get_run_metrics
import src.tasks as tasks
from src.bool_tasks import get_task_sampler
from src.samplers import get_data_sampler
from src.curriculum import Curriculum
from src.args import build_parser
from src.models import build_model
from src.attention_analysis import prefix_scoring_step, nn_scoring_step
from src.utils import model_dist, model_sim
from src.remove_pt import delete_pt_files
import copy

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, start_idx = 0, precision='full'):
	optimizer.zero_grad()
	output = model(xs, ys)

	if precision == 'half':
		ys = ys.to(torch.float16)

	loss = loss_func(output[:, start_idx:], ys[:, start_idx:])
	loss.backward()
	optimizer.step()
	return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
	seeds = set()
	while len(seeds) < count:
		seeds.add(randint(0, total_seeds - 1))
	return seeds


def train(model, args):
	if args.precision == 'half':
		optimizer = torch.optim.Adam(model.parameters(), eps=1e-4, lr=args.learning_rate)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	curriculum = Curriculum(args)
	task = args.task
	device = model.device

	init_model = copy.deepcopy(model)

	starting_step = 0
	state_path = os.path.join(args.out_dir, "state.pt")
	if os.path.exists(state_path):
		state = torch.load(state_path)
		model.load_state_dict(state["model_state_dict"])
		optimizer.load_state_dict(state["optimizer_state_dict"])
		starting_step = state["train_step"]
		for i in range(state["train_step"] + 1):
			curriculum.update()

	n_dims = model.n_dims
	bsize = args.batch_size

	start_idx=0
	if "start_idx" in args.task_kwargs and args.task == "nearest_neighbours":
		start_idx = args.task_kwargs["start_idx"]

	if 'bool' not in args.data:
		task_sampler = tasks.get_task_sampler(
			args.task,
			n_dims,
			bsize,
			num_tasks=args.num_tasks,
			**args.task_kwargs,
		)
	else:
		task_sampler = get_task_sampler(
		args.task,
		n_dims,
		bsize,
		n_points = curriculum.n_points,
		num_tasks=args.num_tasks,
		**args.task_kwargs,
	)

	if 'bool' not in args.data:
		data_sampler = get_data_sampler(args.data, n_dims=n_dims)

	pbar = tqdm(range(starting_step, args.train_steps))

	num_training_examples = args.num_training_examples

	for i in pbar:
		data_sampler_args = {}
		task_sampler_args = {}


		if num_training_examples != 0:
			assert num_training_examples >= bsize
			seeds = sample_seeds(num_training_examples, bsize)
			data_sampler_args["seeds"] = seeds
			task_sampler_args["seeds"] = [s + 1 for s in seeds]

		task = task_sampler(**task_sampler_args)
		if 'bool' not in args.data:
			xs = data_sampler.sample_xs(
				curriculum.n_points,
				bsize,
				curriculum.n_dims_truncated,
				**data_sampler_args,
			)
		else:
			xs = task.sample_xs(
				curriculum.n_points,
				bsize,
			)
		
		ys = task.evaluate(xs)
		
		if args.noise_rate > 0:
			n_points = curriculum.n_points
			ns = args.noise_rate
			noise_mat = torch.tensor(np.random.choice([1, -1], size=(bsize, n_points), p=[1-ns, ns]), dtype=torch.float)
			ys = ys * noise_mat
		

		loss_func = task.get_training_metric()

		loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func, start_idx, precision=args.precision)

		if args.prefix_score_train:
			if i%args.prefix_score_train_interval == 0:
				task = task_sampler()
				if 'bool' not in args.data:
					xs_prefix = data_sampler.sample_xs(
						args.prefix_score_n_points,
						bsize,
						curriculum.n_dims_truncated,
						**data_sampler_args,
					)
				else:
					xs_prefix = task.sample_xs(
						args.prefix_score_n_points,
						bsize,
					) # bs x n_points x n_dims

				ls_xs = [xs_prefix for rep in range(args.prefix_score_n_repeats)]

				xs_prefix = torch.cat(ls_xs, dim=1)
				
				ys_prefix = task.evaluate(xs_prefix)

				xs_prefix = xs_prefix[:args.prefix_score_bsize]
				ys_prefix = ys_prefix[:args.prefix_score_bsize]

				prefix_score, num_induc_heads, prefix_head_scores = prefix_scoring_step(model, xs_prefix.to(device), ys_prefix.to(device), num_repeats=args.prefix_score_n_repeats)
		else:
			prefix_score, num_induc_heads, prefix_head_scores = 0, 0, {"not_run": 0}

		if args.nn_score_train:
			if i%args.nn_score_train_interval == 0:
				task = task_sampler()
				if 'bool' not in args.data:
					xs_nn = data_sampler.sample_xs(
						args.nn_score_n_points,
						bsize,
						curriculum.n_dims_truncated,
						**data_sampler_args,
					)
				else:
					xs_nn = task.sample_xs(
						args.nn_score_n_points,
						bsize,
					) # bs x n_points x n_dims
				
				ys_nn = task.evaluate(xs_nn)

				xs_nn = xs_nn[:args.nn_score_bsize]
				ys_nn = ys_nn[:args.nn_score_bsize]

				nn_score, num_nn_heads, nn_head_scores = nn_scoring_step(model, xs_nn.to(device), ys_nn.to(device), start_point=args.nn_score_start_point, num_neighbours=args.nn_score_num_neighbours, device=device)
		else:
			nn_score, num_nn_heads, nn_head_scores = 0, 0, {"not_run": 0}

		point_wise_tags = list(range(curriculum.n_points))
		point_wise_loss_func = task.get_metric()
		point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)
		mean_acc = point_wise_loss.mean().item()
		null_pred = torch.zeros_like(ys) - 1
		null_acc = point_wise_loss_func(null_pred, ys).mean().item()

		baseline_loss = (
			sum(
				max(curriculum.n_dims_truncated - ii, 0)
				for ii in range(curriculum.n_points)
			)
			/ curriculum.n_points
		)
		
		
		if args.wandb:
			if i % args.log_every_steps == 0 and not args.test_run:
				init_distance = model_dist(curr_model= model, init_model= init_model, weight_only=True)
				wandb.log(
					{
						"mean_acc": mean_acc,
						"null_acc": null_acc,
						"overall_loss": loss,
						"init_distance": init_distance,
						# "misc/excess_loss": loss / baseline_loss,                        
						"pointwise/loss": dict(
							zip(point_wise_tags, point_wise_loss.cpu().numpy())
						),
						"misc/n_points": curriculum.n_points,
						"misc/n_dims": curriculum.n_dims_truncated
					},
					step=i,
				)
				if args.analyze:
					grad_vals = []

					for name, param in model.named_parameters():
						if param.requires_grad and param.grad is not None:
							grad_val = param.grad.norm().item()
							grad_vals.append(grad_val)
							wandb.log(
								{
									f"grads/{name}": grad_val,
								},
								step=i,
							)
					wandb.log(
						{
							"misc/mean": np.mean(grad_vals),							
							"misc/max": np.max(grad_vals),
							"misc/min": np.min(grad_vals),
							"misc/std": np.std(grad_vals),
						},
						step=i,
					)
			if args.nn_score_train and i%args.nn_score_train_interval == 0:
				wandb.log(
					{
						"nn_score": nn_score,
						"num_nn_heads": num_nn_heads,
						"nn_scores/score": nn_head_scores
					},
					step=i,
				)
			if args.prefix_score_train and i%args.prefix_score_train_interval == 0:
				wandb.log(
					{
						"prefix_score": prefix_score,
						"num_induction_heads": num_induc_heads,
						"prefix_scores/score": prefix_head_scores
					},
					step=i,
				)

		curriculum.update()

		pbar.set_description(f"loss {loss}")
		if i % args.save_every_steps == 0 and not args.test_run:
			training_state = {
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"train_step": i,
			}
			torch.save(training_state, state_path)

			if mean_acc > 0.9999 and args.task not in ['linear_regression', 'sparse_linear_regression', 'relu_2nn_regression', 'decision_tree']:
				break 

		if (
			args.keep_every_steps > 0
			and i % args.keep_every_steps == 0
			and not args.test_run
			and i > 0
		):
			torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
	if args.test_run:
		args.curriculum_points_start = args.curriculum_points_end
		args.curriculum_dims_start = args.curriculum_dims_end
		args.train_steps = 100
	else:
		
		args.curriculum_dims_end = args.n_dims
		args.curriculum_dims_start = args.curriculum_dims_end
		if args.wandb:

			wandb.init(
				dir=args.out_dir,
				project=args.project,
				group = str(args.task),
				entity=args.entity,
				config=args.__dict__,
				notes=args.notes,
				name=args.name,                
				resume=True,
			)

	

	device = torch.device("cuda:{}".format(args.gpu))

	model = build_model(args)
	model.to(device)
	model.device = device
	model.train()

	train(model, args)

	if not args.test_run:
		eval_metrics = get_run_metrics(args.out_dir, device=device)  # precompute metrics for eval
	
	if args.wandb:
		eval_metrics = eval_metrics['standard']
		eval_models = list(eval_metrics.keys())
		plot_y = []

		val_acc = eval_metrics[model.name]['mean']
		mean_val_acc = np.mean(val_acc)

		wandb.log(
					{
						"mean_val_acc": mean_val_acc,
					},
				)

		for model_name in eval_models:
			plot_y.append(eval_metrics[model_name]['mean'])
		plot_x = list(range(len(plot_y[0])))

		wandb.log({'eval/mean_acc': wandb.plot.line_series(
										plot_x, 
										plot_y, 
										keys=eval_models,
										title='Accuracy of Different Models',
										xname='Incontext Examples',                                
										)})
	
	if args.delete:
		print('Deleting model (pt) files...')
		delete_pt_files(args.out_dir)

	


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()
	
	print(f"Running with: {args}")

	if not args.test_run:
		run_id = args.resume_id
		if run_id == "":
			run_id = str(uuid.uuid4())[:20]
			print(f"Run ID: {run_id}")
			
			args.name += '_' + args.family
			if args.family in ['gpt', 'mysan', 'attclf']:
				args.name += '_' + args.model_name
			args.name += '_' + run_id[:8]

		args.out_dir = os.path.join(args.out_dir, args.task)
		out_dir = args.out_dir + '_' + args.family
		out_dir = os.path.join(args.out_dir, args.name)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		args.out_dir = out_dir


		with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
			yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

	main(args)

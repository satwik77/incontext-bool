import json
import os
import sys
import pdb

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import src.models as models
from src.samplers import get_data_sampler, sample_transformation
from src.bool_tasks import get_task_sampler
from src.attention_analysis import prefix_scoring_step, nn_scoring_step
import src.tasks as tasks
import src.baselines as baselines


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation


def eval_batch(model, device, task_sampler, xs, xs_p=None):
    task = task_sampler()

    if xs_p is None:
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


def eval_batch_bool(model, task_sampler, n_points, bsize, device):
    task = task_sampler()

    xs = task.sample_xs(n_points, bsize)
    ys = task.evaluate(xs)
    if isinstance(model, baselines.NNCosineModel):
        pred = model(xs, ys)
        metrics = task.get_metric()(pred.cpu(), ys)
    elif isinstance(model, baselines.GDModel):
        pred, gt = model(xs.to(device), ys.to(device))
        pred = pred.detach()
        gt = gt.detach()
        metrics = task.get_metric()(pred.cpu(), gt.cpu())
    else:
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)

    # if isinstance(model, baselines.NNCosineModel):
    #     if not torch.equal(metrics[:, 5:], torch.ones(bsize, 5)):
    #         pdb.set_trace()
    #         ys_temp = task.evaluate(xs)

    return metrics

def get_prefix_score(model, task_sampler, n_points, bsize, prefix_bsize, device, num_repeats=4):
    task = task_sampler()

    xs = task.sample_xs(n_points, bsize)

    ls_xs = [xs for rep in range(num_repeats)]
    xs = torch.cat(ls_xs, dim=1)

    ys = task.evaluate(xs)

    xs = xs[:prefix_bsize]
    ys = ys[:prefix_bsize]

    prefix_score, num_induc_heads, head_scores_dict = prefix_scoring_step(model, xs, ys, num_repeats)
    model.eval()

    return prefix_score, num_induc_heads, head_scores_dict

def get_nn_score(model, task_sampler, n_points, bsize, nn_bsize, start_point, num_neighbours, device):
    task = task_sampler()

    xs = task.sample_xs(n_points, bsize)
    ys = task.evaluate(xs)

    xs = xs[:nn_bsize]
    ys = ys[:nn_bsize]

    xs = xs.to(device)
    ys = ys.to(device)

    nn_score, num_nn_heads, head_scores_dict = nn_scoring_step(model, xs, ys, start_point, num_neighbours, device)
    model.eval()

    return nn_score, num_nn_heads, head_scores_dict

# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}

def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prefix_score_n_points,
    prefix_score_bsize,
    prefix_score_n_repeats,
    prefix_score_eval,
    nn_score_n_points,
    nn_score_bsize,
    nn_score_start_point,
    nn_score_num_neighbours,
    nn_score_eval,
    prompting_strategy,
    device,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    
    task_sampler = tasks.get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    if 'bool' not in data_name:
        data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, device, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)



def eval_model_bool(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prefix_score_n_points,
    prefix_score_bsize,
    prefix_score_n_repeats,
    prefix_score_eval,
    nn_score_n_points,
    nn_score_bsize,
    nn_score_start_point,
    nn_score_num_neighbours,
    nn_score_eval,
    prompting_strategy,
    device,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0

    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, n_points, **task_sampler_kwargs
    )


    all_metrics = []
    all_prefix_scores = 0
    all_induc_heads = 0
    all_prefix_heads = {}
    tot_p = 0
    all_nn_scores = 0
    all_nn_heads = 0
    all_nn_heads_dict = {}
    tot_n = 0

    # generating_func = globals()[f"gen_{prompting_strategy}"]

    for i in range(num_eval_examples // batch_size):
        # xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch_bool(model, task_sampler, n_points, batch_size, device)
        all_metrics.append(metrics)

        if isinstance(model, models.TransformerModel):
            if prefix_score_eval:
                prefix_score, num_induc_heads, prefix_head_scores = get_prefix_score(model, task_sampler, prefix_score_n_points, batch_size, prefix_score_bsize, device, num_repeats=prefix_score_n_repeats)
                all_prefix_scores += prefix_score
                all_induc_heads += num_induc_heads
                tot_p += 1
                for key in prefix_head_scores:
                    if key in all_prefix_heads:
                        all_prefix_heads[key].append(prefix_head_scores[key])
                    else:
                        all_prefix_heads[key] = [prefix_head_scores[key]]
            if nn_score_eval:
                nn_score, num_nn_heads, nn_head_scores = get_nn_score(model, task_sampler, nn_score_n_points, batch_size, nn_score_bsize, nn_score_start_point, nn_score_num_neighbours, device)
                all_nn_scores += nn_score
                all_nn_heads += num_nn_heads
                tot_n += 1
                for key in nn_head_scores:
                    if key in all_nn_heads_dict:
                        all_nn_heads_dict[key].append(nn_head_scores[key])
                    else:
                        all_nn_heads_dict[key] = [nn_head_scores[key]]

    if isinstance(model, models.TransformerModel):
        if prefix_score_eval:
            print("Prefix Score: ", all_prefix_scores/tot_p)
            print("Number of Induction Heads: ", all_induc_heads/tot_p)
            print("Prefix scores for each head: ")
            for key in all_prefix_heads:
                print(key + ": " + str(np.mean(np.array(all_prefix_heads[key]))))
            print()
        if nn_score_eval:
            print("NN Score: ", all_nn_scores/tot_n)
            print("Number of NN Heads: ", all_nn_heads/tot_n)
            print("NN scores for each head: ")
            for key in all_nn_heads_dict:
                print(key + ": " + str(np.mean(np.array(all_nn_heads_dict[key]))))
            print()

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.n_dims
    n_points = conf.curriculum_points_end
    batch_size = conf.batch_size

    task_name = conf.task
    data_name = conf.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
        "prefix_score_n_points": conf.prefix_score_n_points,
        "prefix_score_bsize": conf.prefix_score_bsize,
        "prefix_score_n_repeats": conf.prefix_score_n_repeats,
        "prefix_score_eval": conf.prefix_score_eval,
        "nn_score_n_points": conf.nn_score_n_points,
        "nn_score_bsize": conf.nn_score_bsize,
        "nn_score_start_point": conf.nn_score_start_point,
        "nn_score_num_neighbours": conf.nn_score_num_neighbours,
        "nn_score_eval": conf.nn_score_eval
    }

    base_kwargs["task_sampler_kwargs"] = {}
    if "start_idx" in conf.task_kwargs:
        base_kwargs["task_sampler_kwargs"]["start_idx"] = conf.task_kwargs["start_idx"]
    # else:
    #     base_kwargs["task_sampler_kwargs"]["start_idx"] = 0

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)

        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, device, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}
    
    data_type = evaluation_kwargs["standard"]["data_name"]

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue
            
            if data_type == "boolean":
                
                metrics[model.name] = eval_model_bool(model, device=device, **kwargs)
            else:
                metrics[model.name] = eval_model(model, device=device, **kwargs)

        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, device, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        model.to(device)

        model = model.eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.task, conf.n_dims)
    evaluation_kwargs = build_evals(conf)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, device = device, save_path = save_path, recompute=recompute)
    return all_metrics



def conf_to_model_name(conf):
    if conf.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.n_layer, conf.n_head)]
    else:
        return conf.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    # for task in os.listdir(run_dir):
    for task in ["conjunction"]:
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
        # for run_id in ["0fd47ac6-96a9-46f2-8bd6-fde24eed1fff"]:
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.task_kwargs.items()
            )
            num_tasks = (
                conf.num_tasks if "num_tasks" in conf else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.num_training_examples
                if "num_training_examples" in conf
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.n_dims
            params["n_layer"] = conf.n_layer
            params["n_head"] = conf.n_head
            params["run_name"] = conf.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
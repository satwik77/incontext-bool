import math

import torch
import numpy as np
import pdb


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    '''
    ys_pred: [-inf, inf]
    ys: {-1, 1}
    '''
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    def pac_learn(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=0, **kwargs
):
    task_names_to_classes = {
        "conjunction": Conjunction,
        "mono_conjunction": MonoConjunction,
        "disjunction": Disjunction,
        "parity": Parity,
        "sparse_parity": SparseParity,
        'sparse_halfspace': SparseHalfspace,
        "halfspace": Halfspace,
        "majority": Majority,
        "full_majority": FullMajority,
        "int_halfspace": IntHalfspace,
        "dnf": DNF,
        "cnf": CNF,
        'sparse_thres': SparseThreshold,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks > 0:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError






class Conjunction(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(Conjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        if n_dims <= 7:
            self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.5, 0.25, 0.25]), dtype=torch.float)
        else:
            self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
        self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()
        for b in range(b_size):
            wb, k = self.w_b[b], self.kw[b]            
            pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
            nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
            for i in range(n_points):
                if np.random.choice([0, 1], p=[0.7, 0.3]):
                    xs_b[b, i, pidx] = +1.0
                    xs_b[b, i, nidx] = -1.0
                    assert (xs_b[b, i, :] @ wb).squeeze() >= k

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() - self.kw
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy



class MonoConjunction(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(MonoConjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        if n_dims <= 7:
            self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.5, 0.5]), dtype=torch.float)
        else:
            self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)
        self.kw = self.w_b.sum(dim=1) - 1
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()
        for b in range(b_size):
            wb, k = self.w_b[b], self.kw[b]            
            tidx = [i for i in range(self.n_dims) if wb[i] == 1]
            for i in range(n_points):
                if np.random.choice([0, 1], p=[0.7, 0.3]):
                    xs_b[b, i, tidx] = +1.
                    assert (xs_b[b, i, :] @ wb).squeeze() >= k

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() - self.kw
        return ys_b.sign()

    def get_hypo_op(self, xs, hypo):
        temp_kw = hypo.sum() - 1
        temp_op = (torch.tensor(xs) @ hypo).squeeze() - temp_kw
        return temp_op.sign().item()
    
    def pac_learn(self, xs, ys):
        cur_hypo = torch.ones(self.n_dims)
        for i in range(len(xs)-1):
            cur_x = xs[i]
            cur_y = ys[i]
            if cur_y == 1:
                h_val = self.get_hypo_op(cur_x, cur_hypo)
                if h_val != cur_y:
                    for j in range(len(cur_hypo)):
                        if cur_x[j] == -1:
                            cur_hypo[j] = 0
        return self.get_hypo_op(xs[-1], cur_hypo)
        

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy
    





class Disjunction(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(Disjunction, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        if n_dims <= 7:
            self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.5, 0.25, 0.25]), dtype=torch.float)
        else:
            self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
        self.kw = torch.norm(self.w_b, p=1, dim=1) - 1
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()

        # Manipulate the input to create negative examples to make a more balanced dataset
        for b in range(b_size):
            wb, k = self.w_b[b], self.kw[b]            
            pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
            nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
            for i in range(n_points):
                if np.random.choice([0, 1], p=[0.7, 0.3]):
                    xs_b[b, i, pidx] = -1.0
                    xs_b[b, i, nidx] = +1.0
                    assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() + self.kw
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy






class SparseThreshold(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(SparseThreshold, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        self.w_b = torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.15, 0.15]), dtype=torch.float)
        thres_bound = 3
        self.kw = torch.randint(-thres_bound, thres_bound, (self.b_size, 1),  dtype= torch.float) + 0.5
        
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()
        # for b in range(b_size):
        #     wb, k = self.w_b[b], self.kw[b]            
        #     pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
        #     nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
        #     for i in range(n_points):
        #         if np.random.choice([0, 1], p=[0.7, 0.3]):
        #             xs_b[b, i, pidx] = +1.0
        #             xs_b[b, i, nidx] = -1.0
        #             assert (xs_b[b, i, :] @ wb).squeeze() >= k

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() - self.kw
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy




class SparseHalfspace(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(SparseHalfspace, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
    

        k = 3
        wb = []
        for i in range(self.b_size):
            idx = np.random.choice(range(self.n_dims), k, replace=False)
            w = np.zeros(self.n_dims)
            w[idx] = 1
            wb.append(w)
        
        wb = np.array(wb)
        w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
        w_z = torch.randn(self.b_size, self.n_dims, 1)
        self.w_b = w_b * w_z


    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() 
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy




class Halfspace(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(Halfspace, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
    
        # w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
        # w_z = torch.randn(self.b_size, self.n_dims, 1)
        # self.w_b = w_b * w_z

        self.w_b = torch.randn(self.b_size, self.n_dims, 1)


    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() 
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy




class IntHalfspace(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(IntHalfspace, self).__init__(n_dims, batch_size, pool_dict, seeds)
        bound = 3
        self.w_b = torch.randint(-bound, bound+1, (self.b_size, self.n_dims, 1),  dtype= torch.float)

    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        return xs_b

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() - 0.5
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy





class Majority(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(Majority, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        if n_dims <= 7:
            self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.5, 0.5]), dtype=torch.float)
        else:
            self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.7, 0.3]), dtype=torch.float)
        # self.kw = self.w_b.sum(dim=1) - 1
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

        return xs_b

        
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() - 0.5
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy
    




class FullMajority(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(FullMajority, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        self.w_b = torch.tensor(np.random.choice([-1, 1], size=(self.b_size, self.n_dims, 1), p=[0.5, 0.5]), dtype=torch.float)
        
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        return xs_b

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = (xs_b @ w_b).squeeze() - 0.5
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy
    



class Parity(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(Parity, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        # Approximate 35% of indices will be 1
        funcs = np.random.choice(2**n_dims, size = batch_size)
        all_subsets  = self.generate_subsets(n_dims)
        self.w_b = torch.zeros(size= (batch_size, n_dims, 1))
        # self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
        for i in range(batch_size):
            self.w_b[i, all_subsets[funcs[i]]] = 1
        
    
    def sample_xs(self, n_points, b_size):
        # Input distribution is uniform over {-1, 1}^n_dims
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

        return xs_b

        
    def evaluate(self, xs_b):
        # Output \in {-1, 1}
        xt = (xs_b.clone() +1)/2
        w_b = self.w_b.to(xs_b.device)
        ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
        return ys_b.sign()


    def generate_subsets(self, n):
        subsets = []
        for i in range(2**n):
            subset = [j for j in range(n) if (i & 1 << j)]
            subsets.append(subset)
        return subsets
    
    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy



# class Parity(Task):
#     def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
#         super(Parity, self).__init__(n_dims, batch_size, pool_dict, seeds)
#         # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
#         # Approximate 35% of indices will be 1
#         self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
        
    
#     def sample_xs(self, n_points, b_size):
#         # Input distribution is uniform over {0, 1}^n_dims
#         xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)

#         return xs_b

        
#     def evaluate(self, xs_b):
#         # Output \in {-1, 1}
#         w_b = self.w_b.to(xs_b.device)
#         ys_b = ((xs_b @ w_b).squeeze() % 2) * 2 - 1
#         return ys_b.sign()

#     @staticmethod
#     def get_metric():
#         return accuracy

#     @staticmethod
#     def get_training_metric():
#         return cross_entropy







class SparseParity(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(SparseParity, self).__init__(n_dims, batch_size, pool_dict, seeds)

        k = 2
        wb = []
        for i in range(self.b_size):
            idx = np.random.choice(range(self.n_dims), k, replace=False)
            w = np.zeros(self.n_dims)
            w[idx] = 1
            wb.append(w)
        
        wb = np.array(wb)
        self.w_b = torch.tensor(wb, dtype=torch.float).unsqueeze(2)
        # self.w_b = torch.tensor(np.random.choice([0, 1], size=(self.b_size, self.n_dims, 1), p=[0.65, 0.35]), dtype=torch.float)
        
    
    def sample_xs(self, n_points, b_size):
        # Input distribution is uniform over {0, 1}^n_dims
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1

        return xs_b

        
    def evaluate(self, xs_b):
        # Output \in {-1, 1}
        xt = (xs_b.clone() +1)/2
        w_b = self.w_b.to(xs_b.device)
        ys_b = ((xt @ w_b).squeeze() % 2) * 2 - 1
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy



# Three DNF Task named DNF for simplicity. Complete DNF is hard to learn complexity-wise, so we use a 3-clause DNF
class DNF(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(DNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        self.w_b = [torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.8, 0.1, 0.1]), dtype=torch.float) for i in range(3)] # Create 3 clauses
        self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()
        # Manipulate the input to create positive examples to make a more balanced dataset
        for b in range(b_size):
            cid = np.random.choice([0, 1, 2])        # Choose a clause
            wb, k = self.w_b[cid][b], self.kw[cid][b]
            pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
            nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
            for i in range(n_points):
                if np.random.choice([0, 1], p=[0.65, 0.35]):
                    xs_b[b, i, pidx] = +1.0
                    xs_b[b, i, nidx] = -1.0
                    assert (xs_b[b, i, :] @ wb).squeeze() >= k

        return xs_b

        
    def evaluate(self, xs_b):
        w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
        ys_bs = [(xs_b @ w_bs[i]).squeeze() - self.kw[i] for i in range(3)]
        ys_bs = [ys_bs[i].sign() for i in range(3)]
        # Combine stack three tensors into one
        ys_b = torch.stack(ys_bs, dim=2).max(dim=2)[0]  # 0th Index is the value, 1st index has indices
        
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy



class CNF(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(CNF, self).__init__(n_dims, batch_size, pool_dict, seeds)
        # self.w_b = torch.randint(0, 2, (self.b_size, self.n_dims, 1))
        self.w_b = [torch.tensor(np.random.choice([0, 1, -1], size=(self.b_size, self.n_dims, 1), p=[0.80, 0.1, 0.1]), dtype=torch.float) for i in range(3)] # Create 3 clauses
        self.kw = [torch.norm(self.w_b[i], p=1, dim=1) - 1 for i in range(3)]
    
    def sample_xs(self, n_points, b_size):
        xs_b = torch.randint(0, 2, (b_size, n_points, self.n_dims), dtype= torch.float)*2-1
        # pdb.set_trace()
        
        # Manipulate the input to create negative examples to make a more balanced dataset
        for b in range(b_size):
            cid = np.random.choice([0, 1, 2])        # Choose a clause
            wb, k = self.w_b[cid][b], self.kw[cid][b]
            pidx = [i for i in range(self.n_dims) if wb[i] == 1.0]
            nidx = [i for i in range(self.n_dims) if wb[i] == -1.0]
            for i in range(n_points):
                if np.random.choice([0, 1], p=[0.7, 0.3]):
                    xs_b[b, i, pidx] = -1.0
                    xs_b[b, i, nidx] = +1.0
                    assert (xs_b[b, i, :] @ wb).squeeze() < -1*k

        return xs_b

        
    def evaluate(self, xs_b):
        w_bs = [self.w_b[i].to(xs_b.device) for i in range(3)]
        ys_bs = [(xs_b @ w_bs[i]).squeeze() + self.kw[i] for i in range(3)]
        ys_bs = [ys_bs[i].sign() for i in range(3)]
        # Combine stack three tensors into one
        ys_b = torch.stack(ys_bs, dim=2).min(dim=2)[0]  # 0th Index is the value, 1st index has indices
        
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


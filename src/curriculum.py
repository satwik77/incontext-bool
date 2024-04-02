import math


class Curriculum:
	def __init__(self, args):
		# args.dims and args.points each contain start, end, inc, interval attributes
		# inc denotes the change in n_dims,
		# this change is done every interval,
		# and start/end are the limits of the parameter
		self.args = args
		self.n_dims_truncated = args.curriculum_dims_start
		self.n_points = args.curriculum_points_start
		self.step_count = 0

	def update(self):
		self.step_count += 1
		self.n_dims_truncated = self.update_var_dims(self.n_dims_truncated)
		self.n_points = self.update_var_points(self.n_points)

	def update_var_dims(self, var):
		if self.step_count % self.args.curriculum_dims_interval == 0:
			var += self.args.curriculum_dims_inc

		return min(var, self.args.curriculum_dims_end)
	
	def update_var_points(self, var):
		if self.step_count % self.args.curriculum_points_interval == 0:
			var += self.args.curriculum_points_inc

		return min(var, self.args.curriculum_points_end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
	final_var = init_var + math.floor((total_steps) / n_steps) * inc

	return min(final_var, lim)

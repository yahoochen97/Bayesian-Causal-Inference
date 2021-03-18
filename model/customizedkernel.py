import torch
import gpytorch
from gpytorch.constraints.constraints import Interval, Positive
from gpytorch.lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.kernels.kernel import Kernel


class myIndexKernel(Kernel):

    def __init__(self, num_tasks, rank=1, prior=None, var_constraint=None, **kwargs):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)

        self.register_parameter(
            name="rho", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, 1))
        )
        # self.rho = torch.nn.Parameter(torch.zeros(1))
        self.num_tasks = num_tasks

        if var_constraint is None:
            var_constraint = Interval(-1,1)
        
        self.register_constraint("rho", var_constraint)

    def covar_matrix(self):
        res = torch.eye(self.num_tasks)
        # hyperbolic tangent transform from R to [-1,1]
        tmp = torch.exp(self.rho)
        tmp = (tmp-1/tmp)/(tmp+1/tmp)
        res[0,1] = tmp
        res[1,0] = tmp
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self.covar_matrix()
        index_shape = i1.shape[-2:]

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(index_shape),
            right_interp_indices=i2.expand(index_shape),
        )
        return res


class constantKernel(Kernel):
    r"""
    A kernel for constant value.

    Args:
        :attr:`c` (int):
            constant
        :attr:`var_constraint` (Constraint, optional):
            Constraint for added diagonal component. Default: `Positive`.

    """

    def __init__(self, num_tasks, prior=None, var_constraint=None, **kwargs):
        super().__init__(**kwargs)

        self.c2 = torch.nn.Parameter(torch.ones(1))
        self.num_tasks = num_tasks

        # if var_constraint is None:
        #     var_constraint = Interval(0,1)

        # self.register_constraint("c2", var_constraint)

    def forward(self, x1, x2, **params):
        res = torch.ones(x1.shape[0],x2.shape[0])*self.c2
        return res


class myIndicatorKernel(Kernel):
    r"""
    A kernel for discrete indices. Kernel is defined by a lookup table.

    """

    def __init__(self, num_tasks, **kwargs):
        super().__init__(**kwargs)

        self.num_tasks = num_tasks

    def covar_matrix(self):
        res = torch.eye(self.num_tasks)
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self.covar_matrix()
        index_shape = i1.shape[-2:]

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(index_shape),
            right_interp_indices=i2.expand(index_shape),
        )
        return res
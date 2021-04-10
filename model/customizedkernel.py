import torch
import gpytorch
from gpytorch.constraints.constraints import Interval, Positive
from gpytorch.lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.kernels.kernel import Kernel


class myIndexKernel(Kernel):

    def __init__(self, num_tasks, rho_prior=None, var_constraint=None, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(
            name="raw_rho", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, 1))
        )
        self.num_tasks = num_tasks

        if var_constraint is None:
            var_constraint = Interval(-1,1)
        
        self.register_constraint("raw_rho", var_constraint)

        if rho_prior is not None:
            self.register_prior("rho_prior", rho_prior, lambda m: m.rho, lambda m, v: m._set_rho(v))

    @property
    def rho(self):
        return self.raw_rho_constraint.transform(self.raw_rho)

    @rho.setter
    def rho(self, value):
        self._set_rho(value)

    def _set_rho(self, value):
        self.initialize(raw_rho=self.raw_rho_constraint.inverse_transform(value))

    def covar_matrix(self):
        res = torch.eye(self.num_tasks)
        if self.batch_shape:
            res = res.unsqueeze(0).repeat(self.batch_shape[0], 1, 1)
        # hyperbolic tangent transform from R to [-1,1]
        # tmp = torch.exp(self.rho)
        # tmp = (tmp-1/tmp)/(tmp+1/tmp)
            for i in range(self.batch_shape[0]):
                res[i,0,1] = self.rho[i]
                res[i,1,0] = self.rho[i]
        else:
            res[0,1] = self.rho
            res[1,0] = self.rho
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self.covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], self.batch_shape)
        index_shape = batch_shape + i1.shape[-2:]

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
        :attr:`c2` (int):
            constant
        :attr:`var_constraint` (Constraint, optional):
            Constraint for added diagonal component. Default: `Interval(0,1)`.

    """

    def __init__(self, num_tasks, prior=None, var_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.register_parameter(
            name="raw_c2", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, 1))
        )
        
        if var_constraint is None:
            var_constraint = Positive()

        self.register_constraint("raw_c2", var_constraint)

        if prior is not None:
            self.register_prior("c2_prior", prior, lambda m: m.c2, lambda m, v: m._set_c2(v))

    @property
    def c2(self):
        return self.raw_c2_constraint.transform(self.raw_c2)

    @c2.setter
    def c2(self, value):
        self._set_c2(value)

    def _set_c2(self, value):
        self.initialize(raw_c2=self.raw_c2_constraint.inverse_transform(value))


    def forward(self, x1, x2, **params):
        if self.batch_shape:
            res = torch.ones(x1.shape[1],x2.shape[1])*self.c2
            res = res.unsqueeze(0).repeat(self.batch_shape[0], 1, 1)
        else:
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
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], self.batch_shape)
        index_shape = batch_shape + i1.shape[-2:]

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(index_shape),
            right_interp_indices=i2.expand(index_shape),
        )
        return res
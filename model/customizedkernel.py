import torch
import gpytorch
from gpytorch.constraints.constraints import Interval, Positive
from gpytorch.lazy import DiagLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.kernels.kernel import Kernel
from gpytorch.means.mean import Mean


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

class ConstantVectorMean(Mean):
    def __init__(self, d=1, prior=None, batch_shape=torch.Size(), **kwargs):
        super(ConstantVectorMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constantvector", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, d)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constantvector")

    def forward(self, input):
        return self.constantvector[input]
        

class DriftScaleKernel(Kernel):
    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, T0=None, T1_prior=None, T1_constraint=None,
        T2_prior=None, T2_constraint=None,
        outputscale_prior=None, outputscale_constraint=None, **kwargs):
        # treatment takes place at T0
        # treatment starts to have effect at T0+T1
        # treatment effect becomes stable at T0+T1+T2
        # scaling function: a(t) = 0, t<T0+T1
        #                   a(t) = (t-T0-T1)/T2, T0+T1<t<T0+T1+T2
        #                   a(t) = 1,, t>T0+T1+T2
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(DriftScaleKernel, self).__init__(**kwargs)
        self.T0 = T0
        if outputscale_constraint is None:
            outputscale_constraint = Positive()
        if T1_constraint is None:
            T1_constraint = Positive()
        if T2_constraint is None:
            T2_constraint = Positive()
        

        self.base_kernel = base_kernel
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        T1 = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        T2 = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        self.register_parameter(name="raw_T1", parameter=torch.nn.Parameter(T1))
        self.register_parameter(name="raw_T2", parameter=torch.nn.Parameter(T2))
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior, lambda m: m.outputscale, lambda m, v: m._set_outputscale(v)
            )
        if T1_prior is not None:
            self.register_prior(
                "T1_prior", T1_prior, lambda m: m.T1, lambda m, v: m._set_T1(v)
            )
        if T2_prior is not None:
            self.register_prior(
                "T2_prior", T2_prior, lambda m: m.T2, lambda m, v: m._set_T2(v)
            )
        self.register_constraint("raw_outputscale", outputscale_constraint)
        self.register_constraint("raw_T1", T1_constraint)
        self.register_constraint("raw_T2", T2_constraint)

        if self.T0 is None:
            for name, param in self.named_parameters():
                param.requires_grad = False

    def scaling(self,x):
        x = torch.clamp(x, min=self.T0+(self.T1).item(), max=self.T0+(self.T1+self.T2).item())
        x = (x-self.T0-self.T1)/self.T2
        # y = 0, x<0
        # y = 0.5 - (0.5^2-x^2)^0.5, 0<x<0.5
        # y = 0.5 + (0.5^2-(1-x)^2)^0.5, 0.5<=x<1
        # y = 1, x>1
        return torch.exp(1.0-1.0/(x+1e-8))

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    @property
    def T1(self):
        return self.raw_T1_constraint.transform(self.raw_T1)

    @T1.setter
    def T1(self, value):
        self._set_T1(value)

    def _set_T1(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_T1)
        self.initialize(raw_T1=self.raw_T1_constraint.inverse_transform(value))

    @property
    def T2(self):
        return self.raw_T2_constraint.transform(self.raw_T2)

    @T2.setter
    def T2(self, value):
        self._set_T2(value)

    def _set_T2(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_T2)
        self.initialize(raw_T2=self.raw_T2_constraint.inverse_transform(value))

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        outputscales = self.outputscale
        if self.T0 is not None:
            a1 = self.scaling(x1)
            a2 = self.scaling(x2)
            a12 = a1.matmul(a2.T)
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return delazify(orig_output) * outputscales if self.T0 is None else \
                delazify(orig_output).mul(a12.diag()) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales) if self.T0 is None else \
                orig_output.mul(outputscales).mul(a12)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)

class DriftIndicatorKernel(Kernel):

    def __init__(self, num_tasks, **kwargs):
        super().__init__(**kwargs)

        self.num_tasks = num_tasks

    def covar_matrix(self):
        res = torch.zeros(self.num_tasks,self.num_tasks)
        if self.batch_shape:
            res = res.unsqueeze(0).repeat(self.batch_shape[0], 1, 1)
            for i in range(self.batch_shape[0]):
                res[i,1,1] = 1
        else:
            res[1,1] = 1
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

class DriftMean(Mean):
    def __init__(self, T0=None, T1_prior=None, T1_constraint=None,
        T2_prior=None, T2_constraint=None, effect_prior=None, effect_constraint=None):
        # treatment takes place at T0
        # treatment starts to have effect at T0+T1
        # treatment effect becomes stable at T0+T1+T2
        # scaling function: a(t) = 0, t<T0+T1
        #                   a(t) = (t-T0-T1)/T2, T0+T1<t<T0+T1+T2
        #                   a(t) = 1,, t>T0+T1+T2
        super(DriftMean, self).__init__()
        self.T0 = T0
        if effect_constraint is None:
            effect_constraint = Interval(-1,1)
        if T1_constraint is None:
            T1_constraint = Positive()
        if T2_constraint is None:
            T2_constraint = Positive()
        
        effect = torch.tensor(0.0)
        T1 = torch.tensor(0.0)
        T2 = torch.tensor(0.0)
        self.register_parameter(name="raw_effect", parameter=torch.nn.Parameter(effect))
        self.register_parameter(name="raw_T1", parameter=torch.nn.Parameter(T1))
        self.register_parameter(name="raw_T2", parameter=torch.nn.Parameter(T2))
        if effect_prior is not None:
            self.register_prior(
                "effect_prior", effect_prior, lambda m: m.effect, lambda m, v: m._set_effect(v)
            )
        if T1_prior is not None:
            self.register_prior(
                "T1_prior", T1_prior, lambda m: m.T1, lambda m, v: m._set_T1(v)
            )
        if T2_prior is not None:
            self.register_prior(
                "T2_prior", T2_prior, lambda m: m.T2, lambda m, v: m._set_T2(v)
            )
        self.register_constraint("raw_effect", effect_constraint)
        self.register_constraint("raw_T1", T1_constraint)
        self.register_constraint("raw_T2", T2_constraint)

        if self.T0 is None:
            for name, param in self.named_parameters():
                param.requires_grad = False

    @property
    def effect(self):
        return self.raw_effect_constraint.transform(self.raw_effect)

    @effect.setter
    def effect(self, value):
        self._set_effect(value)

    def _set_effect(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_effect)
        self.initialize(raw_effect=self.raw_effect_constraint.inverse_transform(value))

    @property
    def T1(self):
        return self.raw_T1_constraint.transform(self.raw_T1)

    @T1.setter
    def T1(self, value):
        self._set_T1(value)

    def _set_T1(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_T1)
        self.initialize(raw_T1=self.raw_T1_constraint.inverse_transform(value))

    @property
    def T2(self):
        return self.raw_T2_constraint.transform(self.raw_T2)

    @T2.setter
    def T2(self, value):
        self._set_T2(value)

    def _set_T2(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_T2)
        self.initialize(raw_T2=self.raw_T2_constraint.inverse_transform(value))

    def scaling(self,x):
        x = torch.clamp(x, min=self.T0+(self.T1).item(), max=self.T0+(self.T1+self.T2).item())
        x = (x-self.T0-self.T1)/self.T2
        return x

    def forward(self, x):
        return self.scaling(x)*self.effect
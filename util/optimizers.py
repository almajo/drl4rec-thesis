import torch


class MultiAdam:
    def __init__(self, dense_parameters=None, **kwargs):
        self.adam = None
        if dense_parameters:
            self.adam = torch.optim.Adam(dense_parameters, **kwargs)
        self.kwargs = kwargs
        self.sp_adam = None

    def add_dense_parameters(self, parameters):
        if not self.adam:
            self.adam = torch.optim.Adam(parameters, **self.kwargs)
        else:
            self.adam.add_param_group({"params": parameters})

    def add_sparse_parameters(self, parameters):
        if not self.sp_adam:
            kwargs = self.kwargs.copy()
            kwargs.pop("weight_decay", None)
            self.sp_adam = torch.optim.SparseAdam(parameters, **kwargs)
        else:
            self.sp_adam.add_param_group({"params": parameters})

    def zero_grad(self):
        self.adam.zero_grad()
        if self.sp_adam is not None:
            self.sp_adam.zero_grad()

    def step(self):
        self.adam.step()
        if self.sp_adam is not None:
            self.sp_adam.step()

    def state_dict(self):
        d = {
            "adam": self.adam.state_dict()
        }
        if self.sp_adam:
            d["sp_adam"] = self.sp_adam.state_dict()

        return d

    def load_state_dict(self, state_dict):
        self.adam.load_state_dict(state_dict["adam"])
        if self.sp_adam:
            self.sp_adam.load_state_dict(state_dict["sp_adam"])


class DummyStateOpt:
    def zero_grad(self):
        pass

    def step(self):
        pass

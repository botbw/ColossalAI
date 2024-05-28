# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.accelerator import get_accelerator

from ._utils import flatten, release_param_grad, sync_tensor
from .bookkeeping import BucketStore, GradientStore, ParameterStore


class LowLevelOptStrategyBase(ABC):
    """
    Base class for low-level optimization strategies, this is to reduce the
    coupling between different param group and their process group (parallel settings)

    This class contains only necessary stores/data for optimizer:
        1. params
        2. grads
        3. reduce buckets
    and necessary methods to manipulate them

    The child class should implement the logic for computation and communication
    given a specific process group
    """

    # the store before refactoring supports multiple param groups
    # but currently only one is used
    DEFAULT_STORE_GROUP_ID = 0

    def __init__(self, param_group, pg, partition_grad, cpu_offload, **kwargs):
        # param_group that current strategy is working on
        self.param_group = param_group
        self.pg = pg

        # stage 2 TODO @botbw: this should be unique across all strategies
        self._partion_grads = partition_grad

        self._cpu_offload = cpu_offload

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(pg)
        self._grad_store = GradientStore(pg, partition_grad=partition_grad)
        self._bucket_store = BucketStore(pg)

        # by default this shouldn't be manipulate
        self.require_grad_sync = True

    def zero_grad_store(self):
        self._grad_store.reset_all_gradients()

    @property
    def working_grads(self):
        return self._grad_store.get_working_grads_by_group_id(LowLevelOptStrategyBase.DEFAULT_STORE_GROUP_ID)

    @abstractmethod
    def pre_backward(self, loss, retain_graph=False):
        raise NotImplementedError

    @abstractmethod
    def post_backward(self):
        raise NotImplementedError

    @abstractmethod
    def pre_backward_by_grad(self, tensor, grad):
        raise NotImplementedError

    @abstractmethod
    def post_backward_by_grad(self):
        raise NotImplementedError

    @abstractmethod
    def pre_step(self):
        raise NotImplementedError

    @abstractmethod
    def post_step(self):
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        raise NotImplementedError


class LowLevelOptStrategy(LowLevelOptStrategyBase):
    def __init__(
        self,
        param_group: Dict[str, Any],  # from optimizer.param_groups
        reduce_bucket_size: int = 1024 * 1024,  # communication
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = False,  # TODO @botbw: redundant to partition_grad?
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        dp_process_group: Optional[ProcessGroup] = None,  # the dp pg for comm
        master_weights: bool = True,  # master weights
        **kwargs,
    ):
        super().__init__(
            param_group=param_group,
            pg=dp_process_group,
            cpu_offload=cpu_offload,
            partition_grad=partition_grad,
            **kwargs,
        )

        # if process_group is none, will use the default one
        self._local_rank = dist.get_rank(group=self.pg)
        self._world_size = dist.get_world_size(group=self.pg)

        # working and master params for mixed precision training
        group_params = []
        for param in param_group["params"]:
            if param.requires_grad:
                group_params.append(param)
        master_param_current_rank = self._create_master_param_current_rank(group_params)
        param_group["params"] = master_param_current_rank
        self._working_param_groups: List[torch.Tensor] = group_params
        self._master_param_groups_of_current_rank: List[torch.Tensor] = master_param_current_rank

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # gradient clipping
        # self._clip_grad_norm = clip_grad_norm

        # master weights copy
        self._master_weights = master_weights

        # initialize communication stream for
        # communication-computation overlapping
        if self._overlap_communication:
            self._comm_stream = get_accelerator().Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            # we iterate over the working params
            # on each param, we register a hook to its AccumulateGrad object
            param_group = self._working_param_groups
            for param in param_group:
                if param.requires_grad:

                    def _grad_handler(grad):
                        # if run with no_sync context, would not sync grad when backward
                        if self.require_grad_sync:
                            self._add_to_bucket(param, LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID)
                        return grad

                    param.register_hook(_grad_handler)

        # temporary variables
        self.__real_working_params = None

    def _create_master_param_current_rank(self, param_list):
        # split each param evenly by world size
        params_current_rank = []
        device = "cpu" if self._cpu_offload else get_accelerator().get_current_device()

        for param in param_list:
            padding_size = (self._world_size - param.numel() % self._world_size) % self._world_size
            self._param_store.record_param_padding_size(param, padding_size)

            with torch.no_grad():
                if padding_size > 0:
                    padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                    # reset working params' ptr when no master weights
                    if self._master_weights == False:
                        param.data = padding_param[: param.numel()].view(param.shape)
                else:
                    padding_param = param.data.view(-1)

                splited_params = padding_param.split(padding_param.numel() // self._world_size)
                splited_params = splited_params[self._local_rank]

                # use fp32 when master_weights is True
                if self._master_weights is True:
                    splited_param_current_rank = splited_params.detach().float().to(device)
                else:
                    splited_param_current_rank = splited_params

                params_current_rank.append(splited_param_current_rank)
                self._param_store.link_master_and_working_param(splited_param_current_rank, param)

        return params_current_rank

    ######################################################################
    # pre-backward: sanity check
    # post-backward: deal with grads

    def pre_backward(self, loss, retain_graph=False):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and no_sync are not compatible"

    def post_backward(self):
        if not self.require_grad_sync:
            return

        self._reduce_grad()

        # clear reduced grads
        if self._overlap_communication:
            get_accelerator().synchronize()

        self.zero_grad()

    def pre_backward_by_grad(self, tensor, grad):
        assert not (
            self._partition_grads and not self.require_grad_sync
        ), "ZeRO2(partition_grads) and no_sync are not compatible"

    def post_backward_by_grad(self):
        self.post_backward()

    def _reduce_grad(self):
        # if not overlapping communication (no reduction hook is attached) when zero1
        # we need to manually reduce these gradients
        if not self._partition_grads and not self._overlap_communication:
            self._sync_grad()
        else:
            self._run_reduction()

    def _sync_grad(self):
        param_group = self._working_param_groups
        for param in param_group:
            if param.requires_grad and param.grad is not None:
                self._add_to_bucket(param, LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID)

        self._run_reduction()

    def _run_reduction(self):
        if self._bucket_store.num_elements_in_bucket() <= 0:
            return

        self._bucket_store.build_grad_in_bucket()

        flat_grads = self._bucket_store.get_flatten_grad()
        flat_grads /= self._world_size

        # ready to add other tensors to bucket
        self._bucket_store.reset_num_elements_in_bucket()

        if self._overlap_communication:
            stream = self._comm_stream
            # in case of the memory being reused in the default stream
            flat_grads.record_stream(stream)
            # waiting for ops in the default stream finishing
            stream.wait_stream(get_accelerator().current_stream())
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            group_id = self._bucket_store.current_group_id
            assert group_id == LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID, "after refactoring, group_id should be 0"

            grad_dtype = flat_grads.dtype
            if self._communication_dtype is not None:
                flat_grads = flat_grads.to(self._communication_dtype)

            if not self._partition_grads:
                dist.all_reduce(flat_grads, group=self.pg)
                if flat_grads.dtype != grad_dtype:
                    flat_grads = flat_grads.to(grad_dtype)

                flat_grads_per_rank = flat_grads.split(flat_grads.numel() // self._world_size)
                grad_in_bucket = self._bucket_store.get_grad()
                self._update_unpartitoned_grad(grad_in_bucket.values(), flat_grads_per_rank, group_id)
            else:
                flat_grads_list = list(flat_grads.split(len(flat_grads) // self._world_size))
                recieved_grad = torch.zeros_like(flat_grads_list[0])
                dist.reduce_scatter(recieved_grad, flat_grads_list, group=self.pg)

                if recieved_grad.dtype != grad_dtype:
                    recieved_grad = recieved_grad.to(grad_dtype)

                grad_in_bucket_current_rank = self._bucket_store.get_grad()[self._local_rank]
                self._update_partitoned_grad(grad_in_bucket_current_rank, recieved_grad, group_id, 1)

        self._bucket_store.reset()

    def _add_to_bucket(self, param, group_id):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # or got a grad of param from another group
        # after reduction, the bucket will be empty
        if (
            self._bucket_store.num_elements_in_bucket() + param_size > self._reduce_bucket_size
            or group_id != self._bucket_store.current_group_id
        ):
            self._run_reduction()

        padding_size = self._param_store.get_param_padding_size(param)
        self._bucket_store.add_param_grad(group_id, param, padding_size)

    def _update_partitoned_grad(
        self, origin_grad_list: List, flat_grad: torch.Tensor, group_id: int, partition_num: int
    ) -> None:
        sync_tensor(flat_grad, origin_grad_list)
        for grad in origin_grad_list:
            param_id = self._bucket_store.get_param_id_of_grad(grad)
            self._add_grad(grad, partition_num, group_id, param_id)

    def _add_grad(self, grad: torch.Tensor, partition_num: int, group_id: int, param_id: int, rank: int = 0) -> None:
        if len(self._grad_store.get_partitioned_gradients_by_param_id(group_id, param_id)) < partition_num:
            self._grad_store.append_gradients_by_param_id(grad, group_id, param_id)
        else:
            self._grad_store.add_gradients_by_param_id(grad, rank, group_id, param_id)

    ######################################################################

    def zero_grad(self, set_to_none=True):
        param_group = self._working_param_groups
        for param in param_group:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    param.grad.detach()
                    param.grad.zero_()

    def pre_step(self):
        # record all grads for unscale and clip
        grad_partition_groups = []
        norm_groups = []

        # sometimes not all params are 'really' working
        # for instance, when layer drop, the dropped layer has no grad
        # and should not be updated
        grad_index = 0 if self._partition_grads else self._local_rank
        master_params = self._master_param_groups_of_current_rank
        real_working_params = []
        real_master_params = []
        for splited_param in master_params:
            working_param = self._param_store.master_to_working_param[id(splited_param)]
            # if a working param requires grad and has no grad
            # it is not 'really' working, e.g. the droped layer
            # else the splited grad should be attached to the splited param
            grads = self._grad_store.get_partitioned_gradients_by_param_id(
                LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID, id(working_param)
            )
            if len(grads) > 0:
                real_working_params.append(working_param)
                grad = grads[grad_index]
                # no need to copy fp32 grad if master_weights is False
                if self._master_weights:
                    grad = grad.to(splited_param.dtype).to(splited_param.device)
                splited_param.grad = grad
                grad_partition_groups.append(grad)
                real_master_params.append(splited_param)

        # compute norm
        working_grads = self._grad_store.get_working_grads_by_group_id(LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID)
        norm_group = self._compute_grad_norm(gradients=working_grads)
        norm_groups.append(norm_group)

        self._grad_store.reset_grads_by_group_id(LowLevelOptStrategy.DEFAULT_STORE_GROUP_ID)

        # update the params in the optimizer
        self.param_group["params"] = real_master_params

        assert self.__real_working_params is None
        self.__real_working_params = real_working_params

    def post_step(self):
        real_working_params = self.__real_working_params
        self.__real_working_params = None

        release_param_grad(self._master_param_groups_of_current_rank)

        # update working partition updated by the current rank
        device = get_accelerator().get_current_device()
        master_working_param = self.param_group["params"]
        for idx, splited_param in enumerate(master_working_param):
            working_param = real_working_params[idx]
            all_splited_param = [
                torch.zeros(splited_param.shape, device=device, dtype=self._dtype) for _ in range(self._world_size)
            ]
            dist.all_gather(all_splited_param, splited_param.to(device).to(self._dtype), group=self.pg)
            working_param.data.copy_(flatten(all_splited_param)[: working_param.numel()].reshape_as(working_param))
        self.param_group["params"] = self._master_param_groups_of_current_rank


# class MoeZeroStrategy(LowLevelOptStrategy):
#     def

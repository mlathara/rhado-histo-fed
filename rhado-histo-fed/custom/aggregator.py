import numpy as np

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.aggregators.accumulate_model_aggregator import AccumulateWeightedAggregator, _AccuItem

class NestedAccumulatedWeightedAggregator(AccumulateWeightedAggregator):

    def __init__(self, exclude_vars=None, aggregation_weights=None, expected_data_kind="WEIGHT_DIFF"):
        super().__init__(exclude_vars, aggregation_weights, expected_data_kind)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Aggregate model variables.

        This function is not thread-safe.


        Args:
            fl_ctx (FLContext): System-wide FL Context

        Returns:
            Shareable: Return True to indicates the current model is the best model so far.
        """
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, "aggregating {} updates at round {}".format(len(self.accumulator), current_round))

        # TODO: What if AppConstants.GLOBAL_MODEL is None?
        acc_vars = [set(acc.data.keys()) for acc in self.accumulator]
        acc_vars = set.union(*acc_vars) if acc_vars else acc_vars
        # update vars that are not in exclude pattern
        vars_to_aggregate = (
            [g_var for g_var in acc_vars if not self.exclude_vars.search(g_var)] if self.exclude_vars else acc_vars
        )

        clients_with_messages = []
        aggregated_model = {}
        for v_name in vars_to_aggregate:
            n_local_iters, np_vars = [], []
            for item in self.accumulator:
                assert isinstance(item, _AccuItem)
                client_name = item.client
                data = item.data
                n_iter = item.steps

                if n_iter is None:
                    if self.warning_count.get(client_name, 0) <= self.warning_limit:
                        self.log_warning(
                            fl_ctx,
                            f"NUM_STEPS_CURRENT_ROUND missing in meta of shareable"
                            f" from {client_name} and set to default value, 1.0. "
                            f" This kind of message will show {self.warning_limit} times at most.",
                        )
                        if client_name in self.warning_count:
                            self.warning_count[client_name] = self.warning_count[client_name] + 1
                        else:
                            self.warning_count[client_name] = 0
                    n_iter = 1.0
                if v_name not in data.keys():
                    continue  # this acc doesn't have the variable from client
                float_n_iter = float(n_iter)
                n_local_iters.append(float_n_iter)
                aggregation_weight = self.aggregation_weights.get(client_name)
                if aggregation_weight is None:
                    if self.warning_count.get(client_name, 0) <= self.warning_limit:
                        self.log_warning(
                            fl_ctx,
                            f"Aggregation_weight missing for {client_name} and set to default value, 1.0"
                            f" This kind of message will show {self.warning_limit} times at most.",
                        )
                        if client_name in self.warning_count:
                            self.warning_count[client_name] = self.warning_count[client_name] + 1
                        else:
                            self.warning_count[client_name] = 0
                    aggregation_weight = 1.0

                if isinstance(data[v_name], list):
                    weighted_value_np = np.array(data[v_name]) * float_n_iter * aggregation_weight
                    weighted_value = weighted_value_np.tolist()
                else:
                    weighted_value = data[v_name] * float_n_iter * aggregation_weight
                if client_name not in clients_with_messages:
                    if client_name in self.aggregation_weights.keys():
                        self.log_debug(fl_ctx, f"Client {client_name} use weight {aggregation_weight} for aggregation.")
                    else:
                        self.log_debug(
                            fl_ctx,
                            f"Client {client_name} not defined in the aggregation weight list. Use default value 1.0",
                        )
                    clients_with_messages.append(client_name)
                np_vars.append(weighted_value)
            if not n_local_iters:
                continue  # all acc didn't receive the variable from clients
            new_val = np.sum(np_vars, axis=0) / np.sum(n_local_iters)
            aggregated_model[v_name] = new_val

        self.accumulator.clear()

        self.log_debug(fl_ctx, f"Model after aggregation: {aggregated_model}")

        dxo = DXO(data_kind=self.expected_data_kind, data=aggregated_model)
        return dxo.to_shareable()
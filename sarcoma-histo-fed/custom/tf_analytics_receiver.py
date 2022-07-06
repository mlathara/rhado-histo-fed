# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mostly using nvflare code with minor changes...
# would've tried to inherit but there's a pytorch import that I don't
# want to include here

import os
from typing import List, Optional

import tensorflow as tf
from nvflare.apis.analytix import AnalyticsData
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class TFAnalyticsReceiver(AnalyticsReceiver):
    def __init__(self, tb_folder="tb_events", events: Optional[List[str]] = None):
        super().__init__(events=events)
        self.writers_table = {}
        self.tb_folder = tb_folder
        self.root_log_dir = None

    def initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        # TODO update to fl_ctx.get_job_id() once nvflare is updated
        run_dir = workspace.get_run_dir(fl_ctx.get_run_number())
        root_log_dir = os.path.join(run_dir, self.tb_folder)
        os.makedirs(root_log_dir, exist_ok=True)
        self.root_log_dir = root_log_dir

    def finalize(self, fl_ctx: FLContext):
        for writer in self.writers_table.values():
            writer.flush()
            writer.close()

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin):
        dxo = from_shareable(shareable)
        analytic_data = AnalyticsData.from_dxo(dxo)

        writer = self.writers_table.get(record_origin)
        if writer is None:
            peer_log_dir = os.path.join(self.root_log_dir, record_origin)
            writer = tf.summary.create_file_writer(peer_log_dir)
            self.writers_table[record_origin] = writer

        with writer.as_default():
            for k, v in dxo.data.items():
                self.log_debug(
                    fl_ctx, f"save tag {k} and value {v} from {record_origin}", fire_event=False
                )
                if (
                    not isinstance(analytic_data.kwargs, dict)
                    or "global_step" not in analytic_data.kwargs
                ):
                    self.log_error(
                        fl_ctx,
                        "Federated event should contain global_step. Using 0. Found type: %s, contents: %s key: %s"
                        % (type(analytic_data.kwargs), str(analytic_data.kwargs), k),
                    )
                    # nvflare analytics_sender has a small bug that doesn't send global_step if it is zero
                    # so we'll assume zero here if we don't find global_step
                    tf.summary.scalar(k, v, step=0)
                else:
                    tf.summary.scalar(k, v, step=analytic_data.kwargs["global_step"])

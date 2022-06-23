{
  "format_version": 2,
  "server": {
    "heart_beat_timeout": 600
  },
  "task_data_filters": [],
  "task_result_filters": [],
  "components": [
    {
      "id": "persistor",
      "path": "tf2_model_persistor.TF2ModelPersistor",
      "args": {
        "save_name": "tf2weights.pickle",
        "classes": 3,
        "flipmode": "horizontal_and_vertical",
        "tile_size": 299
      }
    },
    {
      "id": "shareable_generator",
      "path": "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator",
      "args": {}
    },
    {
      "id": "aggregator",
      "path": "aggregator.NestedAccumulatedWeightedAggregator",
      "args": {
        "expected_data_kind": "WEIGHTS"
      }
    },
    {
      "id": "tb_analytics_receiver",
      "path": "nvflare.app_common.pt.tb_receiver.TBAnalyticsReceiver",
      "name": "TBAnalyticsReceiver",
      "args": {"events": ["fed.analytix_log_stats"]}
    }
  ],
  "workflows": [
    {
      "id": "scatter_gather_ctl",
      "path": "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather",
      "args": {
        "min_clients": 1,
        "num_rounds": 3,
        "start_round": 0,
        "wait_time_after_min_received": 10,
        "aggregator_id": "aggregator",
        "persistor_id": "persistor",
        "shareable_generator_id": "shareable_generator",
        "train_task_name": "train",
        "train_timeout": 0
      }
    }
  ]
}
{
  "format_version": 2,
  "executors": [
    {
      "tasks": [
        "train"
      ],
      "executor": {
        "path": "trainer.SimpleTrainer",
        "args": {
          "dataset_path_env_var": "environment-variable-for-parent-dir-of-dataset",
          "epochs_per_round": 2,
          "slideextension": "svs",
          "overlap": 0,
          "workers": 32,
          "augment_tiles": true,
          "output_folder": "name-of-subdir-within-dataset-parent-dir-to-store-generated-tiles",
          "quality": 90,
          "tile_size": 299,
          "background": 25,
          "magnification": 5,
          "labels_file": "name-of-labels-file-within-dataset-parent-dir",
          "labels_map": dict-of-label-to-integer-index,
          "validation_split": 0.2,
          "flipmode": "horizontal_and_vertical",
          "num_epoch_per_auc_calc": 0,
          "tensorboard": dict-of-tensorboard-kwargs,
          "baseimage": "environment-variable-with-path-to-baseimage",
          "analytic_sender_id": "analytic_sender"
        }
      }
    }
  ],
  "components": [
    {
      "id": "analytic_sender",
      "path": "nvflare.app_common.widgets.streaming.AnalyticsSender",
      "name": "AnalyticsSender",
      "args": {"event_type": "analytix_log_stats"}
    },
    {
      "id": "event_to_fed",
      "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
      "name": "ConvertToFedEvent",
      "args": {"events_to_convert": ["analytix_log_stats"], "fed_event_prefix": "fed."}
    }
  ],
  "task_result_filters": [],
  "task_data_filters": []
}

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
          "epochs_per_round": 2,
          "slidepath": "/path/to/slides/*svs",
          "overlap": 0,
          "workers": 32,
          "output_base": "/path/to/output/tiles",
          "quality": 90,
          "tile_size": 299,
          "background": 25,
          "magnification": 5,
          "labels_file": "/path/to/labels",
          "validation_split": 0.2,
          "flipmode": "horizontal_and_vertical",
          "num_epoch_per_auc_calc": 0
        }
      }
    }
  ],
  "task_result_filters": [],
  "task_data_filters": []
}

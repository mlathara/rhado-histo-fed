import os
import shutil
from tempfile import mkdtemp

import tensorflow as tf
from network import build_model
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from preprocess import slides_to_tiles
from slide_aucroc import SlideROCCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_image(image, height=299, width=299):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width])
    img = tf.expand_dims(img, axis=0)

    return tf.keras.applications.inception_v3.preprocess_input(img)


def get_dataset(files, batch_size, num_classes):
    paths = []
    labels = []
    filenames = []
    for label, slides in files.items():
        for slide, tiles_list in slides.items():
            paths.extend([t for t in tiles_list])
            # either we make arrays out of the label elements
            # or we have to tf.reshape the one_hot vectors later
            labels.extend([[label]] * len(tiles_list))
            filenames.extend([slide] * len(tiles_list))

    # convert filenames to integer ids for later reduction
    tempdict = {}
    for f in filenames:
        tempdict[f] = len(tempdict)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    files = tf.data.Dataset.from_tensor_slices([tempdict[f] for f in filenames])
    pixels = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(
        lambda x: tf.one_hot(x, num_classes), num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = tf.data.Dataset.zip((files, pixels, label_ds))
    ds = ds.prefetch(buffer_size=batch_size)
    return ds


class SimpleTrainer(Executor):
    def __init__(
        self,
        epochs_per_round: int,
        dataset_path_env_var: str,
        slideextension: str,
        overlap: int,
        workers: int,
        augment_tiles: bool,
        output_folder: str,
        quality: int,
        tile_size: int,
        background: float,
        magnification: float,
        labels_file: str,
        labels_map: dict,
        validation_split: float,
        flipmode: str,
        num_epoch_per_auc_calc: int,
        tensorboard: str,
        baseimage: str,
        analytic_sender_id: str,
    ):
        super().__init__()
        dataset_dir = os.getenv(dataset_path_env_var)
        if not dataset_dir:
            raise RuntimeError(
                dataset_path_env_var
                + " environment variable was not set. "
                + "Please set it to the path where dataset can be found"
            )
        self.epochs_per_round = epochs_per_round
        self.train_ds = None
        self.validation_ds = None
        self.model = None
        self.slidepath = os.path.join(dataset_dir, "*" + slideextension)
        self.overlap = overlap
        self.workers = workers
        self.augment_tiles = augment_tiles
        self.output_base = os.path.join(dataset_dir, output_folder)
        self.quality = quality
        self.tile_size = tile_size
        self.background = background
        self.magnification = magnification
        self.labels_file = os.path.join(dataset_dir, labels_file)
        self.validation_split = validation_split
        self.num_epoch_per_auc_calc = num_epoch_per_auc_calc
        self.tensorboard = tensorboard
        if self.tensorboard:
            if "log_dir" not in self.tensorboard:
                self.tensorboard["log_dir"] = mkdtemp()

        self.baseimage = os.getenv(baseimage)
        self.analytic_sender_id = analytic_sender_id
        self.labels_map = labels_map
        self.current_round = None
        if flipmode not in ["horizontal", "vertical", "horizontal_and_vertical"]:
            self.flipmode = None
        else:
            self.flipmode = flipmode

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)
        elif event_type == EventType.AFTER_TASK_EXECUTION:
            self.send_federated_events(fl_ctx)

    def add_to_writer(self, fl_ctx, writer, tag, get_tag_func, prefix):
        step_offset = self.current_round * self.epochs_per_round
        for element in tag:
            for _, step, tensor in get_tag_func(element):
                array = tf.make_ndarray(tensor)
                if array.size != 1:
                    raise RuntimeError(
                        "Only scalars are supported as metrics for fed events\n"
                        + "Metric: %s, shape: %s" % (element, array.shape)
                    )
                elif not isinstance(array.item(0), float):
                    # graphs, etc are not supported
                    self.log_warning(
                        fl_ctx,
                        "Metric %s had type %s. Expected float"
                        % (element, str(type(array.item(0)))),
                    )
                else:
                    writer.add_scalar(
                        prefix + element,
                        array.item(0),
                        global_step=step_offset + step,
                    )

    def send_federated_events(self, fl_ctx: FLContext):
        self.log_debug(fl_ctx, "Sending fed events")
        engine = fl_ctx.get_engine()
        writer = engine.get_component(self.analytic_sender_id)
        for subdir, dirs, _files in os.walk(self.tensorboard["log_dir"]):
            if not dirs:
                event_acc = EventAccumulator(subdir)
                event_acc.Reload()
                tags = event_acc.Tags()
                if subdir.strip("/").endswith("validation"):
                    prefix = "validation_"
                elif subdir.strip("/").endswith("train"):
                    prefix = "train_"
                else:
                    prefix = ""
                self.add_to_writer(fl_ctx, writer, tags["scalars"], event_acc.Scalars, prefix)
                self.add_to_writer(fl_ctx, writer, tags["tensors"], event_acc.Tensors, prefix)

        shutil.rmtree(self.tensorboard["log_dir"])

    def setup(self, fl_ctx: FLContext):
        train_files, validation_files = slides_to_tiles(
            self.slidepath,
            self.overlap,
            self.workers,
            self.augment_tiles,
            self.output_base,
            self.quality,
            self.tile_size,
            self.background,
            self.magnification,
            self.labels_file,
            self.labels_map,
            self.validation_split,
            self.baseimage,
        )
        num_classes = len(self.labels_map)

        self.train_ds = get_dataset(train_files, 32, num_classes)
        self.validation_ds = get_dataset(validation_files, 32, num_classes)
        self.model = build_model((self.tile_size, self.tile_size))

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        self.current_round = int(shareable.get_header(AppConstants.CURRENT_ROUND))
        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "train":
            return make_reply(ReturnCode.TASK_UNKNOWN)

        dxo = from_shareable(shareable)
        model_weights = dxo.data

        # update local model weights with received weights
        for layer_idx in range(4):
            layer = self.model.get_layer(index=layer_idx)
            layer.set_weights(model_weights[layer.name])

        # adjust LR or other training time info as needed
        # such as callback in the fit function
        train = self.train_ds.map(lambda file, pixels, label: (pixels, label))
        valid = self.validation_ds.map(lambda file, pixels, label: (pixels, label))

        callbacks = []
        if self.tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(**self.tensorboard))
        if self.num_epoch_per_auc_calc:
            callbacks.append(
                SlideROCCallback(
                    self.train_ds,
                    self.validation_ds,
                    self.num_epoch_per_auc_calc,
                    self.tensorboard["log_dir"],
                )
            )

        self.model.fit(
            train, epochs=self.epochs_per_round, validation_data=valid, callbacks=callbacks
        )

        # report updated weights in shareable
        weights = {
            self.model.get_layer(index=layer_idx)
            .name: self.model.get_layer(index=layer_idx)
            .get_weights()
            for layer_idx in range(4)
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        return new_shareable

import os
import shutil
from tempfile import TemporaryDirectory

import tensorflow as tf
from network import build_model
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
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
            labels.extend([[label]]*len(tiles_list))
            filenames.extend([slide]*len(tiles_list))

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
        self.baseimage = os.getenv(baseimage)
        self.analytic_sender_id = analytic_sender_id
        self.labels_map = labels_map
        if flipmode not in ["horizontal", "vertical", "horizontal_and_vertical"]:
            self.flipmode = None
        else:
            self.flipmode = flipmode

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)
        elif event_type == EventType.AFTER_TASK_EXECUTION:
            self.send_federated_events(fl_ctx)

    def send_federated_events(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        writer = engine.get_component(self.analytic_sender_id)
        event_acc = EventAccumulator(self.tensorboard["log_dir"])
        event_acc.Reload()
        tags = event_acc.Tags()
        # for now, we're only sending scalars
        for scalar in tags["scalars"]:
            print("Sending scalar tag: " + str(scalar) + " len: " + len(event_acc.Scalars(scalar)))
            for _, step, tensor in event_acc.Scalars(scalar):
                array = tf.make_ndarray(tensor)
                # TODO not sure there is any way to make step global in the client
                writer.add_scalar(scalar, array.item(0))

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

        run = fl_ctx.get_run_number()
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
        tensorboard_dir = None
        if self.tensorboard:
            if "log_dir" in self.tensorboard:
                tensorboard_dir = self.tensorboard["log_dir"]
            else:
                tensorboard_dir = TemporaryDirectory().name
                self.tensorboard["log_dir"] = tensorboard_dir
            callbacks.append(tf.keras.callbacks.TensorBoard(**self.tensorboard))
        if self.num_epoch_per_auc_calc:
            callbacks.append(
                SlideROCCallback(
                    self.train_ds,
                    self.validation_ds,
                    self.num_epoch_per_auc_calc,
                    tensorboard_dir,
                    run,
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

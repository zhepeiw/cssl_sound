import ruamel.yaml
import numpy as np
import speechbrain as sb
from speechbrain.utils.train_logger import TrainLogger
import pdb

class MyWandBLogger(TrainLogger):
    """Logger for wandb. To be used the same way as TrainLogger. Handles nested dicts as well.
    An example on how to use this can be found in recipes/Voicebank/MTL/CoopNet/"""

    def __init__(self, *args, **kwargs):
        try:
            yaml_file = kwargs.pop("yaml_config")
            with open(yaml_file, "r") as yaml_stream:
                # Read yaml with ruamel to ignore bangs
                config_dict = ruamel.yaml.YAML().load(yaml_stream)
            self.run = kwargs.pop("initializer", None)(
                *args, **kwargs, config=config_dict
            )
        except Exception as e:
            raise e("There was an issue with the WandB Logger initialization")

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""

        logs = {}
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            logs[dataset] = stats

        step = stats_meta.get("datapoints_seen", None)
        if step is None:
            step = stats_meta.get("epoch", None)
        if step is not None:  # Useful for continuing runs that crashed
            self.run.log({**logs, **stats_meta}, step=step)
        else:
            self.run.log({**logs, **stats_meta})


@sb.utils.checkpoints.register_checkpoint_hooks
class DatapointCounter:
    """
        A counter which can save the number of datapoints the model has seen
    """
    def __init__(self):
        self.current = 0

    def update(self, batch_size):
        self.current += batch_size

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch=True, device=None):
        # NOTE: end_of_epoch = True by default so that when
        #  loaded in parameter transfer, this starts a new epoch.
        #  However, parameter transfer to EpochCounter should
        #  probably never be used really.
        del device  # Not used.
        del end_of_epoch
        with open(path) as fi:
            self.current = int(fi.read())


def prepare_task_classes(num_classes, num_tasks, seed=1234):
    '''
        returns a list of tuples for class-incremental seup
    '''
    arr = np.arange(num_classes)
    rng = np.random.default_rng(seed)
    if num_tasks > 1:
        rng.shuffle(arr)
    task_classes = np.array_split(arr, num_tasks)
    task_classes = [tuple(e) for e in task_classes]
    return task_classes

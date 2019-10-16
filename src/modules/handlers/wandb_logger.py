import wandb
import torch.nn as nn

from ignite.contrib.handlers.base_logger import (BaseLogger,
                                                 BaseOptimizerParamsHandler,
                                                 BaseOutputHandler)


class OutputHandler(BaseOutputHandler):
    def __init__(self,
                 tag,
                 metric_names=None,
                 output_transform=None,
                 another_engine=None):
        super().__init__(tag, metric_names, output_transform, another_engine)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, WandbLogger):
            raise RuntimeError(
                "Handler 'OutputHandler' works only with WandbLogger")

        metrics = self._setup_output_metrics(engine)
        wandb.log(metrics)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    def __init__(self, optimizer, param_name="lr", tag=None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name,
                                                     tag)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, WandbLogger):
            raise RuntimeError(
                "Handler 'OptimizerParamsHandler' works only with WandbLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i):
            float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            logger.writer.add_scalar(k, v, global_step)


class WandbLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        wandb.init(*args, **kwargs)

    def watch(self, model: nn.Module):
        wandb.watch(model)

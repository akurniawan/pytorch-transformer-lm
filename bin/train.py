# import wandb
import torch
import torch.optim as optim
import torch.nn as nn

import hydra
import numpy as np
import random

from functools import reduce
# from src.modules.data import wikitext103
from src.modules.data import Dataset
from src.models.encoder import TransformerEncoder
from src.modules.embedding import TransformerEmbedding
from src.models.lm import TransformerLanguageModel
from src.models.criterion import lm_criterion
from src.modules.handlers.wandb_logger import OutputHandler, WandbLogger

from ignite.engine import Events
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
# from ignite.contrib.handlers import TensorboardLogger
# from ignite.contrib.handlers.tensorboard_logger import (
#     OutputHandler, OptimizerParamsHandler, WeightsHistHandler,
#     GradsHistHandler, WeightsScalarHandler, GradsScalarHandler)
from ignite.contrib.handlers.param_scheduler import (
    create_lr_scheduler_with_warmup, CosineAnnealingScheduler)

# wandb.init("language_model")

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


@hydra.main("../config/config.yaml")
def train(cfg):
    print(cfg.pretty())

    ###################################################################
    # Dataset
    ###################################################################
    wt = Dataset(batch_size=cfg.train.batch_size,
                 bptt_len=cfg.train.bptt_len,
                 dataset_cls=hydra.utils.get_class(cfg.dataset.name))

    ###################################################################
    # Models
    ###################################################################
    base_embedding = hydra.utils.instantiate(cfg.embedding,
                                             ntokens=len(wt.text_field.vocab) +
                                             3)
    embedding = TransformerEmbedding(
        embedding=base_embedding,
        max_length=cfg.train.bptt_len,
        embedding_size=base_embedding.embedding_size,
        use_positional_embedding=False)
    encoder = TransformerEncoder(query_dim=cfg.encoder.query_dim,
                                 att_num_units=cfg.encoder.att_num_units,
                                 ffn_num_unit=cfg.encoder.ffn_num_unit,
                                 max_ext=cfg.encoder.max_ext)
    model = TransformerLanguageModel(embedding, encoder)
    model.init_weight()

    # wandb.watch(model)

    ###################################################################
    # Loss
    ###################################################################
    criterion = lm_criterion(in_features=cfg.encoder.att_num_units[-1],
                             vocab_size=len(wt.text_field.vocab))

    ###################################################################
    # Parameters + Train ops
    ###################################################################
    parameters = (list(model.parameters()) + list(criterion.parameters()))
    tot_params = 0
    for p in parameters:
        tot_params += reduce(lambda x, y: x * y, p.size())
    print("Total Parameters: ", tot_params)
    opt = optim.Adam(parameters, lr=cfg.train.lr)
    model.to(DEVICE)
    criterion.to(DEVICE)

    ###################################################################
    # Train + Evaluation
    ###################################################################
    def train_step(engine, batch):
        model.train()
        opt.zero_grad()

        text = batch.text.to(DEVICE).t().contiguous()
        target = batch.target.to(DEVICE).t().contiguous()

        out, out_past = model(text, engine.state.train_past)
        engine.state.train_past = out_past
        raw_loss = criterion(out.view(-1, out.size(2)), target.view(-1))
        loss = raw_loss[1]

        loss.backward()
        nn.utils.clip_grad_norm_(parameters, cfg.train.clip_grad)
        opt.step()

        return {"train_loss": loss.item(), "train_ppl": loss.exp().item()}

    def eval_step(engine, batch):
        model.eval()

        if not hasattr(engine.state, "eval_past"):
            engine.state.eval_past = None

        target_sample = []
        result_sample = []
        with torch.no_grad():
            text = batch.text.to(DEVICE).t().contiguous()
            target = batch.target.to(DEVICE).t().contiguous()

            out, out_past = model(text, engine.state.eval_past)

            vocab = wt.text_field.vocab
            idx = list(range(32))
            sample = random.choices(idx, k=5)
            for id_sample in sample:
                s = []
                for target_id in target[id_sample]:
                    s.append(vocab.itos[target_id])
                target_sample.append(" ".join(s))

                s = []
                for result_id in out.max(-1)[1][id_sample]:
                    s.append(vocab.itos[result_id])
                result_sample.append(" ".join(s))
            # engine.state.eval_past = out_past
            raw_loss = criterion(out.view(-1, out.size(2)), target.view(-1))
            loss = raw_loss[1]

            return {
                "val_loss": loss.item(),
                "sample": (target_sample, result_sample)
            }

    train_engine = Engine(train_step)
    eval_engine = Engine(eval_step)

    def reset_state(engine):
        engine.state.train_past = None

    def run_eval(_):
        print("start running eval")
        eval_engine.run(wt.valid_iter)
        metrics = eval_engine.state.metrics
        print("Validation loss: ", metrics["val_loss"], ", ppl: ",
              np.exp(metrics["val_loss"]))

    train_engine.add_event_handler(Events.EPOCH_STARTED, reset_state)
    train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_eval)

    ###################################################################
    # LR Scheduler
    ###################################################################
    cosine_scheduler = CosineAnnealingScheduler(opt.param_groups[0],
                                                "lr",
                                                0.0,
                                                2.5e-4,
                                                cycle_size=len(wt.train_iter))
    warmup_scheduler = create_lr_scheduler_with_warmup(cosine_scheduler, 0.0,
                                                       2.5e-4, 200)
    train_engine.add_event_handler(Events.ITERATION_STARTED, warmup_scheduler)

    ###################################################################
    # Metrics
    ###################################################################
    RunningAverage(output_transform=lambda x: x["train_ppl"]).attach(
        train_engine, "train_ppl")
    RunningAverage(output_transform=lambda x: x["train_loss"]).attach(
        train_engine, "train_loss")
    RunningAverage(output_transform=lambda x: x["val_loss"]).attach(
        eval_engine, "val_loss")
    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(train_engine, ["train_ppl", "train_loss"])
    progress_bar_val = ProgressBar(persist=True)
    progress_bar_val.attach(eval_engine, ["val_loss"])

    ###################################################################
    # Tensorboard
    ###################################################################
    # tb_logger = TensorboardLogger(log_dir=log_dir)
    tb_logger = WandbLogger(project="language_model", entity="akurniawan")
    tb_logger.watch(model)

    def stepn_logger(num_steps, handler):
        def logger_runner(engine, log_handler, event_name):
            if engine.state.iteration % num_steps == 0:
                handler(engine, log_handler, event_name)

        return logger_runner

    tb_logger.attach(train_engine,
                     log_handler=stepn_logger(
                         cfg.train.log_steps,
                         OutputHandler(tag="training",
                                       output_transform=lambda loss: loss)),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(eval_engine,
                     log_handler=OutputHandler(
                         tag="validation",
                         output_transform=lambda loss: loss,
                         another_engine=train_engine),
                     event_name=Events.EPOCH_COMPLETED)
    # tb_logger.attach(train_engine,
    #                  log_handler=stepn_logger(log_steps,
    #                                           OptimizerParamsHandler(opt)),
    #                  event_name=Events.ITERATION_STARTED)
    # tb_logger.attach(train_engine,
    #                  log_handler=stepn_logger(log_steps,
    #                                           WeightsScalarHandler(model)),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(train_engine,
    #                  log_handler=stepn_logger(log_steps,
    #                                           GradsScalarHandler(model)),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(train_engine,
    #                  log_handler=stepn_logger(500, WeightsHistHandler(model)),
    #                  event_name=Events.ITERATION_COMPLETED)
    # tb_logger.attach(train_engine,
    #                  log_handler=stepn_logger(500, GradsHistHandler(model)),
    #                  event_name=Events.ITERATION_COMPLETED)

    try:
        train_engine.run(wt.train_iter, max_epochs=cfg.train.epochs)
    except Exception:
        pass
    finally:
        tb_logger.close()


if __name__ == '__main__':
    train()

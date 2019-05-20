import torch
import torch.optim as optim
import torch.nn as nn

from functools import reduce
from src.modules.data import wikitext103
from src.models.encoder import TransformerEncoder
from src.modules.embedding import TransformerEmbedding, DropEmbedding
from src.models.lm import TransformerLanguageModel

from ignite.engine import Events
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler, OptimizerParamsHandler, WeightsHistHandler,
    GradsHistHandler)
from src.criterion import lm_criterion

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def train(epochs=500, batch_size=32, bptt_len=70, lr=0.00025, log_dir=None):
    ###################################################################
    # Dataset
    ###################################################################
    wt2 = wikitext103(batch_size=batch_size, bptt_len=bptt_len)

    ###################################################################
    # Configs
    ###################################################################
    embedding_config = DropEmbedding.Hyperparams(
        len(wt2.text_field.vocab) + 3, ninp=512)
    encoder_config = TransformerEncoder.Hyperparams(max_ext=128)

    ###################################################################
    # Models
    ###################################################################
    base_embedding = DropEmbedding(embedding_config)
    embedding = TransformerEmbedding(
        embedding=base_embedding,
        max_length=bptt_len,
        embedding_size=embedding_config.ninp,
        use_positional_embedding=True)
    encoder = TransformerEncoder(encoder_config)
    model = TransformerLanguageModel(embedding, encoder)
    model.init_weight()

    ###################################################################
    # Loss
    ###################################################################
    criterion = lm_criterion(
        in_features=encoder_config.att_num_units[-1],
        vocab_size=len(wt2.text_field.vocab))

    ###################################################################
    # Parameters + Train ops
    ###################################################################
    parameters = (list(model.parameters()) + list(criterion.parameters()))
    tot_params = 0
    for p in parameters:
        tot_params += reduce(lambda x, y: x * y, p.size())
    print("Total Parameters: ", tot_params)
    opt = optim.Adam(parameters, lr=lr)
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
        nn.utils.clip_grad_norm_(parameters, 1)
        opt.step()

        return {"train_loss": loss.item(), "train_ppl": loss.exp().item()}

    def eval_step(engine, batch):
        model.eval()

        if not hasattr(engine.state, "eval_past"):
            engine.state.eval_past = None

        with torch.no_grad():
            text = batch.text.to(DEVICE).t().contiguous()
            target = batch.target.to(DEVICE).t().contiguous()

            out, out_past = model(text, engine.state.eval_past)
            engine.state.eval_past = out_past
            raw_loss = criterion(out.view(-1, out.size(2)), target.view(-1))
            loss = raw_loss[1]

            return {"val_loss": loss.item(), "val_ppl": loss.exp().item()}

    train_engine = Engine(train_step)
    eval_engine = Engine(eval_step)

    def reset_state(engine):
        engine.state.train_past = None

    def run_eval(_):
        eval_engine.run(wt2.valid_iter)
        metrics = eval_engine.state.metrics
        print("Validation loss: ", metrics["val_loss"], ", ppl: ",
              metrics["val_ppl"])

    train_engine.add_event_handler(Events.EPOCH_STARTED, reset_state)
    train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_eval)

    ###################################################################
    # Metrics
    ###################################################################
    RunningAverage(output_transform=lambda x: x["train_ppl"]).attach(
        train_engine, "train_ppl")
    RunningAverage(output_transform=lambda x: x["train_loss"]).attach(
        train_engine, "train_loss")
    RunningAverage(output_transform=lambda x: x["val_ppl"]).attach(
        eval_engine, "val_ppl")
    RunningAverage(output_transform=lambda x: x["val_loss"]).attach(
        eval_engine, "val_loss")
    progress_bar = ProgressBar(persist=True)
    progress_bar.attach(train_engine, ["train_ppl", "train_loss"])

    ###################################################################
    # Tensorboard
    ###################################################################
    tb_logger = TensorboardLogger(log_dir="experiments")
    tb_logger.attach(
        train_engine,
        log_handler=OutputHandler(
            tag="training", output_transform=lambda loss: loss),
        event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(
        eval_engine,
        log_handler=OutputHandler(
            tag="validation", output_transform=lambda loss: loss),
        event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(
        train_engine,
        log_handler=OptimizerParamsHandler(opt),
        event_name=Events.ITERATION_STARTED)
    tb_logger.attach(
        train_engine,
        log_handler=WeightsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(
        train_engine,
        log_handler=GradsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED)

    train_engine.run(wt2.train_iter, max_epochs=epochs)


if __name__ == '__main__':
    train()
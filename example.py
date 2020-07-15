# Continuing from previous training state
if cfg.checkpointing.continue_from:
    state = TrainingState.load_state(state_path=to_absolute_path(cfg.checkpointing.continue_from))
    model = state.model
    if cfg.training.finetune:
        state.init_finetune_states(cfg.training.epochs)

    # Restore visualization metrics
    if cfg.visualization.visdom:
        visdom_logger.load_previous_values(state.epoch, state.results)
    if cfg.visualization.tensorboard:
        tensorboard_logger.load_previous_values(state.epoch, state.results)
else:
    ...
    # Initialize new training state
    state = TrainingState(model=model)
    state.init_results_tracking(epochs=cfg.training.epochs)

...

model, optimizer = amp.initialize(model, optimizer,
                                  opt_level=cfg.apex.opt_level,
                                  loss_scale=cfg.apex.loss_scale)

# Load previous optimizer/Automatic Mixed Precision (AMP) states before training begins
if state.optim_state is not None:
    optimizer.load_state_dict(state.optim_state)
    amp.load_state_dict(state.amp_state)

# Track states for optimizer/AMP
state.track_optim_state(optimizer)
state.track_amp_state(amp)

...

# Begin DeepSpeech training
for epoch in range(state.epoch, cfg.training.epochs):
    state.set_epoch(epoch=epoch)
    ...
    for i, (data) in enumerate(train_loader, start=state.training_step):
        state.set_training_step(training_step=i)

        ...  # Training step

        state.avg_loss += loss_value  # Record loss value in state

    # Record end of epoch loss
    state.avg_loss /= len(train_dataset)

    # Record metrics for visualization
    state.add_results(epoch=epoch,
                      loss_result=state.avg_loss,
                      wer_result=wer,
                      cer_result=cer)

    # Save model state
    checkpoint_handler.save_checkpoint_model(epoch=epoch, state=state)

    state.reset_training_step()  # Reset state training step for next epoch

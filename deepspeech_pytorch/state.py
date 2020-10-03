import torch

from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import remove_parallel_wrapper


class ResultState:
    def __init__(self,
                 loss_results,
                 wer_results,
                 cer_results):
        self.loss_results = loss_results
        self.wer_results = wer_results
        self.cer_results = cer_results

    def add_results(self,
                    epoch,
                    loss_result,
                    wer_result,
                    cer_result):
        self.loss_results[epoch] = loss_result
        self.wer_results[epoch] = wer_result
        self.cer_results[epoch] = cer_result

    def serialize_state(self):
        return {
            'loss_results': self.loss_results,
            'wer_results': self.wer_results,
            'cer_results': self.cer_results
        }


class TrainingState:
    def __init__(self,
                 model,
                 result_state=None,
                 optim_state=None,
                 amp_state=None,
                 best_wer=None,
                 avg_loss=0,
                 epoch=0,
                 training_step=0):
        """
        Wraps around training model and states for saving/loading convenience.
        For backwards compatibility there are more states being saved than necessary.
        """
        self.model = model
        self.result_state = result_state
        self.optim_state = optim_state
        self.amp_state = amp_state
        self.best_wer = best_wer
        self.avg_loss = avg_loss
        self.epoch = epoch
        self.training_step = training_step

    def track_optim_state(self, optimizer):
        self.optim_state = optimizer.state_dict()

    def track_amp_state(self, amp):
        self.amp_state = amp.state_dict()

    def init_results_tracking(self, epochs):
        self.result_state = ResultState(loss_results=torch.FloatTensor(epochs),
                                        wer_results=torch.FloatTensor(epochs),
                                        cer_results=torch.FloatTensor(epochs))

    def add_results(self,
                    epoch,
                    loss_result,
                    wer_result,
                    cer_result):
        self.result_state.add_results(epoch=epoch,
                                      loss_result=loss_result,
                                      wer_result=wer_result,
                                      cer_result=cer_result)

    def init_finetune_states(self, epochs):
        """
        Resets the training environment, but keeps model specific states in tact.
        This is when fine-tuning a model on another dataset where training is to be reset but the model
        weights are to be loaded
        :param epochs: Number of epochs fine-tuning.
        """
        self.init_results_tracking(epochs)
        self._reset_amp_state()
        self._reset_optim_state()
        self._reset_epoch()
        self.reset_training_step()
        self._reset_best_wer()

    def serialize_state(self, epoch, iteration):
        model = remove_parallel_wrapper(self.model)
        model_dict = model.serialize_state()
        training_dict = self._serialize_training_state(epoch=epoch,
                                                       iteration=iteration)
        results_dict = self.result_state.serialize_state()
        # Ensure flat structure for backwards compatibility
        state_dict = {**model_dict, **training_dict, **results_dict}  # Combine dicts
        return state_dict

    def _serialize_training_state(self, epoch, iteration):
        return {
            'optim_dict': self.optim_state,
            'amp': self.amp_state,
            'avg_loss': self.avg_loss,
            'best_wer': self.best_wer,
            'epoch': epoch + 1,  # increment for readability
            'iteration': iteration,
        }

    @classmethod
    def load_state(cls, state_path):
        print("Loading state from model %s" % state_path)
        state = torch.load(state_path, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(state)
        optim_state = state['optim_dict']
        amp_state = state.get('amp')
        if not amp_state:
            print("WARNING: No state for Apex has been stored in the model.")

        epoch = int(state.get('epoch', 1)) - 1  # Index start at 0 for training
        training_step = state.get('iteration', None)
        if training_step is None:
            epoch += 1  # We saved model after epoch finished, start at the next epoch.
            training_step = 0
        else:
            training_step += 1
        avg_loss = int(state.get('avg_loss', 0))
        loss_results = state['loss_results']
        cer_results = state['cer_results']
        wer_results = state['wer_results']
        best_wer = state.get('best_wer')

        result_state = ResultState(loss_results=loss_results,
                                   cer_results=cer_results,
                                   wer_results=wer_results)
        return cls(optim_state=optim_state,
                   amp_state=amp_state,
                   model=model,
                   result_state=result_state,
                   best_wer=best_wer,
                   avg_loss=avg_loss,
                   epoch=epoch,
                   training_step=training_step)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_best_wer(self, wer):
        self.best_wer = wer

    def set_training_step(self, training_step):
        self.training_step = training_step

    def reset_training_step(self):
        self.training_step = 0

    def reset_avg_loss(self):
        self.avg_loss = 0

    def _reset_amp_state(self):
        self.amp_state = None

    def _reset_optim_state(self):
        self.optim_state = None

    def _reset_epoch(self):
        self.epoch = 0

    def _reset_best_wer(self):
        self.best_wer = None

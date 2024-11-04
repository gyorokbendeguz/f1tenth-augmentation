import deepSI
import torch
from torch import nn
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from deepSI.fit_systems.fit_system import print_array_byte_size, My_Simple_DataLoader, Tictoctimer
import time
from tqdm.auto import tqdm
import itertools
import numpy as np
from model_augmentation.torch_nets import integrator_RK4


class augmentation_encoder(deepSI.fit_systems.SS_encoder_general_hf):
    def __init__(self, nx, na, nb, augm_net, e_net, augm_net_kwargs={}, e_net_kwargs={}, na_right=0, nb_right=0):
        super(augmentation_encoder, self).__init__(nx=nx, na=na, nb=nb, hf_net=augm_net, hf_net_kwargs=augm_net_kwargs,
                                                   e_net=e_net, e_net_kwargs=e_net_kwargs, na_right=na_right,
                                                   nb_right=nb_right, feedthrough=True)
        #  initialize custom augmentation-related terms
        self.U1_orth = None
        self.X_orth = None
        self.U_orth = None
        self.use_orthogonal_loss = False
        self.use_parm_reg_loss = False
        self.parms_orig_regularization = None
        self.regularizationMx = None
        self.use_noise_struct = False
        self.use_l2_reg = False
        self.l2_reg = 0

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):

        # get simulation error on training data
        x = self.encoder(uhist, yhist)  # initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture, 0, 1), torch.transpose(ufuture, 0, 1)):  # iterate over time
            if self.use_noise_struct:
                # if innovation noise structure is used
                yhat, x = self.hfn(x, u, y)
            else:
                yhat, x = self.hfn(x, u)
            errors.append(nn.functional.mse_loss(y, yhat))  # calculate error after taking n-steps
        mse_loss = torch.mean(torch.stack(errors))

        # orthogonalization-based cost term
        if self.use_orthogonal_loss:
            orthCost = self.hfn.calculate_orthogonalisation(self.X_orth, self.U_orth, self.U1_orth)
        else:
            orthCost = 0

        # get parameter regularization cost term (if needed)
        if self.use_parm_reg_loss:
            parmsTuned = self.hfn.sys.P
            parmsDiff = parmsTuned - self.parms_orig_regularization
            parm_regularization = parmsDiff.unsqueeze(dim=0) @ self.regularizationMx @ parmsDiff.unsqueeze(dim=1)
        else:
            parm_regularization = 0

        # L2 regularization
        l2_loss = torch.tensor(0.)

        if self.use_l2_reg:
            for param in self.hfn.net.parameters():
                l2_loss += self.l2_reg * torch.linalg.norm(param)**2

            if hasattr(self.encoder, 'net'):
                for param in self.encoder.net.parameters():
                    l2_loss += self.l2_reg * torch.linalg.norm(param) ** 2

        return mse_loss + (parm_regularization + orthCost + l2_loss) / self.N_batch_updates_per_epoch

    def init_augmentation_loss_params(self, **loss_kwargs):

        # check for orthogonalization parameters is loss_kwargs
        if 'U1_orth' in loss_kwargs:
            self.U1_orth = loss_kwargs.get('U1_orth')
            self.X_orth = loss_kwargs.get('X')
            self.U_orth = loss_kwargs.get('U')
            U1_orth_exists = True
        else:
            U1_orth_exists = False

        # check is orthogonalization is needed at all
        if hasattr(self.hfn, 'Pcorr_enab') and hasattr(self.hfn, 'orthLambda') and self.hfn.orthLambda != 0 and U1_orth_exists:
            # orthogonalization is necessary
            self.use_orthogonal_loss = True
            print('Orthogonalization-based cost term is used for the loss function.')
        else:
            self.use_orthogonal_loss = False

        # check is parameter regularization is necessary
        if hasattr(self.hfn, 'Pcorr_enab') and self.hfn.Pcorr_enab and self.hfn.regLambda != 0:
            self.use_parm_reg_loss = True
            print('Regularization of the physical parameters is used for the loss function.')
        else:
            self.use_parm_reg_loss = False

        # get parameter regularization parameters
        if self.use_parm_reg_loss:
            self.parms_orig_regularization = self.hfn.sys.P_orig
            nParms = self.parms_orig_regularization.shape[0]
            self.regularizationMx = self.hfn.regLambda * torch.min(torch.diag(1 / self.parms_orig_regularization ** 2),
                                                                   1e6 * torch.ones(nParms, nParms))

        # check if innovation noise structure is used
        if hasattr(self.hfn, 'innov') and self.hfn.innov:
            self.use_noise_struct = True
            print("Innovation noise structure is used during training the model.")
        else:
            self.use_noise_struct = False

        # L2 regularization
        if self.hfn.l2_reg > 0:
            self.use_l2_reg = True
            self.l2_reg = self.hfn.l2_reg
            print("L2 regularization applied to augmentation and encoder nets.")

        return

    def fit(self, train_sys_data, val_sys_data, epochs=30, batch_size=256, loss_kwargs={}, auto_fit_norm=True,
            validation_measure='sim-NRMS', optimizer_kwargs={}, concurrent_val=False, cuda=False, timeout=None,
            verbose=1, sqrt_train=True, num_workers_data_loader=0, print_full_time_profile=False, scheduler_kwargs={}):
        """
        Only modification to the original deepSI implementation is that N_batch_updates_per_epoch is included into the
        class attributes.

        Further information can be found: https://deepsi.readthedocs.io/en/latest/fit_systems.html#deepSI.fit_systems.System_torch.fit
        """

        def validation(train_loss=None, time_elapsed_total=None):
            self.eval()
            self.cpu()
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
            self.Loss_val.append(Loss_val)
            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            if self.bestfit >= Loss_val:
                self.bestfit = Loss_val
                self.checkpoint_save_system()
            if cuda:
                self.cuda()
            self.train()
            return Loss_val

        # --------------- Initialization ---------------
        if self.init_model_done is False:
            if verbose:
                print('Initilizing the model and optimizer')
            device = 'cuda' if cuda else 'cpu'
            optimizer_kwargs = deepcopy(optimizer_kwargs)
            parameters_optimizer_kwargs = optimizer_kwargs.get('parameters_optimizer_kwargs', {})
            if parameters_optimizer_kwargs:
                del optimizer_kwargs['parameters_optimizer_kwargs']
            self.init_model(sys_data=train_sys_data, device=device, auto_fit_norm=auto_fit_norm,
                            optimizer_kwargs=optimizer_kwargs, parameters_optimizer_kwargs=parameters_optimizer_kwargs,
                            scheduler_kwargs=scheduler_kwargs)
        else:
            if verbose:
                print('Model already initilized (init_model_done=True), skipping initilizing of the model, the norm and the creation of the optimizer')
            self._check_and_refresh_optimizer_if_needed()

        if self.scheduler is False and verbose:
            print('!!!! Your might be continuing from a save which had scheduler but which was removed during saving... check this !!!!!!')

        # -------- Setting model augmentation-related loss function parameters --------
        self.init_augmentation_loss_params(**loss_kwargs)

        self.dt = train_sys_data.dt
        if cuda:
            self.cuda()
        self.train()

        self.epoch_counter = 0 if len(self.epoch_id) == 0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id) == 0 else self.batch_id[-1]
        extra_t = 0 if len(self.time) == 0 else self.time[-1]  # correct timer after restart

        # -------- Getting the data --------
        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        if not isinstance(data_train, Dataset) and verbose:
            print_array_byte_size(sum([d.nbytes for d in data_train]))

        # -------- transforming it back to a list to be able to append. --------
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(
            self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        # -------- init monitoring values --------
        Loss_acc_val, N_batch_acc_val, val_counter, best_epoch, batch_id_start = 0, 0, 0, 0, self.batch_counter  # to print the frequency of the validation step.
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        self.N_batch_updates_per_epoch = N_training_samples // batch_size
        if verbose > 0:
            print(
                f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {self.N_batch_updates_per_epoch}')

        # -------- convert to dataset --------
        if isinstance(data_train, Dataset):
            persistent_workers = False if num_workers_data_loader == 0 else True
            data_train_loader = DataLoader(data_train, batch_size=batch_size, drop_last=True, shuffle=True, \
                                           num_workers=num_workers_data_loader, persistent_workers=persistent_workers)
        else:  # add my basic DataLoader
            data_train_loader = My_Simple_DataLoader(data_train, batch_size=batch_size)  # is quite a bit faster for low data situations

        if concurrent_val:
            self.remote_start(val_sys_data, validation_measure)
            self.remote_send(float('nan'), extra_t)
        else:  # start with the initial validation
            validation(train_loss=float('nan'), time_elapsed_total=extra_t)  # also sets current model to cuda
            if verbose:
                print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        try:
            t = Tictoctimer()
            start_t = time.time()  # time keeping
            epochsrange = range(epochs) if timeout is None else itertools.count(start=0)
            if timeout is not None and verbose > 0:
                print(f'Starting indefinite training until {timeout} seconds have passed due to provided timeout')

            for epoch in (tqdm(epochsrange) if verbose > 0 else epochsrange):
                bestfit_old = self.bestfit  # to check if a new lowest validation loss has been achieved
                Loss_acc_epoch = 0.
                t.start()
                t.tic('data get')
                for train_batch in data_train_loader:
                    if cuda:
                        train_batch = [b.cuda() for b in train_batch]
                    t.toc('data get')

                    def closure(backward=True):
                        t.toc('optimizer start')
                        t.tic('loss')
                        Loss = self.loss(*train_batch, **loss_kwargs)
                        t.toc('loss')
                        if backward:
                            t.tic('zero_grad')
                            self.optimizer.zero_grad()
                            t.toc('zero_grad')
                            t.tic('backward')
                            Loss.backward()
                            t.toc('backward')
                        t.tic('stepping')
                        return Loss

                    t.tic('optimizer start')
                    training_loss = self.optimizer.step(closure).item()
                    t.toc('stepping')
                    if self.scheduler:
                        t.tic('scheduler')
                        self.scheduler.step()
                        t.tic('scheduler')
                    Loss_acc_val += training_loss
                    Loss_acc_epoch += training_loss
                    N_batch_acc_val += 1
                    self.batch_counter += 1
                    self.epoch_counter += 1 / self.N_batch_updates_per_epoch

                    t.tic('val')
                    if concurrent_val and self.remote_recv():  # -------- validation --------
                        self.remote_send(Loss_acc_val / N_batch_acc_val, time.time() - start_t + extra_t)
                        Loss_acc_val, N_batch_acc_val, val_counter = 0., 0, val_counter + 1
                    t.toc('val')
                    t.tic('data get')
                t.toc('data get')

                # --------------- end of epoch clean up ---------------
                train_loss_epoch = Loss_acc_epoch / self.N_batch_updates_per_epoch
                if np.isnan(train_loss_epoch):
                    if verbose > 0:
                        print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at epoch {epoch}, breaking from loop &&&&&&&&&&')
                    break

                t.tic('val')
                if not concurrent_val:
                    validation(train_loss=train_loss_epoch, time_elapsed_total=time.time() - start_t + extra_t)  # updates bestfit and goes back to cpu and back
                t.toc('val')
                t.pause()

                # -------- Printing Routine --------
                if verbose > 0:
                    time_elapsed = time.time() - start_t
                    if bestfit_old > self.bestfit:
                        print(
                            f'########## New lowest validation loss achieved ########### {validation_measure} = {self.bestfit}')
                        best_epoch = epoch + 1
                    if concurrent_val:  # if concurrent val than print validation freq
                        val_feq = val_counter / (epoch + 1)
                        valfeqstr = f', {val_feq:4.3} vals/epoch' if (
                                    val_feq > 1 or val_feq == 0) else f', {1 / val_feq:4.3} epochs/val'
                    else:  # else print validation time use
                        valfeqstr = f''
                    trainstr = f'sqrt loss {train_loss_epoch ** 0.5:7.4}' if sqrt_train and train_loss_epoch >= 0 else f'loss {train_loss_epoch:7.4}'
                    Loss_val_now = self.Loss_val[-1] if len(self.Loss_val) != 0 else float('nan')
                    Loss_str = f'Epoch {epoch + 1:4}, {trainstr}, Val {validation_measure} {Loss_val_now:6.4}'
                    loss_time = (t.acc_times['loss'] + t.acc_times['optimizer start'] + t.acc_times['zero_grad'] +
                                 t.acc_times['backward'] + t.acc_times['stepping']) / t.time_elapsed
                    time_str = f'Time Loss: {loss_time:.1%}, data: {t.acc_times["data get"] / t.time_elapsed:.1%}, val: {t.acc_times["val"] / t.time_elapsed:.1%}{valfeqstr}'
                    self.batch_feq = (self.batch_counter - batch_id_start) / (time.time() - start_t)
                    batch_str = (f'{self.batch_feq:4.1f} batches/sec' if (
                                self.batch_feq > 1 or self.batch_feq == 0) else f'{1 / self.batch_feq:4.1f} sec/batch')
                    print(f'{Loss_str}, {time_str}, {batch_str}')
                    if print_full_time_profile:
                        print('Time profile:', t.percent())

                # -------- Timeout Breaking --------
                if timeout is not None:
                    if time.time() >= start_t + timeout:
                        break
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')

        self.train()
        self.cpu()
        del data_train_loader

        # --------------- end of training concurrent things ---------------
        if concurrent_val:
            if verbose:
                print(
                f'Waiting for started validation process to finish and one last validation... (receiving = {self.remote.receiving})',
                end='')
            if self.remote_recv(wait=True):
                if verbose:
                    print('Recv done... ', end='')
                if N_batch_acc_val > 0:
                    self.remote_send(Loss_acc_val / N_batch_acc_val, time.time() - start_t + extra_t)
                    self.remote_recv(wait=True)
            self.remote_close()
            if verbose:
                print('Done!')

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(
            self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
        if verbose:
            print(
                f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_epoch} (epoch_id={self.epoch_id[-1] if len(self.epoch_id) > 0 else 0:.2f})')


# ----------------------------------------------------------------------------------------------------------------------
#                                         CT AUGMENTATION ENCODER
# ----------------------------------------------------------------------------------------------------------------------

class augmentation_encoder_deriv(augmentation_encoder):
    def __init__(self, nx, na, nb, augm_net, e_net, integrator_net=integrator_RK4, augm_net_kwargs={}, e_net_kwargs={},
                 integrator_net_kwargs={}, na_right=0, nb_right=0):
        super(augmentation_encoder_deriv, self).__init__(nx=nx, na=na, nb=nb, augm_net=augm_net, e_net=e_net,
                                                         augm_net_kwargs=augm_net_kwargs, e_net_kwargs=e_net_kwargs,
                                                         na_right=na_right, nb_right=nb_right)
        self.integrator_net = integrator_net
        self.integrator_net_kwargs = integrator_net_kwargs

    def init_nets(self, nu, ny):  # a bit weird
        na_right = self.na_right if hasattr(self, 'na_right') else 0
        nb_right = self.nb_right if hasattr(self, 'nb_right') else 0
        self.encoder = self.e_net(nb=self.nb + nb_right, nu=nu, na=self.na + na_right, ny=ny, nx=self.nx,
                                  **self.e_net_kwargs)
        self.hfn = self.hf_net(nx=self.nx, nu=self.nu, ny=self.ny, **self.hf_net_kwargs)
        self.derivn = self.hfn  # move fn to become the derivative net
        self.excluded_nets_from_parameters = ['derivn']
        self.hfn = self.integrator_net(augm_structure=self.derivn, **self.integrator_net_kwargs)

    def init_augmentation_loss_params(self, **loss_kwargs):

        # check for orthogonalization parameters is loss_kwargs
        if 'U1_orth' in loss_kwargs:
            self.U1_orth = loss_kwargs.get('U1_orth')
            self.X_orth = loss_kwargs.get('X')
            self.U_orth = loss_kwargs.get('U')
            U1_orth_exists = True
        else:
            U1_orth_exists = False

        # check is orthogonalization is needed at all
        if hasattr(self.hfn, 'Pcorr_enab') and hasattr(self.hfn,
                                                       'orthLambda') and self.hfn.orthLambda != 0 and U1_orth_exists:
            # orthogonalization is necessary
            self.use_orthogonal_loss = True
            print('Orthogonalization-based cost term is used for the loss function.')
        else:
            self.use_orthogonal_loss = False

        # check is parameter regularization is necessary
        if hasattr(self.hfn, 'Pcorr_enab') and self.hfn.Pcorr_enab and self.hfn.regLambda != 0:
            self.use_parm_reg_loss = True
            print('Regularization of the physical parameters is used for the loss function.')
        else:
            self.use_parm_reg_loss = False

        # get parameter regularization parameters
        if self.use_parm_reg_loss:
            self.parms_orig_regularization = self.hfn.sys.P_orig
            nParms = self.parms_orig_regularization.shape[0]
            self.regularizationMx = self.hfn.regLambda * torch.min(torch.diag(1 / self.parms_orig_regularization ** 2),
                                                                   1e6 * torch.ones(nParms, nParms))

        # check if innovation noise structure is used
        if hasattr(self.hfn, 'innov') and self.hfn.innov:
            self.use_noise_struct = True
            print("Innovation noise structure is used during training the model.")
        else:
            self.use_noise_struct = False

        # L2 regularization
        if self.hfn.augm_structure.l2_reg > 0:
            self.use_l2_reg = True
            self.l2_reg = self.hfn.augm_structure.l2_reg
            print("L2 regularization applied to augmentation and encoder nets.")

        return

    """
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        self.hfn.dt = dt
        """

import torch
import torch.nn as nn
import statistics

class APHYNITYExperiment():
    def __init__(self, train, test, net, optimizer, min_op, lambda_0, tau_2, niter, nlog, nupdate, nepoch):
        self.path = './exp'
        self.traj_loss = nn.MSELoss()
        self.net = net.to('cpu')
        self.optimizer = optimizer

        self.min_op = min_op
        self.tau_2 = tau_2
        self.niter = niter
        self._lambda = lambda_0

        self.nlog = nlog
        self.nupdate = nupdate
        self.nepoch = nepoch
        self.train = train
        self.test = test

    def train_step(self, batch):
        self.net.train(True)
        #batch = torch.as_tensor(batch, device ='cpu')
        loss, output = self.step(batch)
        with torch.no_grad():
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def val_step(self, batch):
        self.net.train(False)
        with torch.no_grad():
            loss, output = self.step(batch, backward=False)
            metric = self.metric(**output, **batch)

        return batch, output, loss, metric

    def log(self, epoch, metrics):
        message = '[{epoch}/{max_epoch}]'.format(
            epoch=epoch +1,
            max_epoch=self.nepoch
        )
        for name, value in metrics.items():
            message += ' | {name}: {value:.2e}'.format(name=name, value=value)
        print(message)

    def lambda_update(self, loss):
        self._lambda = self._lambda + self.tau_2 * loss

    def _forward(self, states, t, u, backward):
        target = states
        y0 = states[:, :, 0]
        pred = self.net(y0, t, u)
        loss = self.traj_loss(pred, target)
        loss_op  = 0


        if self.net.model_aug != None :
            aug_deriv = self.net.model_aug.get_derivatives(states,u)
            if self.min_op == 'l2_normalized':
                loss_op = ((aug_deriv.norm(p=2, dim=1) / (states.norm(p=2, dim=1) +1e-5)) ** 2).mean()
            elif self.min_op == 'l2':
                loss_op = (aug_deriv.norm(p=2, dim=1) ** 2).mean()

        if backward:
            loss_total = loss * self._lambda + loss_op
            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss = {
            'loss': loss,
            'loss_op': loss_op,
        }

        output = {
            'states_pred': pred,
        }
        return loss, output

    def step(self, batch, backward=True):
        states = batch['states']
        t = batch['t'][0]
        u = None
        if batch['actions'] is not None:
            u = batch['actions']
        loss, output = self._forward(states, t, u, backward)
        return loss, output

    def metric(self, states, states_pred, **kwargs):
        metrics = {}
        if self.net.model_phy is None : return metrics
        metrics['param_error'] = statistics.mean(abs(v1-float(v2))/v1 for v1, v2 in zip(self.train.dataset.params.values(), self.net.model_phy.params.values()))
        metrics.update({f'{k}': v.data.item() for k,v in self.net.model_phy.params.items()})
        metrics.update({f'{k}_real': v for k, v in self.train.dataset.params.items() if k in metrics})
        return metrics

    def run(self):
        loss_test_min = None
        for epoch in range(self.nepoch):
                for _ in range(self.niter):
                    _, _, loss, metric = self.train_step(next(iter(self.train)))
                loss_train = loss['loss'].item()
                self.lambda_update(loss_train)

                if (epoch+1)  % self.nlog == 0:
                    self.log(epoch, loss | metric)

                if (epoch+1) % self.nupdate == 0:
                    with torch.no_grad():
                        _, _, loss, metric = self.val_step(next(iter(self.test)))
                        loss_test = loss['loss'].item()

                        if True : #loss_test_min == None or loss_test_min > loss_test:
                            loss_test_min = loss_test
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss_test_min,
                            }, self.path + f'/model_{loss_test_min:.2e}.pt')
                        loss_test = {
                            'loss_test': loss_test,
                        }
                        print('#' * 80)
                        self.log(epoch, loss_test | metric)
                        print(f'lambda: {self._lambda}')
                        print('#' * 80)
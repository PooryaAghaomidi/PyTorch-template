# -*- coding: utf-8 -*-
"""Train model"""

import torch
from tqdm import tqdm


class TrainModel:
    def __init__(self,
                 model,
                 train_gen,
                 val_gen,
                 optimizer,
                 loss,
                 epoch,
                 batch_size,
                 steps_per_epoch,
                 steps_per_val,
                 writer,
                 model_name,
                 info_interval):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.opt = optimizer
        self.loss = loss
        self.epoch = epoch
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_val = steps_per_val
        self.writer = writer
        self.model_name = model_name
        self.info_interval = info_interval

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        running_acc = 0.
        last_acc = 0.
        total = 0

        for i in tqdm(range(self.steps_per_epoch)):
            x, y_true = self.train_gen.data_generation(i)

            self.opt.zero_grad()

            y_pred = self.model(x)
            loss = self.loss(y_pred, y_true)
            loss.backward()

            self.opt.step()

            running_loss += loss.item()
            _, predicted_output = torch.max(y_pred, 1)
            _, predicted_target = torch.max(y_true, 1)

            total += x.size(0)
            running_acc += (predicted_output == predicted_target).sum().item()

            if i % self.info_interval == (self.info_interval - 1):
                last_loss = running_loss / total
                last_acc = running_acc / total

                print('\n')
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                print('  batch {} accuracy: {}'.format(i + 1, round(last_acc*100, 2)))
                print('\n')

                tb_x = epoch_index * self.steps_per_epoch + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', last_acc, tb_x)
                running_loss = 0.

        return last_loss, last_acc

    def training(self):
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(self.epoch):
            print('\n\n ####################### EPOCH {} ####################### \n'.format(epoch_number + 1))

            self.model.train(True)
            avg_loss, avg_acc = self.train_one_epoch(epoch_number, self.writer)

            running_vloss = 0.0
            running_vacc = 0.0
            total = 0
            self.model.eval()

            with torch.no_grad():
                for i in range(self.steps_per_val):
                    xv, yv_true = self.val_gen.data_generation(i)

                    yv_pred = self.model(xv)
                    vloss = self.loss(yv_pred, yv_true)

                    running_vloss += vloss
                    _, predicted_output = torch.max(yv_pred, 1)
                    _, predicted_target = torch.max(yv_true, 1)

                    total += xv.size(0)
                    running_vacc += (predicted_output == predicted_target).sum().item()

            avg_vloss = running_vloss / total
            avg_vacc = running_vacc / total

            print('LOSS train {}, validation {}'.format(avg_loss, avg_vloss))
            print('ACCURACY train {}, validation {}'.format(round(avg_acc*100, 2), round(avg_vacc*100, 2)))

            self.writer.add_scalars('Training vs. Validation loss',
                                    {'Training_loss': avg_loss, 'Validation_loss': avg_vloss},
                                    epoch_number + 1)
            self.writer.add_scalars('Training vs. Validation accuracy',
                                    {'Training_accuracy': avg_acc, 'Validation_accuracy': avg_vacc},
                                    epoch_number + 1)
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), self.model_name)

            epoch_number += 1

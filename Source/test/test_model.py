# -*- coding: utf-8 -*-
""" Test the model """

import torch


def test_model(model, model_path, test_gen, steps_per_test):
    print('\n\n ####################### Test results: #######################')

    model.load_state_dict(torch.load(model_path))

    accuracy = 0.0
    total = 0
    with torch.no_grad():
        for i in range(steps_per_test):
            x, y_true = test_gen.data_generation(i)

            y_pred = model(x)

            _, predicted_output = torch.max(y_pred, 1)
            _, predicted_target = torch.max(y_true, 1)

            total += x.size(0)
            accuracy += (predicted_output == predicted_target).sum().item()

    avg_vacc = (accuracy / total) * 100

    print('\n')
    print('Test accuracy is {}'.format(round(avg_vacc*100, 2)))

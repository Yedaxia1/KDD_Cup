# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import argparse
import paddle
import paddle.nn.functional as F
import tqdm
import yaml
import numpy as np
from easydict import EasyDict as edict

import pgl
from pgl.utils.logger import log
from paddle.io import DataLoader
import random

import time

from common import Experiment

from test_data import TestData

import loss as loss_factory
from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset
from model import WPFModel
import optimization as optim
from metrics import regressor_scores, regressor_detailed_scores
from utils import save_model, _create_if_not_exist, load_model
import matplotlib.pyplot as plt


@paddle.no_grad()
def predict(settings, train_data, test_x):  #, valid_data, test_data):
    data_mean = paddle.to_tensor(train_data.data_mean, dtype="float32")
    data_scale = paddle.to_tensor(train_data.data_scale, dtype="float32")

    graph = train_data.graph
    graph = graph.tensor()

    con_path = os.path.realpath(__file__)[:-10] + 'config.yaml'
    config = edict(yaml.load(open(con_path), Loader=yaml.FullLoader))

    model = WPFModel(config=config)

    model_path = os.path.realpath(__file__)[:-10] + 'output/baseline'
    global_step = load_model(model_path, model)
    model.eval()

    test_x_ds = TestPGL4WPFDataset(filename=settings["path_to_test_x"])

    ex_test_y_path = os.path.realpath(__file__)[:-10] + "predict_data/test_y/0001out.csv"
    test_y_ds = TestPGL4WPFDataset(filename=ex_test_y_path)

    test_x = paddle.to_tensor(
        test_x_ds.get_data()[:, :, -config.input_len:, :], dtype="float32")
    test_y = paddle.to_tensor(
        test_y_ds.get_data()[:, :, :config.output_len, :], dtype="float32")

    pred_y = model(test_x, test_y, data_mean, data_scale, graph)
    pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :,
                                                                    -1])

    pred_y = np.expand_dims(pred_y.numpy(), -1)
    test_y = test_y[:, :, :, -1:].numpy()

    pred_y = np.transpose(pred_y, [
        1,
        0,
        2,
        3,
    ])
    test_y = np.transpose(test_y, [
        1,
        0,
        2,
        3,
    ])
    test_y_df = test_y_ds.get_raw_df()

    # _mae, _rmse = regressor_detailed_scores(
    #     pred_y, test_y, test_y_df, config.capacity, config.output_len)
    # print('\n\tThe {}-th prediction for File {} -- '
    #       'RMSE: {}, MAE: {}, Score: {}'.format(i, test_y_f, _rmse, _mae, (
    #           _rmse + _mae) / 2))
    # maes.append(_mae)
    # rmses.append(_rmse)

    res_pred_y = np.squeeze(pred_y, axis=1)

    # avg_mae = np.array(maes).mean()
    # avg_rmse = np.array(rmses).mean()
    # total_score = (avg_mae + avg_rmse) / 2

    # print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
    # print('--- Final Score --- \n\t{}'.format(total_score))

    return res_pred_y


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions as a tensor \in R^{134 * 288 * 1}
    """
    start_time = time.time()
    size = [settings["input_len"], settings["output_len"]]
    # predictions = []
    # settings["turbine_id"] = 0
    exp = Experiment(settings)
    # train_data = Experiment.train_data
    train_data = PGL4WPFDataset(
        settings["data_path"],
        filename=settings["filename"],
        size=size,
        flag='train',
        total_days=settings["total_size"],
        train_days=settings["train_size"],
        val_days=settings["val_size"],
        test_days=settings["val_size"])
    if settings["is_debug"]:
        end_train_data_get_time = time.time()
        print("Load train data in {} secs".format(end_train_data_get_time - start_time))
        start_time = end_train_data_get_time

    test_x = Experiment.get_test_x(settings)
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("Get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time


    predictions = predict(settings, train_data, test_x)  #, valid_data, test_data)

    return np.array(predictions)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))

    print(config)
    size = [config.input_len, config.output_len]
    train_data = PGL4WPFDataset(
        config.data_path,
        filename=config.filename,
        size=size,
        flag='train',
        total_days=config.total_days,
        train_days=config.train_days,
        val_days=config.val_days,
        test_days=config.test_days)

    predict(config, train_data)  #, valid_data, test_data)

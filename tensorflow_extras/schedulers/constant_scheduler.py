# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class ConstantScheduler(LearningRateSchedule):
    """
    A simple scheduler that just returns a constant learning rate.
    This is used mainly so constants can be combined with other schedulers.
    """
    def __init__(self, learning_rate):
        super(ConstantScheduler, self).__init__()
        self.learning_rate = learning_rate
    
    @tf.function
    def __call__(self, step):
        # just throw out step and return a constant value
        return self.learning_rate
    
    def get_config(self):
        scheduler_config = {'learning_rate': self.learning_rate}
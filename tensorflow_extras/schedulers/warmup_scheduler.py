# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow_extras.schedulers.constant_scheduler import ConstantScheduler
from typing import Union

class WarmupScheduler(LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear warmup
    """
    
    def __init__(self, schedule: Union[float, LearningRateSchedule], 
                 initial_learning_rate: float, warmup_steps: int, warmup_type='linear',
                 dtype=tf.float32):
        super(WarmupScheduler, self).__init__()
        if isinstance(schedule, float):
            schedule = ConstantScheduler(schedule)
        self.scheduler = schedule
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.warmup_type = warmup_type
        self.dtype = dtype
        self.scheduler_learning_rate = schedule(0)
        
    def compute_linear_warmup(self, step):
        return ((self.scheduler_learning_rate*step) + \
                (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
    
    @tf.function
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp>=self.warmup_steps:
            return self.scheduler(global_step_recomp - self.warmup_steps)
        return self.compute_linear_warmup(global_step_recomp)
    
    def get_config(self):
        scheduler_config = self.scheduler.get_config()
        scheduler_config['initial_learning_rate'] = self.initial_learning_rate
        scheduler_config['warmup_steps'] = self.warmup_steps
        scheduler_config['warmup_type'] = self.warmup_type
        return scheduler_config
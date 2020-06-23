# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from typing import Union, List

class CombinationScheduler(LearningRateSchedule):
    """
    Takes two lists of breakpoints and schedulers, creates a new scheduler that
    transitions between each in the list. For example, to create a scheduler with
    a linear warmup, period of constant learning rate, and finally an exponential
    decay use
    schedule_constant = ConstantScheduler(0.01)
    schedule_constant = WarmupScheduler(scehdule_constant, 0.001, 500)
    decay_schedule = ExponentialDecay(0.01, 10000, 0.001)
    break_points = 5000
    scehdule = CombinationScheduler(break_points, [scehdule_constant, decay_schedule])
    """
    def __init__(self, break_points: Union[int, List[int]], 
                 schedulers: List[LearningRateSchedule], 
                 dtype=tf.float32):
        super(CombinationScheduler, self).__init__()
        if isinstance(break_points, int):
            break_points = [break_points]
        assert len(break_points)+1==len(schedulers), \
                "Number of schedulers must be one longer than break points"
        self.break_points = break_points
        self.break_points.append(-1)
        self.schedulers = schedulers
        self.dtype = dtype
        self.schedule_pos = 0
        self.next_breakpoint = 0
        self.offset = tf.constant(0, dtype=self.dtype)
        
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if step==self.break_points[self.next_breakpoint]:
            self.offset = tf.cast(self.break_points[self.next_breakpoint], self.dtype)
            self.next_breakpoint+=1
            self.schedule_pos+=1
        return self.schedulers[self.schedule_pos](global_step_recomp - self.offset)
    
    def get_config(self):
        scheduler_config = {'scheduler_configs': 
                            [scheduler.get_config() for scheduler in self.schedulers]}
        scheduler_config['break_points'] = self.break_points
        return scheduler_config
    
        
        
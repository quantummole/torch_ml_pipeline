# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:15:00 2018

@author: quantummole
"""

from signals import Signal
from copy import deepcopy

class Pipeline :
    def __init__(self,op_set,input_names,score_combiner,pipeline = None) :
        self.op_set = op_set
        self.input_names = input_names
        self.pipeline = pipeline
        self.output_names = None if not self.pipeline else self.pipeline.input_names
        self.score_combiner = score_combiner
    def __repr__(self) :
        return repr(self.op_set)
    def execute(self,inputs, params_state) :
        scores = []
        for op in self.op_set :
            self.inputs = deepcopy(inputs)
            op.initialize()
            curr_score = Signal.NO_SCORE
            op_scores = []
            op_inputs = dict([(name,inputs.get(name,None)) for name in self.input_names])
            while op.completion_state == Signal.INCOMPLETE :
                new_params = deepcopy(params_state)
                new_params, outputs = op.execute(curr_score,new_params,op_inputs)
                if self.pipeline :
                    for i,name in enumerate(outputs) :
                        self.inputs[self.output_names[i]] = outputs[i]
                    curr_score = self.pipeline.execute(self.inputs,new_params)
                    op_scores.append(curr_score)
                else :
                    op_scores.append(outputs)
            scores.append(op_scores)
        return self.score_combiner(scores)


class PipelineOp :
    def __init__(self,op_name,op_class,config_sets,evaluator = None) :
        self.op_name,self.op_class,self.config_sets,self.evaluator = op_name,op_class,config_sets,evaluator
        print("{} has {} configurations".format(self.op_name,self.config_sets.max_configs))
        self.op_completion_state = Signal.COMPLETE
        self.completion_state = Signal.INCOMPLETE
        self.params = None
        self.params_id = None
        self.curr_op = None
    def __repr__(self):
        return repr(self.op_name)+repr(self.op_class)+repr(self.config_sets)
    def initialize(self) :
        self.config_sets.reinitialize()
        self.op_completion_state = Signal.COMPLETE
        self.completion_state = Signal.INCOMPLETE
        self.params = None
        self.params_id = None
        self.curr_op = None
    def execute(self,score,global_params,inputs) :
        if self.completion_state == Signal.INCOMPLETE and self.op_completion_state == Signal.COMPLETE:
            self.params_id,self.params =  self.config_sets.get_next(score)
            self.new_params = global_params
            self.new_params[self.op_name] = self.params
            if self.evaluator :
                self.evaluator.set_config(self.new_params)
                self.evaluator.log_config()
            self.curr_op = self.op_class(**self.params,evaluator = self.evaluator)
            self.op_completion_state = Signal.INCOMPLETE
        if self.op_completion_state == Signal.INCOMPLETE :
            self.op_completion_state,outputs = self.curr_op.execute(**inputs)
        self.completion_state = Signal.COMPLETE if self.config_sets.completion_state == self.op_completion_state and self.config_sets.completion_state == Signal.COMPLETE else Signal.INCOMPLETE 
        return self.new_params, outputs
        
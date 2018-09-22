# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:15:00 2018

@author: quantummole
"""

from signals import Signal
from copy import deepcopy

class Pipeline :
    def __init__(self,op_set,input_names,score_combiner,evaluator = None,pipeline = None) :
        self.op_set = op_set
        self.input_names = input_names
        self.pipeline = pipeline
        self.output_names = None if not self.pipeline else self.pipeline.input_names
        self.evaluator = evaluator
        self.score_combiner = score_combiner
    def __repr__(self) :
        return repr(self.op_set)
    def execute(self,inputs,curr_pipeline_id, params_state) :
        scores = []
        for op_name,op_class,config_set in self.op_set :
            config_space = (op_class,config_set)
            self.inputs = deepcopy(inputs)
            op = PipelineOp(curr_pipeline_id+"#"+op_name,config_space)
            curr_score = Signal.NO_SCORE
            op_scores = []
            op_inputs = dict([(name,inputs.get(name,None)) for name in self.input_names])
            while op.completion_state == Signal.INCOMPLETE :
                if self.evaluator :
                    self.evaluator.set_config(params_state)
                new_params_state = deepcopy(params_state)
                new_params = op.generate_config(curr_score,self.evaluator)
                new_params_state[op_name]  = new_params
                if self.evaluator :
                    self.evaluator.log_config()
                next_pipeline_id,outputs = op.execute(op_inputs)
                if self.pipeline :
                    for i,name in enumerate(self.output_names) :
                        self.inputs[name] = outputs[i]
                    curr_score = self.pipeline.execute(self.inputs,next_pipeline_id,new_params_state)
                    op_scores.append(curr_score)
                else :
                    op_scores.append(outputs)
                if self.evaluator :
                    self.evaluator.log_score(next_pipeline_id,outputs)
            scores.append(op_scores)
        return self.score_combiner(scores)

class PipelineOp :
    def __init__(self,curr_pipeline_id,config_space) :
        self.curr_id = curr_pipeline_id
        self.op_class,self.config_sets = config_space
        self.op_completion_state = Signal.COMPLETE
        self.completion_state = Signal.INCOMPLETE
        self.params = None
        self.params_id = None
        self.curr_op = None
    def generate_config(self,score,evaluator) :
        if self.config_sets.completion_state == Signal.INCOMPLETE and self.op_completion_state == Signal.COMPLETE:
            self.params_id,self.params =  self.config_sets.get_next(score)
            if evaluator :
                evaluator.update_config(self.params)
            self.curr_op = self.op_class(**self.params)
            self.op_completion_state = Signal.INCOMPLETE
        return self.params
    def execute(self,inputs) :
        if self.op_completion_state == Signal.INCOMPLETE :
            self.op_completion_state,op_id,outputs = self.curr_op.execute(**inputs)
        self.completion_state = Signal.COMPLETE if self.config_sets.completion_state == self.op_completion_state and self.config_sets.completion_state == Signal.COMPLETE else Signal.INCOMPLETE 
        return self.curr_id+"__"+str(self.params_id)+"_"+op_id, outputs
        
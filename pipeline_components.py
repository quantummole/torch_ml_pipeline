# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:15:00 2018

@author: quantummole
"""

from signals import Signal
from copy import deepcopy

class PipelineBlock :
    def __init__(self,curr_pipeline_id,curr_chain,params_state,evaluator = None) :
        self.curr_pipeline_id = curr_pipeline_id
        self.op_set,input_names,self.score_combiner = curr_chain[0]
        self.next_block,output_names,_ = curr_chain[1] if len(curr_chain) > 1 else (None,None,None)
        self.remainig_chain = curr_chain[1:]
        self.params_state = params_state
        self.input_names = input_names
        self.output_names = output_names
        self.evaluator = evaluator
    def execute(self,inputs) :
        scores = []
        for op_name,op_class,config_set in self.op_set :
            config_space = (op_class,config_set)
            self.inputs = deepcopy(inputs)
            op = PipelineOp(self.curr_pipeline_id+"___"+op_name,config_space)
            curr_score = Signal.NO_SCORE
            op_scores = []
            op_inputs = dict([(name,inputs.get(name,None)) for name in self.input_names])
            while op.completion_state == Signal.INCOMPLETE :
                next_pipeline_id,outputs = op.execute(op_inputs,curr_score)
                new_params_state = deepcopy(self.params_state)
                new_params_state[op_name] = op.params
                if self.evaluator :
                    self.evaluator.log_config(next_pipeline_id,new_params_state)
                if self.next_block :
                    for i,name in enumerate(self.output_names) :
                        self.inputs[name] = outputs[i]
                    block = PipelineBlock(next_pipeline_id,self.remainig_chain,new_params_state)
                    curr_score = block.execute(self.inputs)
                    op_scores.append(curr_score)
                else :
                    op_scores.append(outputs)
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
    def execute(self,inputs,score) :
#        print(self.op_class.__class__.__name__,self.config_sets.completion_state,self.op_completion_state)
        if self.config_sets.completion_state == Signal.INCOMPLETE and self.op_completion_state == Signal.COMPLETE:
            self.params_id,self.params =  self.config_sets.get_next(score)
            self.curr_op = self.op_class(**self.params)
            self.op_completion_state = Signal.INCOMPLETE
        if self.op_completion_state == Signal.INCOMPLETE :
            self.op_completion_state,op_id,outputs = self.curr_op.execute(**inputs)
        self.completion_state = Signal.COMPLETE if self.config_sets.completion_state == self.op_completion_state and self.config_sets.completion_state == Signal.COMPLETE else Signal.INCOMPLETE 
        return self.curr_id+"__"+str(self.params_id)+"_"+op_id, outputs
        
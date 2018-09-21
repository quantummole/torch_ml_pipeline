# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:15:00 2018

@author: quantummole
"""

from signals import Signal
class PipelineBlock :
    def __init__(self,curr_pipeline_id,curr_chain,params_state) :
        self.curr_pipeline_id = curr_pipeline_id
        self.op_set,self.score_combiner = curr_chain[0]
        self.next_block,_ = curr_chain[1] if len(curr_chain) > 1 else (None,None)
        self.remainig_chain = curr_chain[1:]
        self.params_state = params_state
    def execute(self,inputs) :
        scores = []
        for op_class,config_sets in self.op_set :
            op = op_class(self.curr_pipeline_id,config_sets)
            curr_score = Signal.NO_SCORE
            op_scores = []
            while op.completion_state == Signal.INCOMPLETE :
                next_pipeline_id,outputs = op.execute(inputs,curr_score)
                new_params_state = self.params_state
                new_params_state[op.name] = op.params
                if self.next_block :
                    block = self.next_block(next_pipeline_id,self.remainig_chain,new_params_state)
                    curr_score = block.execute(outputs)
                    op_scores.append(curr_score)
            scores.append(op_scores)
        return self.score_combiner(scores)

class PipelineOp :
    def __init__(self,curr_pipeline_id,config_sets) :
        self.curr_id = curr_pipeline_id
        self.max_configs = []
        self.searchers = []
        self.op_classes = []
        for searcher_class,searcher_options,op_class,config_space in config_sets:
            searcher = searcher_class(config_space,**searcher_options)
            self.searchers(searcher)
            self.max_configs.append(searcher.max_configs)
            self.op_classes.append(op_class)
        self.search_completion_state = Signal.INCOMPLETE
        self.op_completion_state = Signal.INCOMPLETE
        self.completion_state = Signal.INCOMPLETE
        self.curr_op = None
    def execute(self,inputs,score) :
        if not score == Signal.NO_SCORE :
            self.searchers[0].tune(score)
        if self.op_completion_state == Signal.COMPLETE :
            config_id,params =  self.searchers[0]
            self.curr_params = params
            self.curr_op = self.op_classes[0](**self.curr_params)
            self.name = self.curr_op.name
            self.op_classes = self.op_classes[1:]
            self.max_configs[0] = self.max_configs[0] - 1
            if self.max_configs[0] == 0 :
                self.max_configs = self.max_configs[1:]
                self.searchers = self.searchers[1:]
            if not len(self.searchers) > 0 :
                self.search_completion_state = Signal.COMPLETE
        op_completion_signal,op_id,outputs = self.curr_op(inputs)
        self.op_completion_state = op_completion_signal
        self.completion_state = Signal.COMPLETE if self.search_completion_state == self.op_completion_state and self.search_completion_state == Signal.COMPLETE else Signal.INCOMPLETE 
        return self.curr_id+"__"+config_id+"_"+op_id, outputs
        
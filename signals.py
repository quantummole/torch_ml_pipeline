# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:18:26 2018

@author: quantummole
"""

from enum import Enum

class Signal(Enum) :
    INCOMPLETE = 0
    COMPLETE = 1
    NO_SCORE = 2
    
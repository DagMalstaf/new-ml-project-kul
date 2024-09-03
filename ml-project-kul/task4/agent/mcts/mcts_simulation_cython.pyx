# cython: language_level=3
# distutils: language = c++
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
import numpy as np


cdef extern from "mcts_simulation.cpp":
    cdef cppclass Node:
        Node(int num_actions)
        vector[int] legal_actions
        vector[double] Q
        vector[int] N
        vector[double] W
        void update_legal_actions(const vector[int]& actions)
        void update_statistics(int action, double reward)

    void simulate(Node* node, int num_simulations, double exploration_coefficient)

cdef class MCTSAgent:
    cdef Node* node
    cdef double exploration_coefficient

    def __cinit__(self, int num_actions, double exploration_coefficient):
        self.node = new Node(num_actions)
        self.exploration_coefficient = exploration_coefficient

    def __dealloc__(self):
        del self.node

    def run_simulations(self, int num_simulations):
        simulate(self.node, num_simulations, self.exploration_coefficient)

    def update_legal_actions(self, list legal_actions_py):
        cdef vector[int] legal_actions = vector[int]()
        for action in legal_actions_py:
            legal_actions.push_back(action)
        self.node.update_legal_actions(legal_actions)

    def update_tree(self, double reward):
        for action in self.node.legal_actions:
            self.node.update_statistics(action, reward) 

    def get_action(self):
        cdef int max_index = -1
        cdef double max_value = -float('inf')
        cdef int total_visits = sum(self.node.N) + 1  # Ensure non-zero denominator
        cdef double Q, U

        for i in range(self.node.legal_actions.size()):
            action = self.node.legal_actions[i]
            Q = self.node.Q[action]
            if self.node.N[action] > 0:
                U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + self.node.N[action]))
            else:
                U = float('inf')  # Encourage exploration of unvisited nodes

            if Q + U > max_value:
                max_value = Q + U
                max_index = action

        if max_index == -1:
            raise ValueError("No valid action found. Possible sync issue with legal actions.")
        return max_index, max_value


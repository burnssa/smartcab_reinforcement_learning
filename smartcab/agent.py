import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pdb
import math

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_table = {}
        self.states = []
        # self.previous_steps_forward = 0
        # self.previous_directional_turns = 0
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state + add other variables to inputs
        state = inputs
        #set id for state do make more tractable
        if state not in self.states:
            self.states.append(inputs)

        state_index = self.states.index(state)
        if state_index not in self.q_table.keys():
            self.q_table[state_index] = {}
            self.q_table[state_index]['action_reward'] = {}

        # TODO: Select action according to your policy
        action = self.choose_action(self.q_table, self.states, state)

        action = None if action == 'None' else action
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
    
        self.q_table[state_index]['action_reward'][action] = reward
        print self.q_table

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


    # def find_or_create_state(self, states, inputs):
    #     if inputs in states:
    #         state = states.index(inputs)
    #     else:
    #         states.append(inputs):
    #     return states

    def choose_action(self, q_table, states, state):
        #TODO: add proper random factor to prevent always responding to state in same way
        relevant_q_table = q_table[states.index(state)]
        if not relevant_q_table['action_reward']:
            action = self.random_action()
        else:
            rewards = relevant_q_table['action_reward'].values()
            best_reward = max(rewards)
            action = relevant_q_table['action_reward'].keys()[relevant_q_table['action_reward'].values().index(best_reward)]  #choosing first of best rewards - in case two actions return identical rewards
        return action

    def random_action(self):
        actions = ['None', 'forward', 'left', 'right']
        index = int(math.floor(random.random()*4))
        action = actions[index]
        return action



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pdb
import math

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    actions = ['None', 'forward', 'left', 'right']

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_table = {}
        self.states = []
        self.learning_rate = .4
        self.time = 0
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
        time = self.time + 1
        inputs['time'] = time

        # TODO: Update state + add other variables to inputs
        state = inputs
        #set id for state do make more tractable
        if state not in self.states:
            self.states.append(state)

        state_index = self.states.index(state)
        if state_index not in self.q_table.keys():
            self.q_table[state_index] = {}
            self.q_table[state_index]['action_reward'] = {}

        #Raw action implemented to address key issues in using None, while preserving valid cab actions
        raw_action = self.choose_action(self.q_table, self.states, state)

        action = None if raw_action == 'None' else raw_action
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        if self.q_table[state_index]['action_reward'].get(raw_action):
            reward = existing_reward + reward * math.pow(self.learning_rate, time)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def choose_action(self, q_table, states, state):
        #TODO: add proper random factor to prevent always responding to state in same way
        relevant_q_table = q_table[states.index(state)]
        if not relevant_q_table['action_reward']:
            action = self.random_action(self.actions)
        else:
            rewards = relevant_q_table['action_reward'].values()
            best_reward = max(rewards)
            action = relevant_q_table['action_reward'].keys()[relevant_q_table['action_reward'].values().index(best_reward)]  #choosing first of best rewards - in case two actions return identical rewards
            untried_actions =  list(set(self.actions) - set(relevant_q_table['action_reward'].keys()))
            if best_reward < 0 & len(untried_actions) > 0:
                action = self.random_action(untried_actions)
        return action

    def random_action(self, actions):
        index = int(math.floor(random.random()*len(actions)))
        action = self.actions[index]
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

    sim.run(n_trials=20)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

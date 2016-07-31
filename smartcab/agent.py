import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pdb
import math
import pprint

#Learning structure
DECISION_APPROACH = 'learning'
TRIALS = 100
MIN_REWARD = 0.0 #threshold for accepting best reward from previous action q table
DISCOUNT_FACTOR = 0.5
EXPLORATION_RATE = 0.1
DECAY_FACTOR = 1.0
LEARNING_RATE = 0.9

#Visual features
DISPLAY = True
UPDATE_DELAY = 0.00

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    actions = ['None', 'forward', 'left', 'right']

    def __init__(self, env, learning_rate, exploration_rate, decision_approach):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_table = {}
        self.states = []
        self.discount_factor = DISCOUNT_FACTOR
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.decision_approach = decision_approach
        self.time = 0
        self.trial = 0
        self.simulation_score = {}
        self.trial_time = {}
        self.trial_score = 0
        self.q_value = 0
        self.last_ten_trial_tables = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        print "Your score from the previous trial was {}".format(self.trial_score)
        self.simulation_score[self.trial] = self.trial_score
        self.trial_time[self.trial] = self.time
        self.trial_score = 0
        self.trial += 1

        print "Prepare for a new trip! This is trial {}".format(self.trial)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.time += 1

        waypoint = { 'waypoint': self.next_waypoint }
        inputs.update(waypoint) #if last_action is not None else inputs
        # inputs.update(before_last_action) #if before_last_action is not None else inputs
        state = inputs
        # set id for state to make more tractable
        if state not in self.states:
            self.states.append(state)

        self.state = tuple((inputs['light'],inputs['oncoming'],inputs['left'], inputs['right'], self.next_waypoint))

        state_index = self.states.index(state)
        if state_index not in self.q_table.keys():
            self.q_table[state_index] = {}
            self.q_table[state_index]['action_reward'] = {}

        #Add ability to toggle for fully random approach and compare with learning performance
        if self.decision_approach == 'random':
            raw_action = self.random_action(self.actions)
        else:
            raw_action = self.choose_action(self.q_table, self.states, state)

        #Raw action implemented to address key issues in using None, while preserving valid cab actions
        action = None if raw_action == 'None' else raw_action
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.trial_score += reward

        if self.decision_approach == 'learning':
            self.q_table = self.set_q_table(self.q_table, state_index, raw_action, reward)
            print self.q_table #Used for testing purposes
            print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
            if self.trial > (TRIALS - 5) and reward < 0.0:
                self.last_ten_trial_tables.append(tuple(({"Table Action-Rewards": self.q_table[state_index]['action_reward']},
                                                    {"Inputs: ": inputs},
                                                    {"Action: ": action},
                                                    {"Reward: ": reward})))


    def choose_action(self, q_table, states, state):
        random_selection_threshold = self.exploration_rate * math.pow(DECAY_FACTOR, self.time)
        new_random_action = random.random() < random_selection_threshold
        relevant_q_table = q_table[states.index(state)]
        #Select a random action if:
        #If there are no q-table entries for the state or
        #This action is selected to be randomized
        if not relevant_q_table['action_reward'] or new_random_action:
            action = self.random_action(self.actions)
        #Choose the action with the highest reward (above our MIN_REWARD value)
        else:
            rewards = relevant_q_table['action_reward'].values()
            best_reward = max(rewards)
            action = relevant_q_table['action_reward'].keys()[relevant_q_table['action_reward'].values().index(best_reward)]
            untried_actions = list(set(self.actions) - set(relevant_q_table['action_reward'].keys()))
            if best_reward <= MIN_REWARD and len(untried_actions) > 0:
                action = self.random_action(untried_actions)
        return action

    def random_action(self, actions):
        index = int(math.floor(random.random()*len(actions)))
        action = self.actions[index]
        return action

    def set_q_table(self, q_table, state_index, raw_action, reward):
        if self.q_table[state_index]['action_reward'].get(raw_action):
            previous_q_value = self.q_table[state_index]['action_reward'][raw_action]
            optimal_next_action = max([max(q['action_reward'].values()) for q in self.q_table.values()])
            learned_value = reward + self.discount_factor * optimal_next_action - previous_q_value
            previous_q_value += self.learning_rate * learned_value
        else:
            self.q_table[state_index]['action_reward'][raw_action] = reward
        return q_table

def run(trials, learning_rate, exploration_rate, decision_approach):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, learning_rate, exploration_rate, decision_approach)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=UPDATE_DELAY, display=DISPLAY)  # create simulator (uses pygame when display=True, if available)

    sim.run(n_trials=trials)  # run for a specified number of trials
    trial_scores = a.simulation_score.values()
    trial_times = a.trial_time.values()
    print "Your trial scores are {}".format(trial_scores)
    average_trial_score = sum(a.simulation_score.values()) / float(trials)
    print "###### Your average score per trial is {} ######".format(average_trial_score)
    print "###### Relevant tables and actions for final trials with suboptimal outcomes: #######"
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(a.last_ten_trial_tables)
    return trial_scores, trial_times



if __name__ == '__main__':
    run(trials=TRIALS, learning_rate=LEARNING_RATE, exploration_rate=EXPLORATION_RATE, decision_approach=DECISION_APPROACH)

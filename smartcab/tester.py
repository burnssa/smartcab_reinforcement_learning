import numpy as np
from agent import run
import matplotlib
import seaborn as sns
from seaborn import FacetGrid, lmplot
import pdb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from textwrap import wrap

TRIAL_SETS = 100

TRIALS = [20, 50, 200]
LEARNING_RATES = [0.5, 0.7, 0.9]
EXPLORATION_RATES = [0.1, 0.3, 0.5]
APPROACHES = ['learning', 'random']

class Tester():
    def __init__(self):
	   self.simulation_scores = []

    def run_trials(self, learning_rate, decision_approach, exploration_rate=0.2, trials=TRIALS[0]):
    	trial_scores, trial_times = run(trials, learning_rate, exploration_rate, decision_approach)
        trial_scores.pop(0), trial_times.pop(0) # Remove 0 for exponential fitting later
        trial_scores, trial_times = np.array(trial_scores), np.array(trial_times)
        total_score = np.sum(trial_scores)
        total_time = np.max(trial_times)

        coefficients = np.polyfit(np.log(trial_times), trial_scores, 1)
        score_improvement_rate = coefficients[0]

        fit = np.poly1d(coefficients)
        fit_projections = fit(np.log(trial_times))
        max_fit_score = np.max(fit_projections)
        print "##### Trial statistics #####"
        print "##### Decision approach: {} #####".format(decision_approach)
        print "##### Score improvement rate: {} #####".format(coefficients[0])
        print "##### Average score over time: {} #####".format(total_score/total_time)
        print "##### Maximum fit projection score: {} #####".format(max_fit_score)

        projected_data = pd.DataFrame({'trial_times':trial_times, 'projected_trial_scores':fit_projections})
        actual_data = pd.DataFrame({'trial_times':trial_times, 'trial_scores':trial_scores})

        return projected_data, actual_data, score_improvement_rate, max_fit_score

def run_trial_sets(learning_rate, decision_approach, exploration_rate, trials, dimensions, subplot):
    projection_list = []
    improvement_rates = []
    max_fit_scores = []
    height = 0

    if decision_approach == 'random':
        learning_rate = 'N/A'
        exploration_rate = 'N/A'
        height = 12

    for trial in range(0, TRIAL_SETS):
        t = Tester()
        projected, actual, improvement_rate, max_fit_score = t.run_trials(learning_rate,
                                                                            decision_approach,
                                                                            exploration_rate,
                                                                            trials)
        projection_list.append(projected)
        improvement_rates.append(improvement_rate)
        max_fit_scores.append(max_fit_score)

    mean_improvement_rate = np.mean(improvement_rates)
    mean_max_fit_score = np.mean(max_fit_scores)
    std_max_fit_score = np.std(max_fit_scores)

    print ""
    print "##### Summary trial set statistics #####"
    print "##### Average improvement rate: {} #####".format(mean_improvement_rate)
    print "##### Average maximum fit projection: {} #####".format(mean_max_fit_score)

    plt.subplot(dimensions[0], dimensions[1], subplot)
    for projection in projection_list:
        plt.plot(projection['trial_times'], projection['projected_trial_scores'], '-', lw=4, alpha=0.6)

    x_max = np.max(projection['trial_times'])
    plt.plot(actual['trial_times'], actual['trial_scores'], alpha=0.25, color='gray')
    plt.ylim([-5,30])
    plt.xlim([0, x_max])
    plt.xlabel('Smartcab time units (moves) per trial set', size=8)
    plt.ylabel('Trial scores', size=8)
    plt.text(9 * x_max / 20, 2 + height, 'Gray: actual scores from most recent trial set', style='italic', size=4)
    plt.text(9 * x_max / 20, 0 + height, 'Color: projected trial set averages', style='italic', size=4)
    plt.text(9 * x_max / 20, 4 + height, "Trial set mean improvement rate:{}".format(round(mean_improvement_rate,2)), size=7)
    plt.text(9 * x_max / 20, 7 + height, "Std dev. of trial set max projections:{}".format(round(std_max_fit_score,2)), size=7)
    plt.text(9 * x_max / 20, 10 + height, "Mean of trial set max projections:{}".format(round(mean_max_fit_score,2)), size=7)
    plt.title("\n".join(wrap("Learning rate: {} | " \
                            "Decision approach: {} | " \
                            "Exploration rate: {} | " \
                            "Trials: {}".format(learning_rate, decision_approach, exploration_rate, trials),50)), size=8)
    return plt

subplot = 0

font = {'family' : 'sans-serif',
        'size' : 6}

mpl.rc('font', **font)
fig = plt.figure(figsize=(12, 9), dpi=100)

for rate in EXPLORATION_RATES:
    dimensions = [len(EXPLORATION_RATES)+1, len(TRIALS)] #To include random row in subplots
    for trial in TRIALS:
        subplot += 1
        run_trial_sets(LEARNING_RATES[2], APPROACHES[0], rate, trial,
                            dimensions, subplot)
for trial in TRIALS:
    subplot += 1
    run_trial_sets(LEARNING_RATES[2], APPROACHES[1], EXPLORATION_RATES[0], trial,
                        dimensions, subplot) #Note: discount factors and exploration rates are not used
plt.suptitle('Learning performance under different exploration rates', size=12, y=0.97)
plt.tight_layout(h_pad=1.7, rect=[0, 0, 1, 0.93])
plt.savefig('exploration_rate_charts.png')
plt.show()

subplot = 0
fig = plt.figure(figsize=(12, 9), dpi=100)

for rate in LEARNING_RATES:
    dimensions = [len(LEARNING_RATES)+1, len(TRIALS)] #To include random row in subplots
    for trial in TRIALS:
        subplot += 1
        run_trial_sets(rate, APPROACHES[0], EXPLORATION_RATES[0], trial,
                            dimensions, subplot)
for trial in TRIALS:
    subplot += 1
    run_trial_sets(LEARNING_RATES[0], APPROACHES[1], EXPLORATION_RATES[0], trial,
                        dimensions, subplot) #Note: learning and exploration rates are not used
plt.suptitle('Learning performance under different learning rates', size=12, y=0.97)
plt.tight_layout(h_pad=1.7, rect=[0, 0, 1, 0.93])
plt.savefig('learning_rate_charts.png')
plt.show()

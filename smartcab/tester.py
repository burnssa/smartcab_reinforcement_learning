import numpy as np
from agent import run, DECISION_APPROACH, LEARNING_RATE, RANDOM_VARIATION_RATE
import matplotlib
import seaborn as sns
from seaborn import FacetGrid, lmplot
import pdb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from textwrap import wrap

TRIAL_SETS = 10

TRIALS = [50, 500]
LEARNING_RATES = [0.5, 0.7, 0.9]
RANDOM_VARIATION_RATES = [0.1, 0.3, 0.5]
APPROACHES = ['learning', 'random']


class Tester():
    def __init__(self):
	   self.simulation_scores = []

    def run_trials(self, learning_rate, decision_approach, random_variation_rate=0.2, trials=TRIALS[1]):
    	trial_scores, trial_times = run(trials, learning_rate, random_variation_rate, decision_approach)
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

def run_trial_sets(learning_rate, decision_approach, random_variation_rate, trials, dimensions, subplot):
    projection_list = []
    improvement_rates = []
    max_fit_scores = []

    for trial in range(0, TRIAL_SETS):
        t = Tester()
        projected, actual, improvement_rate, max_fit_score = t.run_trials(learning_rate, 
                                                                            decision_approach,
                                                                            random_variation_rate,
                                                                            trials)
        projection_list.append(projected)
        improvement_rates.append(improvement_rate)
        max_fit_scores.append(max_fit_score)

    mean_improvement_rate = np.mean(improvement_rates)
    mean_max_fit_score = np.mean(max_fit_score)

    print ""
    print "##### Summary trial set statistics #####"
    print "##### Average improvement rate: {} #####".format(mean_improvement_rate)
    print "##### Average maximum fit projection: {} #####".format(mean_max_fit_score)


    plt.subplot(dimensions[0], dimensions[1], subplot)
    for projection in projection_list:
        plt.plot(projection['trial_times'], projection['projected_trial_scores'], '-', lw=4, alpha=0.7)   

    x_max = np.max(projection['trial_times'])
    plt.plot(actual['trial_times'], actual['trial_scores'], alpha=0.3, color='gray')
    plt.ylim([-5,20])
    plt.xlim([0, x_max])
    plt.xlabel('Smartcab time units (moves) per trial set')
    plt.ylabel('Trial scores')
    plt.text(2 * x_max / 5, 13, 'Gray: actual scores from most recent trial set', style='italic')
    plt.text(2 * x_max / 5, 15, 'Color: projected trial set averages', style='italic')
    plt.text(2 * x_max / 5, 17, "Trial set mean improvement rate:{}".format(round(mean_improvement_rate,2)))
    plt.text(2 * x_max / 5, 19, "Trial set mean maximum fit projection:{}".format(round(mean_max_fit_score,2)))
    plt.title("\n".join(wrap(" Learning rate: {}," \
                            "Decision approach: {}, " \
                            "Random variation rate: {}, " \
                            "Trials: {}".format(learning_rate, decision_approach, random_variation_rate, trials),40)))
    
    return plt

subplot = 0
for trial in TRIALS:
    for rate in LEARNING_RATES:
        subplot += 1 
        dimensions = [len(TRIALS),len(LEARNING_RATES)]
        plt = run_trial_sets(rate, APPROACHES[0], RANDOM_VARIATION_RATES[0], trial, 
                            dimensions, subplot)
        plt.suptitle('Learning performance under varied parameters')
plt.show()



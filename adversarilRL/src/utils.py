import numpy as np
import matplotlib.pyplot as plt






prisoner_action_map = {
    0 : "Prisoner moves up 1 block", 
    1 : "Prisoner moves down 1 block", 
    2 : "Prisoner moves left 1 block", 
    3 : "Prisoner moves right 1 block", 
    4 : "Prisoner moves up 2 blocks", 
    5 : "Prisoner moves down 2 blocks", 
    6 : "Prisoner moves left 2 blocks", 
    7 : "Prisoner moves right 2 blocks", 
} 

helper_action_map = {
    0 : "Helper builds up 2 blocks", 
    1 : "Helper builds down 2 blocks", 
    2 : "Helper builds left 2 blocks", 
    3 : "Helper builds right 2 blocks", 
    4 : "Helper builds up 1 block away", 
    5 : "Helper builds down 1 block away", 
    6 : "Helper builds left 1 block away", 
    7 : "Helper builds right 1 block away", 
} 




def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

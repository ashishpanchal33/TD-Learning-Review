#!/usr/bin/env python
# coding: utf-8

# Note : Readers of this code are expected to have general understanding of Dynamic programming and Supervised learning.<br>
# Brief description of the thought process is shared in between, to align the readers, when possible.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#Declaring states:
# instead of using ABCDEFG, using 0123456,
# as it is easier to call the call the values.
states = [0,1,2,3,4,5,6]
non_terminal_states = [1,2,3,4,5]


#creating state vectors, with Xi = 1 for ith non terminal state.
initial_states = np.eye(len(states))
initial_weights  = np.ones(len(non_terminal_states))
initial_values = np.zeros(len(states))
initial_values[len(states)-1] = 1
initial_values[1:6] = initial_states[1:6,1:6].dot(initial_weights)


# Suttons uses this paper to expand on the effectiveness of the Temporal difference methods.
# which, as presented in his research, perform better than supervised learning techniques, in case of "Dynamic state systems"
# Additionally TD methods have an edge over Supervised methods in terms of learning at every step of experience and updating the expected prediction, rather than weighting till a experience terminal is reached. And therefore TD methods are proved to be more efficient in learning from the limited experience.
# 

# -------------------------

# # <u> Chapter 1. Function definitions</u>

# ## 1. Sequence generator.
# a single game sequence generator, to mimic the markovs general process, starting at the middle state "D" and taking right or left steps with 50% probability, until the game reaches the terminal state ( A or G)
# 
# however the code is made moldable to start from a different state.



def seq_generator(state = 3,sequence = np.array([])):
    
    #initiallize recording the sequence from the start state
    sequence = np.append(sequence,state)
        
    #generate the sequence until the terminal state is reached.
    while((state < 6) and (state > 0)):

        
        #move left or right from the current state with 50% probability
        state += np.random.choice([-1,1],p=[0.5,0.5])

        sequence = np.append(sequence,state)
        
    return(sequence)


# ## 2. Test sets generator.
# 
# Generates a collection of Test sets, as a record in a ragged arrays of different sequences per test set, using the sequence generator function.
# 

def Test_set_generator(state=3,seed =1,Set_count =10,sequence_count=5):
    Test_sets = []
    np.random.seed(seed)
        
    for i in range(Set_count):

        #creating a list of sequence
        sequence_list = []

        for j in range(sequence_count):

            #generating sequence
            sequence =seq_generator(state = state)
            sequence_list = sequence_list + [sequence]

        Test_sets = Test_sets + [sequence_list]
    return(Test_sets)
    


# ## 3. Delta W - generator:
# learning from a single sequence.


def Dw_generator(state = 3,sequence = np.array([]) ,values = initial_values, lambda_ = 0.1, alpha =0.1):
    

    e = 0
    dw_list  = np.array([])
    
    for n,i in enumerate(sequence[0:-1]):

        #setting State to int
        state = int(i)
        
        
        ################## calculating e: for t = 1, e1  == x1 , it is a vector of length 5
        #other wise et = lambda_*e(t-1) + xt
        e = lambda_*e + initial_states[state][1:6]


        
        ################## calculating dw, : dwt = alpha*(p(t+1) - p(t))*et
        
        #pt = wT.xt
        #p(t+1) = wT.x(t+1)
        # wT is the the same vector used to calculate the pt p(t+1) as mentioned by the paper
        #Addtionally since the value does not change before the test set ends, same values will be used without update for all the states and sequences in the test set.
        
        dw = alpha*( values[int(sequence[n+1])] - values[state] )*e
        

        if n == 0 :
            dw_list  = dw
        else:
            dw_list = np.vstack((dw_list,dw))
    return(dw_list.sum(axis =0))
    #return(dw_list)




# ## 4. Error calculation - RMSE from the ideal prediction

def rmse(arr_1,arr_2 =[1/6,1/3,1/2,2/3,5/6] ):
    #print(arr_1)
    return np.sqrt(((arr_1 - np.array(arr_2))**2).mean())


# ----------------------
#######################################3
# # Chapter 2: Experiment 1:
# learning TD lambda.
#########################################
# In the experiment 1, 
# 
# Sutton shows us the how TD method could be more efficient learners as comparied to Supervised learning.
# And this experiment also depicts the impact of the repeated presentation, of a set of sequences of the game, to learn from a limited experience of the observed universe.
# 
# **why does TD methods perform?**
# Sutton mentions, according to (Widrow & Stearns, 1985), the Widrow-Hoff procedure under repeated presentations, minimizes the RMS error between its predictions and the actual outcomes in the training set. 
# 
# However, "the Widrow-Hoff procedure only minimizes error on the training set, it does not necessarily minimize error for future experience."
# 
# Where as linear TD(0), converges to considered "the optimal estimates for matching future experience" - those consistent with the maximum-likelihood estimate of the underlying Markov process.
# 
# 
# --------------------------------------
# **Repeated for clarity:**
# In the experiment we staudy the impact of different horizons of experience on prediction performance, with different values of "Lambda".
# 
# --------------------------------------
# 
# Additionally, he has accepted the inherent stochasticity in the game, due to the random walk, which can result in vastly different experiences, and therefore showes biased results based on chance.
# 
# to answer this he has averaged the errors, to showcase a general behaviour of the game, over multiple different set of game experiences.

# ## 1. Procedure definition

# The procedure is trying peform the followings:
# 
# - for Different Lambda values
# - over 100 test sets.
#     - learn from the the experience of 10 sequences, where learning is accumilated till the end of a set, and update the expected predictions.
#         - Continue to learn from the current training set, till there is no more statistically significant learning observed, with repreated presentation of the training set.
#     - calculate the RSME error of expected prediction, updated with learning from the set and the ideal target values.
# - average the RMSE error over all the training sets
# - store the RMSE average error for different lambda values and compare


def experiment_1(wt_random = False,Test_Sets =np.array([]),initial_weights = initial_weights,seed =1,state = 3, Set_count =10,sequence_count=5, alpha = 0.2,variable_convo = False , wt_conf = 0.5, lambda_ = [0,0.1,0.3,0.5,0.7,0.9,1], convergence=0.001 ):

    if(len(Test_Sets) ==0):
        Test_Sets = Test_set_generator(state=state, seed =seed,
                                   Set_count = Set_count,
                                   sequence_count=sequence_count)        
    
    
    lambda_error_list =[]
    
    for n,gi in enumerate(lambda_):
        
        error_list = []
        
        for q,i in enumerate(Test_Sets):
            track = 0
            dw_update = 1
            
            #
            
            #as mentioned in the paper, the convergence is achiveable with any initial weights for a given small learning rate.
            
            values = initial_values.copy()
            
            if (wt_random):
                weights = np.random.rand(1, 5).reshape(5)
                
            else:
                weights = initial_weights.copy()
                weights = weights*wt_conf
            values[1:6] = initial_states[1:6,1:6].dot(weights)
            
            
            
            ####looping till the change in weights from iterating on the test set is less than 0.0001
            while( True): 

                
                
                #initializing : list of all the dw of 1 test set and all sequence in the test set
                dw_list  = np.array([])
                
                
                
                ####looping on the sequencese of the current test set
                for w,k in enumerate(i):
                    
                    
                    #####calculating the dw for the current sequence
                    
                    dw_sum =Dw_generator(state = 3,sequence = k,values = values, lambda_ = gi, alpha =alpha)
                    
                    
                    #append the sum of dw for the episode in the list of dws for the test set.
                    if(w == 0):
                        dw_list = dw_sum.copy()
                    else :
                        #dw_list = np.vstack((dw_list,dw_sum))
                        dw_list += dw_sum
                
                
                #updating weights after the test sets
                
                #new_weights = np.vstack((weights,dw_list.mean(axis=0))).sum(axis=0)
                dw_update = dw_list
                
                
                
                
                
                #setting convergence critera.
                #the amount of change which can imported is a function of lambda and alpha, and therefore a dynamic critrea of convergence can be created, wrt the minmax change
                if(variable_convo):
                    if(gi*alpha ==0):
                        convergence =convergence
                    else:
                        convergence = gi*alpha*0.5
                    
                
                #checking for convergence, if converged, then break
                if(abs(max(dw_update)) < convergence):
                    #print()
                    break
                
                
                #updating weights and calculating new values
                weights += dw_update
                values[1:6] = initial_states[1:6,1:6].dot(weights)
            
            
            #accumiliated error after learning from 10 sequences
            #why calculating error over values rather than weight?
            #Values are the representation of expected value of the target given the current state. 
            #for this experiment the feature vector defining the state is simply equals to 1 at the state's position in the experiment.
            #however this need not be true in the real world and the state will be defined on multiple different features.
            error_list.append(rmse(values[1:6]))
            
        # average of error of experiencing 10 sequences, over 100 training sets
        lambda_error_list.append(np.mean(error_list))
    return lambda_error_list


# ## 2. Creating Common Test Sets

# A common list of test set is created, before hand rather than creating the test sets on the go:
# 
# **Reason:**
# - to compare the different performance of different lamda and learning rate.

#create test sets
Test_Sets = Test_set_generator(state=3, seed =1,
                                   Set_count = 100,
                                   sequence_count=10)





###############################################
# ## 3. Simulating Game Experiment 1:##########
# #############################################
# **Replication of the Suttons 88, figure 3.**


exp1_errors_list = experiment_1(wt_random = False,Test_Sets = Test_Sets,state = 3, 
                    Set_count =100,sequence_count=10, alpha = 0.01,
                    variable_convo =False,convergence=0.000005, lambda_ = [0,0.1,0.3,0.5,0.7,0.9,1] )

exp1_lambda_errors = pd.DataFrame(data = exp1_errors_list, index = [0,0.1,0.3,0.5,0.7,0.9,1], columns =  ["rmse"])


###############################################
# ## Experiment 1 : Figure 3##########
# #############################################


ax_exp1 = exp1_lambda_errors.plot(kind='line', style ='-',color="green",marker ="o",fontsize=14,
                                  markerfacecolor='yellow', legend=False,figsize=(5,5))

ax_exp1.set_title('Figure 3 replications \n')
ax_exp1.set_xlabel("Lambda", fontsize=14)
ax_exp1.set_ylabel("RMSE Error", fontsize=14)

plt.yticks(np.arange(np.round(min(exp1_errors_list),2)-0.01,np.round(max(exp1_errors_list),2)+0.02,0.01))
plt.xticks(np.arange(0,1.1, 0.1))
plt.show()
ax_exp1.get_figure().savefig('fig_3_replications.png')



# **Observations**
# 1. Based on the methodology shared in the paper by Sutton, error calulated with local simulation , post learning over 10 randomly generated sequences, averaged over 100 test sets, with repeated presentation, is lower than what is shared on the paper, on average by 0.06 -0.07.
# 2. However the behaviour of the procedure is similar to Sutton's results, as dipected in Figure 3 -erratum, in the paper, that is performance of smaller TD values is better than the larger TD values and subsiquently Supervised Learning algorithms.
#     - Therefore, the semulated results support Sutton's statement that TD(Lambda) methods learn more efficiently as compared to Supervised learning methods, as they are moving towards a ideal perediction value
# 
# 
# ![Figure%203.PNG](attachment:Figure%203.PNG)
# 
# 
# **Hypothesis**:
# However, there are some assumptions which can be put to question, as possible reason for the observed deviation.
# 1. Enough number of Training sets to represent the idea of convergence
# 2. small learning rate : vagueness in description , how much is small?
# 3. sequence in set of random states
# 4. convergence criteria : how much is a significant change?
# 5. number of sequences per set.
# 6. Weight initialisation : stated any initial weight will converege.
# 
# 
# To study this further, extended learning procedures were performed, As detailed in the annxure.










# ###################################

# # Chapter 3 : Experiment 2 
########################################
def experiment_2(Test_Sets =np.array([]),seed =1,state = 3, Set_count =10,sequence_count=5, alpha = 0.2,values = initial_values,initial_states=initial_states, initial_weights=initial_weights, wt_conf = 0.5, lambda_ = [0,0.1,0.3,0.5,0.7,0.9,1], convergence=0.001 ):


    
    #values_lambda_ = np.empty([len(lambda_),len(states)])
    if(len(Test_Sets) ==0):
        Test_Sets = Test_set_generator(state=state, seed =seed,
                                   Set_count = Set_count,
                                   sequence_count=sequence_count)   
    
    lambda_error_list =[]
    
    for n,gi in enumerate(lambda_):

        error_list = []
        
        for q,i in enumerate(Test_Sets):
            track = 0
            dw_update = 1
            
            weights = initial_weights.copy()
            weights = weights*wt_conf
            values[1:6] = initial_states[1:6,1:6].dot(weights)
            
            #looping till the change in weights from iterating on the test set is less than 0.0001

                #initializing : list of all the dw of 1 test set and all sequence in the test set
            dw_list  = np.array([])

            #looping on the sequencese of the current test set
            for w,k in enumerate(i):
                #print(k)
                #calculating the dw for the current sequence
                ##************* should this is individual dw per state or a sum .. **** as to achive MC, it should be the sum of increments of dw

                dw_sum =Dw_generator(state = 3,sequence = k,values = values, lambda_ = gi, alpha =alpha)

                #updating weights and calculating new values
                weights += dw_sum
                values[1:6] = initial_states[1:6,1:6].dot(weights)
                
                error_list.append(rmse(values[1:6]))
        # averaged measure over 100 training sets
        lambda_error_list.append(np.mean(error_list))
    return lambda_error_list




###############################################
# ## 3. Simulating Game Experiment 2:##########
# #############################################

error_alpha_lamda_list  = [experiment_2(Test_Sets=Test_Sets,state = 3, Set_count =100,sequence_count=10, alpha = i, lambda_ = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] ) for i in np.arange(0,1.1,0.05)]

ex2_error = pd.DataFrame(data = error_alpha_lamda_list, index = np.arange(0,1.1,0.05),  columns =  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

ex2_error_fig_4  = ex2_error[[0,0.3,0.8,1]]
ex2_error_fig_4[ex2_error_fig_4 > 0.65] = np.nan

ex2_error_fig_4.columns =["λ ="+str(i) for i in ex2_error_fig_4.columns]


###############################################
# ## Experiment 2 : Figure 4##########
# #############################################


ax_exp2 = ex2_error_fig_4.plot(kind='line', style ='-',marker ="o",fontsize=14,figsize=(5,5))

ax_exp2.set_title('Figure 4 Replication\n')
ax_exp2.set_xlabel("Alpha - α",fontsize=14)
ax_exp2.set_ylabel("Error (RMSE)",fontsize=14)
ax_exp2.legend(fontsize=10)
#ax_exp2.set_ylim(0,0.8)
#ax_exp2.set_xlim(0,0.65)

ax_exp2.autoscale_view()

plt.ylim(-0.05, 0.75)
plt.xlim(-0.05, 0.65)
plt.yticks(np.arange(0,0.8,0.1))
#plt.xticks(np.arange(0,0.65, 0.05))
plt.show()
ax_exp2.get_figure().savefig('fig_4_replications.png')





###############################################
# ## Experiment 2 : Figure 5##########
# #############################################



t = ex2_error.apply(lambda x : x.idxmin(), axis =0).reset_index()
t.columns =["lambda","alpha"]
t["error"]= 0

for i in t.to_records():
    t["error"].loc[i[0]] = ex2_error.loc[i[2],i[1]]


ax_fig5 = t.plot(x = "lambda", y = 'error', kind='line', style ='-',color="green",marker ="o",
                                  markerfacecolor='yellow',fontsize=14,legend=False,figsize=(5,5))

ax_fig5.set_title('Figure 5 replication\n')
ax_fig5.set_xlabel("Lambda - λ",fontsize=14)
ax_fig5.set_ylabel("Error for best α",fontsize=14)

plt.yticks(np.arange(np.round(min(t['error']),2)-0.01,np.round(max(t['error']),2)+0.02,0.01))
#plt.ax_fig5(np.arange(0,1.1, 0.1))
plt.show()
ax_fig5.get_figure().savefig('fig_5_replications.png')




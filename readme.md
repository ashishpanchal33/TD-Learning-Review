Project 1 readme

Libraries to be installed before running the code: 
1. numpy 
2. pandas
3. matplotlib.pyplot

In the script initial weights and stages are defined. Post which below functions are defined. 

1. seq_generator(state = 3,sequence = np.array([]))
: a single game sequence generator, to mimic the markovs general process, starting at the middle state "D" and taking right or left steps with 50% probability, until the game reaches the terminal state ( A or G),
however the code is made moldable to start from a different state., 
: where "state" argument is the starting state,between 1 and 5, sequence argument need not be provided.
: random walk sequence is generated, as per the definition in Sutton's orignal text, and returned  as a numpy 1D array.

2. Test_set_generator(state=3,seed =1,Set_count =10,sequence_count=5)
: Generates a collection of Test sets, as a record in a ragged arrays of different sequences per test set, using the sequence generator function.
: Inputs : 
-State: starting state - int - between 1 and 5, 
-Seed : - int - random seed definition,  
-Set_count : - int -number of test sets to be generated
-sequence_count: -int - number of sequences per test set.
-Output : Ragged 3D list of test sets of sequences.


3. Dw_generator(state = 3,sequence = np.array([]) ,values = initial_values, lambda_ = 0.1, alpha =0.1),
: learning module, Calculates dW for a sequence.
Input : 
-state : starting state - int - between 1 and 5,
-sequence : 1D numpy array -int-
-values : 1D numpy array with 7 elements - float
- lambda_ : -float- between 0 and 1
- alpha : -float- learning rate
Output: 1D 5 element dW array - float- 

4. rmse(arr_1,arr_2 =[1/6,1/3,1/2,2/3,5/6] )
calulates RMSE error for a set of predictions from ideal prediction of the game
Input:
- arr_1 :  1D 5 element dW array - float-
return : - int - RMSE error

5. experiment_1(wt_random = False,Test_Sets =np.array([]),initial_weights = initial_weights,seed =1,state = 3, Set_count =10,sequence_count=5, alpha = 0.2,variable_convo = False , wt_conf = 0.5, lambda_ = [0,0.1,0.3,0.5,0.7,0.9,1], convergence=0.001 ):

Inputs required: 
-wt_random, : -Boolean- : if True , the randomize weights for each training set itteration. else initial_weights and wt_conf are used to calculate weights
-initial_weights, : 1D 5 element array, will be ignored if wt_random = False, based weights for computation
-wt_conf,: - float - , weight multiplier to adjust the initial weights.
-Test_Sets, : ragged 3D array of test sets and sequences. - int-, between 0-6if not provided, the code will generate the test set with the configurations shared: seed, state,Set_count, sequence_count. if given, then the cnfigurations will be ignored.
-seed, -int- random number generator key
-state, - int - between 1 and 5, starting state, when creating test set.
-Set_count, - -int- ,number of Set, to be generated, for averaging the error
-sequence_count, - number of sequences to be generated, - int -
-alpha, - learning rate, -float, between 0-1
-lambda_, - list of lambda, - float-, between 0-1
-variable_convo, - Boolean, if true, a dynamic convergence criteria will be generated, else convergence critera will be used as shared
-convergence - convergence threshold, -float-, will be ignored it variable_convo is True.
Output : 1D array containing Avg RMSE errors for the given configurations, for the number of lambda values, share in sequence of the lambda input. - float



6. experiment_2(Test_Sets =np.array([]),seed =1,state = 3, Set_count =10,sequence_count=5, alpha = 0.2, initial_weights=initial_weights, wt_conf = 0.5, lambda_ = [0,0.1,0.3,0.5,0.7,0.9,1], convergence=0.001 ):


Inputs required: 
-Test_Sets,
-seed,
-state, 
-Set_count,
-sequence_count, 
-alpha,
-wt_conf,
initial_weights,
-lambda_, 
-convergence

-initial_weights, : 1D 5 element array, will be ignored if wt_random = False, based weights for computation
-wt_conf,: - float - , weight multiplier to adjust the initial weights.
-Test_Sets, : ragged 3D array of test sets and sequences. - int-, between 0-6if not provided, the code will generate the test set with the configurations shared: seed, state,Set_count, sequence_count. if given, then the cnfigurations will be ignored.
-seed, -int- random number generator key
-state, - int - between 1 and 5, starting state, when creating test set.
-Set_count, - -int- ,number of Set, to be generated, for averaging the error
-sequence_count, - number of sequences to be generated, - int -
-alpha, - learning rate, -float, between 0-1
-lambda_, - list of lambda, - float-, between 0-1
-convergence - not used.
Output : 1D array containing Avg RMSE errors for the given configurations, for the number of lambda values, share in sequence of the lambda input. - float

To replicate Experiment 2,  function 10 has to be looped over different alpha values.



#Global variables:
-states = [0,1,2,3,4,5,6] : a list of all possible states
-non_terminal_states = [1,2,3,4,5] : a list of non terminal states 
-initial_states = np.eye(len(states)) : state vectors, with Xi = 1 for ith non terminal state, - 2D array with size "length of states" x "length of state"
- initial_weights  = np.ones(len(non_terminal_states)) : initial template of weights, for non terminal states, 1D array with size = 5
- initial_values : initial values of the states : 1 D array with size = 7







#how to run:
run the code from any python ide or scripting prompt:
for running through CMD or Anaconda type : python {file_path}\\Project_1_RL_TD_learning_Apanchal33.py
- the code will create and save 3 PNG files, correponding to the semulation of Figure 3,4 and 5, in the current directory of the python file. as fig_3_replications.png, fig_4_replications.png, fig_5_replications.png

additionally, Ipython notebook is also shared containing the code, additionally containing exploratory analysis and results. (Project_1_RL_TD_learning_Apanchal33.ipynb)
- Please run the all the code cells in sequence, from the 1st cell - till the end of Chapter 3 : Experiment 2.
- While running on the code on the Jupyter, you will also be able to visualize the charts on the go, additionally, Sutton's orignal charts are saved for reference.

Modular running
for only running Experiment 1:
- import all libraries.
- initiallize all the global parameters ( can change is required)
- initize and define all the functions.
- run the all the code snippets in the chapter 2. including defining test_sets if required.
	- self defined parameters can be passed in the function call under Chapter 2: Experiment 1: 3. Simulating Game Experiment.


for only running Experiment 2:
- import all libraries.
- initiallize all the global parameters ( can change is required)
- initize and define all the functions.
- run the all the code snippets in the chapter 3. and test_sets definition in chapter 2.
	- self defined parameters can be passed in the function call under Chapter 3: Experiment 2: 3. Simulating Game Experiment.
	- please note figure 4 is created based on desired lambda values, which are filterd out in the same snippet as the function call, additionally list of apha and lambda can alpha be modified in the same snippit.

#please note, no error/exeption handliong techniques are employed in the code, if the parameters are not used as mentioned, the executor will be requried to handle the error in case of exploration and code change. 




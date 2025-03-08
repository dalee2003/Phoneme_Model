import math
import copy

# create the HMM number models

# Inputs:
# phonemodel (aka NumberModel)  - the phone state definitions for our words/commands (dictionary)...from NumberModels.py
# avephonetimes  - the ave length of each phone in seconds (dictionary)...from PhoneInfo.py
# phonepriors    - probability of each phone (dictionary)...from PhoneInfo.py
# hoptime        - the frame hope time in seconds

# Outputs:
# model          - the built HMM models
def CreateHMMModels(phonemodel, avephonetimes, phonepriors, hoptime):
    model = {}  #make dictionary called model
    # loop through all the models
    for m in phonemodel: # for each state
        model[m] = {}                       # create a dictionary for each state, ZERO, ONE, TWO, etc.
        model[m]['states'] = phonemodel[m]  # set 'states' to the phone definitions of this model (Ex. ONE = SIL, W, AH, N, SIL)
        
        # self transition of each state is p = 1 -(1/L) where L is the expected length of the phoneme (in frames)
        # so p = 1 - (1/(avephonetimes/hoptime)) where avephonetimes/hoptime converts from time to frames
        model[m]['A'] = []
        model[m]['prior'] = []
        for phone in model[m]['states']:
            p = 1.0 - (1.0/(avephonetimes[phone] / hoptime))
            model[m]['A'].append([math.log10(p), math.log10(1.0-p)])  # save it in the log domain (as value are quite small in decimal)
            model[m]['prior'].append(math.log10(phonepriors[phone]))  # save it in the log domain
    
    # return the models that were built
    return model




# create the memory needed and initialized it to initial values
# memory is updated every frame
# Inputs
# model - the set of HMM models
#
# Outputs
# mem   - the initialized memory required for the set of models
def CreateAndInitModelMemory(model):
    min_v = -1e+6
    mem = {}  # create a dictionary
    for m in model:
        mem[m] = {}          # create a dictionary for each model
        nstates = len(model[m]['states'])  # number of states in model m
        mem[m]['delta'] = [min_v for s in range(nstates)]  # init delta vector
        mem[m]['delta'][0] = 0.0  # start state prob = 1.0 => log(1.0) = 0.0
        mem[m]['mlstate'] = 0

    # return the mem that was just initialized
    return mem


# perform a single update of the models based on the phone probabilities from the CNN
# for this current frame/hop.
#Thus, this is frame by frame, whether you want to compute a CNN Vprob and send it in to HMM right away,
#or wait for the entire CNN to finish and then send the Vprob in one by one, is up to design choice. 


# Inputs
#    model      - the set of HMM models
#    mem        - the current memory for the set of models
#    phoneprobs - the current frame/hop phoneme probabilities from the CNN
#    phonendx   - the ordered index of phones into the prob vector

def HMMrec(model, mem, phoneprobs, phonendx):
    decision = 0
    ID = -1
    maxmaxdelta = -99999999.0
    maxmaxID = -1

    # loop through all the models and update the delta with Viterbi
    for m in model:
        Viterbi(model[m], mem[m], phoneprobs, phonendx)
        # update the most likely state in the model
        mem[m]['mlstate'] = 0
        maxdelta = mem[m]['delta'][0]
        for s in range(1, len(mem[m]['delta'])):
            if mem[m]['delta'][s] > maxdelta:
                maxdelta = mem[m]['delta'][s]
                mem[m]['mlstate'] = s
        # update the global max
        if maxdelta > maxmaxdelta:
            maxmaxdelta = maxdelta
            maxmaxID = m
    # make a decision
    # if the model with the best delta is also in the 2nd to last or last state
    if mem[maxmaxID]['mlstate'] >= len(model[maxmaxID]['states'])-2:
        ID = maxmaxID
        decision = 1

    return decision, ID

def ComputeObservationProb(model_i, staten, phoneprobs, phonendx):
    # pull the phoneme for the current state from the CNN vector of probs
    B = phoneprobs[phonendx[model_i['states'][staten]]]
    if B <= 0 : 
        B = 0.0000000001 
    #if the probability vector from CNN is negative => set to very small value
    #float32 for B so min is 10^-45 and max is 10^38 so it is not due to truncation error
    logB = math.log10(B)   # do math in the log domain to maintain precision
    return logB

# perform Viterbi algorithm to evaluate the i_th model and update the state memories
# based on the prior memories and the current phoneme probabilities from the CNN

def Viterbi(model_i, mem_i, phoneprobs, phonendx):
    # make of copy of delta since we need to use the original during computations
    delta = copy.deepcopy(mem_i['delta'])
    nstates = len(model_i['states'])
    B = [0 for s in range(nstates)]  # initialize the vector to hold B (observation probs)
    # compute the observation prob for each state in the model
    for i in range(nstates):
        B[i] = ComputeObservationProb(model_i, i, phoneprobs, phonendx)

    # do the first state separately
    delta[0] = mem_i['delta'][0] + B[0] + model_i['A'][0][0]    # ['A'][0] is self transition

    # do the remaining states
    for i in range(1, nstates):
        tmp1 = mem_i['delta'][i] + model_i['A'][i][0]   # self to self
        tmp2 = mem_i['delta'][i-1] + model_i['A'][i-1][1] # transition from last state
        delta[i] = max(tmp1, tmp2) + B[i] - model_i['prior'][i]

    # update the memory with the new deltas
    mem_i['delta'] = delta

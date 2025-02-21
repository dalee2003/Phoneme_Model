# Define the HMM models for the digits that are going to be recognized
# Each HMM begins and ends with silence, with each state in between
# defined by each phoneme in the target word

NumberModel = {
    "ZERO" : ['SIL','Z', 'IH', 'R', 'OW', 'SIL'],
    "ONE" : ['SIL','W', 'AH', 'N','SIL'],
    "TWO" : ['SIL', 'T', 'UW', 'SIL'],
    "THREE" : ['SIL', 'TH', 'R', 'IY', 'SIL'],
    "FOUR" : ['SIL', 'F', 'AA', 'R', 'SIL'],
    "FIVE" : ['SIL', 'F', 'AY', 'V', 'SIL'],
    "SIX" : ['SIL', 'S', 'IH', 'K', 'S', 'SIL'],
    "SEVEN" : ['SIL', 'S', 'EH', 'V', 'AH', 'N','SIL'],
    "EIGHT" : ['SIL', 'EY', 'T', 'SIL'],
    "NINE" : ['SIL', 'N', 'AY','N','SIL']
}

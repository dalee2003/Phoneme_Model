# Define the HMM models for the digits that are going to be recognized
# Each HMM begins and ends with silence, with each state in between
# defined by each phoneme in the target word

NumberModel = {
    "ZERO" : ['H#','Z', 'IH', 'R', 'OW', 'H#'],
    "ONE" : ['H#','W', 'AH', 'N','H#'],
    "TWO" : ['H#', 'T', 'UW', 'H#'],
    "THREE" : ['H#', 'TH', 'R', 'IY', 'H#'],
    "FOUR" : ['H#', 'F', 'AA', 'R', 'H#'],
    "FIVE" : ['H#', 'F', 'AY', 'V', 'H#'],
    "SIX" : ['H#', 'S', 'IH', 'K', 'S', 'H#'],
    "SEVEN" : ['H#', 'S', 'EH', 'V', 'AH', 'N','H#'],
    "EIGHT" : ['H#', 'EY', 'T', 'H#'],
    "NINE" : ['H#', 'N', 'AY','N','H#']
}

# the phone token list
phonetokens = ['AA', 'AE', 'AH', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'DX', 'EH', 'ER', 'EY', 'F', 'G', 'H#', 'HH', 
               'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z']

# get the numbered index into the token list so that it can be accessed later
# Inputs
# ptokens   - the phoneme list

# Outputs
# pindex    - dictionary containing each token and corresponding numbered index

def GetPhoneIndex(ptokens):
    pindex = dict()
    for i in range(len(ptokens)): #for entire phoneme list...0 to length-1 
        pindex[ptokens[i]] = i   #first phoneme in list gets assigned to 0...etc.

    return pindex

#Because some of these values are so small we will log all of them to increase the magnitude. 
#These values were the output of the "phone.analysis.py" script from the "PHN_AVG_LENGTH" project. 

avephonetimes = {
    'AA' : 0.12311517321785476,
    'AE' : 0.14821226857643233,
    'AH' : 0.06229741893180734,
    'AW' : 0.16076183127572016,
    'AY' : 0.14399084728033473,
    'B'  : 0.017448188904172397,
    'CH' : 0.08631265206812652,
    'D'  : 0.023008860625704623,
    'DX' : 0.02869592100406054,
    'DH' : 0.03763749557678697,
    'EH' : 0.09120258564754737,
    'ER' : 0.09489842747111682,
    'EY' : 0.12679995617879053,
    'F'  : 0.10300829196750902,
    'G'  : 0.029823035448686166,
    'HH' : 0.06691973590715301,
    'IH' : 0.06155950120499525,
    'IY' : 0.09042247950524952,
    'JH' : 0.05829528535980149,
    'K'  : 0.05136392080426754,
    'L'  : 0.06445693868483413,
    'M'  : 0.061794092997268436,
    'N'  : 0.05362948670394887,
    'NG' : 0.06225324378654971,
    'OW' : 0.1258175620318352,
    'OY' : 0.16104788011695909,
    'P'  : 0.0442730148763524,
    'R'  : 0.06060033070805934,
    'S'  : 0.11294208193979934,
    'SH' : 0.11323940456257849,
    'T'  : 0.04807778127864345,
    'TH' : 0.09154560585885485,
    'UH' : 0.07623364485981309,
    'UW' : 0.1074459754364596,
    'V'  : 0.060325068956870606,
    'W'  : 0.0666061703821656,
    'Y'  : 0.0660622084548105,
    'Z'  : 0.08355908759607739,
    'H#': 0.5
}

#phone probabilities a priors
phonepriors = {
    'AA' : 0.04252034305220144,
    'AE' : 0.028306764020594462,
    'AH' : 0.04455287777171873,
    'AW' : 0.005162779827624059,
    'AY' : 0.01692598599179904,
    'B'  : 0.015445847467830004,
    'CH' : 0.005821406060777745,
    'D'  : 0.025126944894938493,
    'DX' : 0.019185144791541257,
    'DH' : 0.020013739084863636,
    'EH' : 0.027286955659582304,
    'ER' : 0.038618159670828524,
    'EY' : 0.016161129721039922,
    'F'  : 0.015693717555576015,
    'G'  : 0.014284399056677267,
    'HH' : 0.014950107292337981,
    'IH' : 0.09697386032874655,
    'IY' : 0.04924116343137185,
    'JH' : 0.008562141030997925,
    'K'  : 0.0345176802192588,
    'L'  : 0.04781768092745905,
    'M'  : 0.028519224095805328,
    'N'  : 0.06205250596658711,
    'NG' : 0.009688179429615518,
    'OW' : 0.015127157355013703,
    'OY' : 0.004844089714807759,
    'P'  : 0.01832822248819076,
    'R'  : 0.046309214393461896,
    'S'  : 0.05293796874004093,
    'SH' : 0.016918903989292012,
    'T'  : 0.030905858940674064,
    'TH' : 0.005318583882778695,
    'UH' : 0.003788871341260455,
    'UW' : 0.01744297217481215,
    'V'  : 0.014121512999015602,
    'W'  : 0.022237487872070708,
    'Y'  : 0.012145634299554542,
    'Z'  : 0.026720395459019993,
    'H#': 1     #log(1) = 0 so we set to 1 so we don't worry about it
}

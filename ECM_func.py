import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math
from math import sqrt
from kneed import KneeLocator
from numpy import  log10
from scipy import optimize
import scipy.signal as signal
import  cmath
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
import impedance


def download_EIS_data():
# store EIS data with frequency, real part and imaginary part
    EIS_path = 'EIS_data/'
    EIS_files = ['EIS_state_V_25C01.txt', 'EIS_state_V_25C02.txt', 'EIS_state_V_25C03.txt','EIS_state_V_25C04.txt','EIS_state_V_25C05.txt','EIS_state_V_25C06.txt','EIS_state_V_25C07.txt','EIS_state_V_25C08.txt','EIS_state_V_35C01.txt','EIS_state_V_35C02.txt', 'EIS_state_V_45C01.txt', 'EIS_state_V_45C02.txt']
    EIS = []
    for filename in EIS_files:
        EIS.append(pd.DataFrame(np.loadtxt(EIS_path + filename,comments='t',delimiter='\t')))

    for i in range(len(EIS)):
        EIS[i] = EIS[i].iloc[:,2:5]
        EIS[i].columns = ['freq','re','im']

    EIS_long = []
    for i in range(len(EIS)):
        EIS_long.append(int(EIS[i].shape[0]/60))
        
    return EIS, EIS_long






def  download_Capacity_data():
    
    # download Capacity data
    Capacity_path = 'F:/nature/Capacity data/'
    C_files = ['Data_Capacity_25C01.txt','Data_Capacity_25C02.txt','Data_Capacity_25C03.txt','Data_Capacity_25C04.txt','Data_Capacity_25C05.txt','Data_Capacity_25C06.txt','Data_Capacity_25C07.txt','Data_Capacity_25C08.txt', 'Data_Capacity_35C01.txt', 'Data_Capacity_35C02.txt', 'Data_Capacity_45C01.txt' , 'Data_Capacity_45C02.txt']

    Capacity_ = []
    for filename in C_files:
        Capacity_.append(pd.DataFrame(np.loadtxt(Capacity_path + filename,comments='t',delimiter='\t')))

    # store Capacity data with Capacity
    Capacity = []
    path = 'F:/nature/Capacity/'
    for filename in C_files:
        Capacity.append(pd.DataFrame(np.loadtxt(path + filename,comments='t',delimiter='\t')))

    # Aligning EIS data with capacity length since the length of EIS data is not totally equal to the length of capacity data
    Capacity_long = []
    for i in range(len(Capacity)):
        Capacity_long.append(len(Capacity[i]))
    
    return Capacity, Capacity_long





def EIS_fit_test(E):
    E = pd.DataFrame(E)



    # plot EIS curve, choose EIS_state_V_25C06 as sample

    # select a sample to plot Nyquist plot of EIS curve
    if (E.iloc[0, 2] > 0):
        R = E.iloc[0, 2]
    else:
        print('error')

    p = np.array(E.iloc[:, 2])

    print(signal.argrelextrema(-p, np.greater)[0])
    print(signal.argrelextrema(p, np.greater)[0])

    if bool(list(signal.argrelextrema(-p, np.greater)[0])):
        mini = signal.argrelextrema(-p, np.greater)[0]
    if bool(list(signal.argrelextrema(p, np.greater)[0])):
        circlep = signal.argrelextrema(p, np.greater)[0]

    x1 = list(E.iloc[:, 1])
    y = list(E.iloc[:, 2])

    kneedle_con_inc = KneeLocator(x1,
    y,
    curve = 'concave',
    direction = 'increasing',
    online = True)
    if (not list(kneedle_con_inc.all_knees)[0] is None):
        infp = x1.index(list(kneedle_con_inc.all_knees)[0])


    outputpath = ''
    A = E.iloc[:, :]
    A = pd.DataFrame(A)
    A.iloc[:, 2] = A['im'].map(lambda x: -x)

    A.to_csv(outputpath + 'simulate_data.csv', index=False, header=False)

    # use impedance package to fit
    frequencies, Z = preprocessing.readCSV('simulate_data.csv')
    frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)

    # fit the first half circle
    circuit1 = 'R0-p(R1,CPE1)'
    Rs = R
    R1 = R
    w1 = A.iloc[infp, 0]
    C1 = 1 / (R1 * w1)
    initial_guess1 = [Rs, R1, C1, 0.5]
    circuit1 = CustomCircuit(circuit1, initial_guess=initial_guess1)
    start = 0
    for j in range(60):
        if(A.iloc[j, 2]>0):
            continue
        else:
            start = j
            break
    frequencies1 = frequencies[:infp + 1 - start]
    Z1 = Z[:infp + 1 - start]

    circuit1.fit(frequencies1, Z1)
    Z_fit1 = circuit1.predict(frequencies1)
    print(circuit1)

    parameters_1 = circuit1.parameters_

    # fit the second half circle

    circuit2 = 'R0-p(R2,CPE2)'
    Rs = A.iloc[infp, 1]
    R2 = 0.5
    w2 = 10
    C2 = 1 / (R2 * w2)
    initial_guess2 = [Rs, R2, C2, 0.5]
    circuit2 = CustomCircuit(circuit2, initial_guess=initial_guess2)

    frequencies2 = frequencies[infp + 1 - start:]
    Z2 = Z[infp + 1 - start:]

    try:
        circuit2.fit(frequencies2, Z2)
    except:
        print('error')
    Z_fit2 = circuit2.predict(frequencies2)
    print(circuit2)

    parameters_2 = circuit2.parameters_

    # fit the warburg impedance
    Rs = 0.2
    circuit3 = 'R0-Ws1'
    initial_guess3 = [Rs, 1.5, 40]
    circuit3 = CustomCircuit(circuit3, initial_guess=initial_guess3)

    frequencies3 = frequencies[1 - start:]
    Z3 = Z[1 - start:]

    circuit3.fit(frequencies3, Z3)
    Z_fit3 = circuit3.predict(frequencies3)
    print(circuit3)

    parameters_3 = circuit3.parameters_

    # final fit
    circuit = 'R0-p(R1,CPE1)-p(R2-Ws1,CPE2)'

    initial_guess = [parameters_1[0], parameters_1[1], parameters_1[2], parameters_1[3], parameters_2[1], parameters_3[1],
                    parameters_3[2], parameters_2[2], parameters_2[3]]
    circuit = CustomCircuit(circuit, initial_guess=initial_guess)
    circuit.fit(frequencies, Z)
    Z_fit = circuit.predict(frequencies)
    print(circuit)
    parameters = list(circuit.parameters_)
    return parameters



def R_value(EIS):
    # initialize the R value of the first half circle
    R_sum = []

    for x in range(len(EIS)):
        r = []
        for i in range(int(EIS[x].shape[0]/60)):
            print(i)
            if(EIS[x].iloc[60*i,2] > 0):
                R = EIS[x].iloc[60*i,2]
            else:
                for j in range(60):
                    if(EIS[x].iloc[60*i+j,2] > 0):
                        id_1 = 60*i+j-1
                        id_2 = 60*i+j
                        break
                print(id_1)
                print(((-EIS[x].iloc[id_1,2])/(EIS[x].iloc[id_2,2]-EIS[x].iloc[id_1,2]))*(EIS[x].iloc[id_2,1]-EIS[x].iloc[id_1,1]))
                R = EIS[x].iloc[id_1,1] + ((-EIS[x].iloc[id_1,2])/(EIS[x].iloc[id_2,2]-EIS[x].iloc[id_1,2]))*(EIS[x].iloc[id_2,1]-EIS[x].iloc[id_1,1])
            r.append(R)
        R_sum.append(r)
    return R_sum







def mini_circle(EIS):

    # initialize the inflection point of the first half circle
    mini_sum = []
    circlep_sum = []
    for x in range(len(EIS)):
        mini = []
        circlep = []
        for i in range(int(EIS[x].shape[0]/60)):
            print(i)
            p = np.array(EIS[x].iloc[60*i:60*(i+1),2])

            print(signal.argrelextrema(-p, np.greater)[0])
            print(signal.argrelextrema(p, np.greater)[0])

            if bool(list(signal.argrelextrema(-p, np.greater)[0])):
                m = signal.argrelextrema(-p, np.greater)[0]
            if bool(list(signal.argrelextrema(p, np.greater)[0])):
                c = signal.argrelextrema(p, np.greater)[0]
            else:
                c = [circlep[-1]]
            mini.append(m[-1])
            circlep.append(c[-1])
        mini_sum.append(mini)
        circlep_sum.append(circlep)
    return mini_sum, circlep_sum







def infp_circle(EIS, R_sum):   
    
    infp_sum = []
    for x in range(len(EIS)):
        infp = []
        for i in range(int(EIS[x].shape[0]/60)):

            x1 = list(EIS[x].iloc[60*i:60*(i+1),1])
            y = list(EIS[x].iloc[60*i:60*(i+1),2])

            kneedle_con_inc = KneeLocator(x1,
                                            y,
                                            curve='concave',
                                            direction='increasing',
                                            online=True)
            if(not list(kneedle_con_inc.all_knees)[0] is None):
                infp.append(x1.index(list(kneedle_con_inc.all_knees)[0]))
            else:
                infp.append(infp[-1])
            plt.plot(list(kneedle_con_inc.all_knees)[0], list(kneedle_con_inc.all_knees_y)[0], 'ro')

        infp_sum.append(infp)

    return infp_sum





def EIS_curve_fit(folder_path):
    
    EIS, EIS_long = download_EIS_data()
    Capacity, Capacity_long = download_Capacity_data()
    R_sum = R_value(EIS)
    mini_sum, circlep_sum = mini_circle(EIS)
    infp_sum = infp_circle(EIS, R_sum)
    
    compare = pd.DataFrame([EIS_long,Capacity_long])
    compare = np.array(compare)
    compare = compare.T
    compare = pd.DataFrame(compare)
    compare.columns = ['EIS','Capacity']

    length = []
    for i in range(len(compare)):
        length.append(min(compare.iloc[i, :]))

    length = pd.DataFrame(length)
    length.to_csv('length.csv')
    length = list(length)


    er = []
    feature_sum = []
    feature_files = ['feature25c01.csv','feature25c02.csv','feature25c03.csv','feature25c04.csv','feature25c05.csv','feature25c06.csv','feature25c07.csv','feature25c08.csv','feature35c01.csv','feature35c02.csv','feature45c01.csv','feature45c02.csv']
    feature_path = 'feature/'
    outputpath = 'feature/'

    for x in range(8):
        feature = np.zeros((length[x], 9))
        for i in range(length[x]):

            A = EIS[x].iloc[60*i:60*(i+1) ,:]
            A = pd.DataFrame(A)
            A.iloc[:, 2] = A['im'].map(lambda x: -x)

            A.to_csv(outputpath+'simulate_data.csv',index=False,header=False)

            # use impedance package to fit
            frequencies, Z = preprocessing.readCSV('F:/nature/simulate_data.csv')
            frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)

            # fit the first half circle
            circuit1 = 'R0-p(R1,CPE1)'
            Rs=R_sum[x][i]
            R1 = A.iloc[infp_sum[x][i],1]-R_sum[x][i]
            w1 = A.iloc[infp_sum[x][i],0]
            C1 = 1/(R1*w1)
            initial_guess1 = [Rs, R1, C1, 0.5]
            circuit1 = CustomCircuit(circuit1, initial_guess=initial_guess1)
            start = 0
            for j in range(60):
                if(A.iloc[j, 2]>0):
                    continue
                else:
                    start = j
                    break

            frequencies1 = frequencies[:infp_sum[x][i]+1-start]
            Z1 = Z[:infp_sum[x][i]+1-start]

            circuit1.fit(frequencies1, Z1)
            Z_fit1 = circuit1.predict(frequencies1)
            print(circuit1)


            parameters_1 =  circuit1.parameters_

            # fit the second half circle

            circuit2 = 'R0-p(R2,CPE2)'
            Rs = A.iloc[infp_sum[x][i], 1]
            R2 = A.iloc[mini_sum[x][i],1]-A.iloc[infp_sum[x][i],1]
            w2 = A.iloc[circlep_sum[x][i],0]
            C2 = 1/(R2*w2)
            initial_guess2 = [Rs, R2, C2, 0.5]
            circuit2 = CustomCircuit(circuit2, initial_guess=initial_guess2)

            frequencies2 = frequencies[infp_sum[x][i]+1-start:mini_sum[x][i]+1-start]
            Z2 = Z[infp_sum[x][i]+1-start:mini_sum[x][i]+1-start]

            try:
                circuit2.fit(frequencies2, Z2)
            except:
                er.append([x,i])
                continue
            Z_fit2 = circuit2.predict(frequencies2)
            print(circuit2)


            parameters_2 =  circuit2.parameters_

            # fit the warburg impedance
            Rs = A.iloc[mini_sum[x][i], 1]
            circuit3 = 'R0-Ws1'
            initial_guess3 = [Rs, 1.5, 40]
            circuit3 = CustomCircuit(circuit3, initial_guess=initial_guess3)

            frequencies3 = frequencies[mini_sum[x][i]+1-start:]
            Z3 = Z[mini_sum[x][i]+1-start:]

            circuit3.fit(frequencies3, Z3)
            Z_fit3 = circuit3.predict(frequencies3)
            print(circuit3)


            parameters_3 = circuit3.parameters_

            # final fit
            circuit = 'R0-p(R1,CPE1)-p(R2-Ws1,CPE2)'

            initial_guess = [parameters_1[0], parameters_1[1], parameters_1[2], parameters_1[3], parameters_2[1], parameters_3[1], parameters_3[2], parameters_2[2],parameters_2[3]]
            circuit = CustomCircuit(circuit, initial_guess=initial_guess)
            circuit.fit(frequencies, Z)
            Z_fit = circuit.predict(frequencies)
            print(circuit)
            parameters = list(circuit.parameters_)

            for j in range(9):
                feature[i,j] = parameters[j]
            print(i)
        feature = pd.DataFrame(feature)
        feature.to_csv(folder_path+feature_files[x], index=False, header=False)
        feature_sum.append(feature)
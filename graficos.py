import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the data from the file
#"./perfisvelocidade/35Hz/x_D_1/MF1_Bomba_Grupo_G30P1_20221215.txt"
xd1 = ["./perfisvelocidade/35Hz/x_D_1/MF1_Bomba_Grupo_19_20230929.txt",
       "./perfisvelocidade/35Hz/x_D_1/MF1_Bomba_Grupo_21_20230929.txt",
       "./perfisvelocidade/35Hz/x_D_1/MF1_Bomba_Grupo_24_20230929.txt"]

xd3 = ['./perfisvelocidade/35Hz/x_D_3/MF1_Bomba_Grupo_32_20231004.txt',
       './perfisvelocidade/35Hz/x_D_3/MF1_Bomba_Grupo_38_20231004.txt',
       './perfisvelocidade/35Hz/x_D_3/MF1_Bomba_Grupo_40_20231004.txt']

xd6 = ['./perfisvelocidade/35Hz/x_D_6/MF1_Bomba_Grupo_20_20230929.txt',
       './perfisvelocidade/35Hz/x_D_6/MF1_Bomba_Grupo_29_20230929.txt']

rhoxd1 = 1.176
rhoxd3 = 1.176
rhoxd6 = 1.178
rho_ar = 1.1674

def getxyvals(xd):
    if xd == xd1:
        xdstr = 'x/D=1'
        grupos = ['G19', 'G21', 'G24']
    elif xd == xd3:
        xdstr = 'x/D=3'
        grupos = ['G32', 'G38', 'G40']
    elif xd == xd6:
        xdstr = 'x/D=6'
        grupos = ['G20', 'G29']

    # Loop through the files and extract x values
    x_values_list = []
    y_values = None
    for filename in xd:
        data = np.genfromtxt(filename, delimiter="\t", skip_header=1)
        x_values_list.append(data[:, -4])  # 3rd last column
        if y_values is None:
            y_values = data[:, 1]  # 2nd column (assuming it's the same in all files)

    # Calculate the average x values
    average_x_values = np.mean(x_values_list, axis=0)

    def plotavgs():
        # Create the plot
        plt.figure(figsize=(8, 6))
        for i, x_values in enumerate(x_values_list):
            plt.plot(x_values, y_values, marker='o', linestyle='-', label=grupos[i])
        plt.plot(average_x_values, y_values, marker='o', linestyle='-', label='AVG')
        plt.xlabel("V_c [m/s]")
        plt.ylabel("y [mm]")
        plt.title(xdstr)
        plt.grid(True)
        #plt.xlim(0, 8)
        plt.legend()
        plt.show()

    plotavgs()
    return list(average_x_values), list(y_values)


def m_ponto(average_x_values, y_values, rho):
    vinv = average_x_values
    #max_index = list(average_x_values).index(max(average_x_values))
    #vinv = average_x_values[:max_index + 1][::-1]
    #y = list(map(lambda x: x/1000, y_values[:max_index + 1]))
    #y = list(map(lambda x: x/1000, y_values))
    y = list(map(lambda x: (x-30)/1000, y_values))
    #print(f"integrou-se de 0 a {y[max_index]}")
    integral = np.trapz(list(map(lambda i, j: i * abs(j), vinv, y)), x=y)
    return integral * np.pi * rho


def balanco_movimento(average_x_values, y_values, rho):
    vinv = average_x_values
    y = list(map(lambda x: (x-30)/1000, y_values))
    #print(f"balanco_movimento: integrou-se de 0 a {y[max_index]}")
    integral = np.trapz(list(map(lambda i, j: i * abs(j) * i, vinv, y)), x=y)
    return integral * np.pi * rho

def balanco_energia(average_x_values, y_values, rho):
    vinv = average_x_values
    y = list(map(lambda x: (x-30)/1000, y_values))
    #print(f"balanco_movimento: integrou-se de 0 a {y[max_index]}")
    # só 1/2 v^2, sem u e sem gz
    integral = np.trapz(list(map(lambda i, j: i * abs(j) * (i * i / 2), vinv, y)), x=y)
    return integral * np.pi * rho


def plot(average_x_values, y_values, xdstr):
    # Create the plot
    plt.figure(figsize=(8, 6))
    #pontos da media
    plt.plot(average_x_values[:53], y_values[:53], marker='o', linestyle='-')
    #pontos extrapolados
    plt.plot(average_x_values[53:], y_values[53:], marker='o', linestyle='-')
    #linha a unir os ultimos 2
    # Plot the line using plt.plot()
    plt.plot([average_x_values[52], average_x_values[53]], [y_values[52], y_values[53]], marker='', linestyle='-', color='black')
    plt.xlabel("Average V_c [m/s]")
    plt.ylabel("y [mm]")
    plt.title("Perfil médio " + xdstr)
    plt.grid(True)
    #plt.xlim(0, 8)
    plt.show()


def extrapolarxy16(average_x_values, y_values):
    x_data = np.arange(30, 53, 1)  # Assuming you want to fit the function for x = [0, 0.1, 0.2, ..., 0.9]
    y_data = np.array(average_x_values[30:])

    x_data2 = np.arange(0, 30, 1)
    y_data2 = np.array(average_x_values[:30])

    def func(x, a, b, c):
        return a * x**2 + b * x*2 + c


    params, covariance = curve_fit(func, x_data, y_data)
    a, b, c = params[0], params[1], params[2]

    params2, covariance2 = curve_fit(func, x_data2, y_data2)
    a2, b2, c2 = params2[0], params2[1], params2[2]

    x_fit = np.linspace(0, 60, 100)  # Generating x values for the fitted curve
    y_fit = func(x_fit, *params2)
    y_fit2 = func(x_fit, *params)
    
    ymm = 53
    extrapolated_x = list(average_x_values).copy()
    new_y_vals = list(y_values).copy()
    while ymm <= 60:
        #extrapolated_x.append(func(ymm, a, b, c) * 0.5 + func((60-ymm), a2, b2, c2) * 0.0 + extrapolated_x[60-ymm] * 0.5)
        extrapolated_x.append(func(ymm, a, b, c) * 0.5 + extrapolated_x[60-ymm] * 0.5)
        new_y_vals.append(ymm)
        ymm += 1

    def plotextr():
        if max(extrapolated_x) > 6:
            xdstr = 'x/D=1'
        else:
            xdstr = 'x/D=6'
        
        #pontos da media
        plt.scatter(x_data2, y_data2, label='Data y=0 to y=30')
        plt.scatter(x_data, y_data, label='Data y=30 to y=52')
        plt.plot(x_fit, y_fit, label='Best Fit y=0 to y=30', color='red')
        plt.plot(x_fit, y_fit2, label='Best Fit y=30 to y=52', color='green')
        #pontos extrapolados
        plt.scatter(new_y_vals[53:], extrapolated_x[53:], label='Extrapolated Data y=53 to y=60')
        plt.title("Ajuste por extrapolação + simetria " + xdstr)
        plt.legend()
        plt.show()
        plot(extrapolated_x, new_y_vals, xdstr)

    #plotextr()
    return extrapolated_x, new_y_vals


avx1, y1 = getxyvals(xd1)
maxv1 = max(avx1)#7.106
extrapolated_x1, new_y_vals1 = extrapolarxy16(avx1, y1)

avx6, y6 = getxyvals(xd6)
maxv6 = max(avx6) #2.44
extrapolated_x6, new_y_vals6 = extrapolarxy16(avx6, y6)

v1plusv6 = maxv1 + maxv6
ratiov1 = maxv1/v1plusv6
ratiov6 = maxv6/v1plusv6

avx3, y3 = getxyvals(xd3)
extrapolated_x3 = list(map(lambda i, j: i*ratiov1 + j*ratiov6, extrapolated_x1[:19], extrapolated_x6[:19])) + avx3[19:53] + \
                list(map(lambda i, j: i*ratiov1 + j*ratiov6, extrapolated_x1[53:], extrapolated_x6[53:]))
new_y_vals3 = new_y_vals6.copy()



# Create the plot
plt.figure(figsize=(8, 6))

#plt.plot(avx1, y1, marker='o', linestyle='-', color='red', label='avg1')
#plt.plot(avx3, y3, marker='o', linestyle='-', color='green', label='avg3')
#plt.plot(avx6, y6, marker='o', linestyle='-', color='blue', label='avg6')

plt.plot(extrapolated_x1, new_y_vals1, marker='o', linestyle='-', color='darkred', label='x/D=1')
plt.plot(extrapolated_x3, new_y_vals1, marker='o', linestyle='-', color='darkgreen', label='x/D=3')
plt.plot(extrapolated_x6, new_y_vals1, marker='o', linestyle='-', color='darkblue', label='x/D=6')

#plt.plot(extrapolated_x3[:19], new_y_vals1[:19], marker='o', linestyle='--', color='darkgreen', label='Pontos aproximados')
#plt.plot(extrapolated_x3[53:], new_y_vals1[53:], marker='o', linestyle='--', color='darkgreen', label='Pontos aproximados')
#plt.plot(extrapolated_x3[19:53], new_y_vals1[19:53], marker='o', linestyle='--', color='darkred', label='Pontos da média')
#plt.plot([extrapolated_x3[18], extrapolated_x3[19]], [new_y_vals1[18], new_y_vals1[19]], marker='', linestyle='-', color='black')
#plt.plot([extrapolated_x3[52], extrapolated_x3[53]], [new_y_vals1[52], new_y_vals1[53]], marker='', linestyle='-', color='black')

plt.xlabel("V_c [m/s]")
plt.ylabel("y [mm]")
plt.title("Y vs. V_c")
plt.grid(True)
plt.xlim(0, 7.2)
plt.legend()
plt.show()


def average_of_squares(lst):
    return sum(list(map(lambda x: x*x, lst))) / len(lst)

def calcula_bernoulli(pressao, rho, vquadmed):
    g = 9.80665
    return pressao/(rho*g) + vquadmed/(2*g)


pressao1 = 1.022475820472e5
print("### x/D=1 ###")
print(f"m_ponto: {m_ponto(extrapolated_x1, new_y_vals3, rho_ar)}")
print(f"balanco_movimento: {balanco_movimento(extrapolated_x1, new_y_vals3, rho_ar)}")
print(f"balanco_energia: {balanco_energia(extrapolated_x1, new_y_vals3, rho_ar)}")
vquadmed1 = average_of_squares(extrapolated_x1)
print(f'media quadrados velocidades: {vquadmed1}')
print(f'max v: {max(extrapolated_x1)}, y: {extrapolated_x1.index(max(extrapolated_x1))}')
#print(f"grad: {np.gradient(new_y_vals3, extrapolated_x1).tolist()}")
print('### ### ###')

pressao3 = 1.0224971763775e5
print("### x/D=3 ###")
print(f"m_ponto: {m_ponto(extrapolated_x3, new_y_vals3, rho_ar)}")
print(f"balanco_movimento: {balanco_movimento(extrapolated_x3, new_y_vals3, rho_ar)}")
print(f"balanco_energia: {balanco_energia(extrapolated_x3, new_y_vals3, rho_ar)}")
vquadmed3 = average_of_squares(extrapolated_x3)
print(f'media quadrados velocidades: {vquadmed3}')
print(f'u3-u1 = {calcula_bernoulli(pressao3, rho_ar, vquadmed3) - calcula_bernoulli(pressao1, rho_ar, vquadmed1)}')
print(f'max v: {max(extrapolated_x3)}, y: {extrapolated_x3.index(max(extrapolated_x3))}')
#print(f"grad: {np.gradient(new_y_vals3, extrapolated_x3).tolist()}")
print('### ### ###')

pressao6 = 1.0226039559056e5
print("### x/D=6 ###")
print(f"m_ponto: {m_ponto(extrapolated_x6, new_y_vals3, rho_ar)}")
print(f"balanco_movimento: {balanco_movimento(extrapolated_x6, new_y_vals3, rho_ar)}")
print(f"balanco_energia: {balanco_energia(extrapolated_x6, new_y_vals3, rho_ar)}")
vquadmed6 = average_of_squares(extrapolated_x6)
print(f'media quadrados velocidades: {vquadmed6}')
print(f'u6-u3 = {calcula_bernoulli(pressao6, rho_ar, vquadmed6) - calcula_bernoulli(pressao3, rho_ar, vquadmed3)}')
print(f'max v: {max(extrapolated_x6)}, y: {extrapolated_x6.index(max(extrapolated_x6))}')
#print(f"grad: {np.gradient(new_y_vals3, extrapolated_x6).tolist()}")

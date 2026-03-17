import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D

def run_simulation(F, S, xcf,xB,xC,yB,yC):
    eq_interp = interp1d(xC, yC, kind='cubic', fill_value="extrapolate")

    def equilibrium(x):
        y = eq_interp(x)
        return np.clip(y,0,1)

    def cross_stage(vars, Rin, xin, S):

        xout, Rout, Eout = vars

        yout = equilibrium(xout)

        eq1 = Rin + S - Rout - Eout
        eq2 = Rin*xin - Rout*xout - Eout*yout
        eq3 = yout - equilibrium(xout)

        return [eq1, eq2, eq3]

    guess = [0.2, 900, 400]

    solution = fsolve(cross_stage, guess, args=(F,xcf,S))

    xout, Rout, Eout = solution

    yout = equilibrium(xout)

    def crosscurrent(N, xF, Rin, S):

        xin = xF

        for i in range(N):

            guess = [xin*0.7, Rin*0.9, S*0.5]

            xout, Rout, Eout = fsolve(
                cross_stage,
                guess,
                args=(Rin, xin, S)
            )

            Rin = Rout
            xin = xout

        removal = (xF - xout)/xF * 100

        return removal


    stages = np.arange(1,21)
    feed_conc = np.linspace(0.05,xcf,20)




    removal_matrix = np.zeros((len(feed_conc),len(stages)))

    for i,xF in enumerate(feed_conc):

        for j,N in enumerate(stages):

            removal_matrix[i,j] = crosscurrent(N,xF,F,S)


    X,Y = np.meshgrid(stages,feed_conc)
    Z = removal_matrix

    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121, projection='3d')

    surf1 = ax1.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='none'
    )

    ax1.set_title(f"Cross-current Extraction\nF={F}, S={S}, xcf={xcf}")
    ax1.set_xlabel("Stages")
    ax1.set_ylabel("Feed $x_C$")
    ax1.set_zlabel("Removal")

    plt.tight_layout()
    plt.show()
    #plot 1

    def countercurrent(vars, N, F, S, xF):

        x = vars[:N]
        y = vars[N:]

        eqs = []

        x_in = xF

        for i in range(N):

            xi = x[i]
            yi = y[i]

            if i < N-1:
                y_next = y[i+1]
            else:
                y_next = 0

            # solute balance
            eq1 = F*x_in + S*y_next - F*xi - S*yi

            # equilibrium relation
            eq2 = yi - equilibrium(xi)

            eqs.append(eq1)
            eqs.append(eq2)

            x_in = xi

        return eqs



    def solve_countercurrent(N, xF, F, S):
        def obj(vars):
        # Split the giant list of guesses into x values and y values
            x, y = vars[:N], vars[N:]
            eqs = []
            for i in range(N):
                # If stage 1, feed is xF. Otherwise, feed is from previous stage
                x_in = xF if i == 0 else x[i-1]
                
                # If last stage, fresh solvent y_in is 0. Otherwise, solvent comes from NEXT stage
                y_in = 0 if i == N-1 else y[i+1]
                
                # Append the equations for THIS stage to the massive list
                eqs.append(F*x_in + S*y_in - F*x[i] - S*y[i])
                eqs.append(y[i] - equilibrium(x[i]))
                
            return eqs

        guess_x = np.linspace(xF*0.8, xF*0.1, N)
        guess_y = equilibrium(guess_x)
        
        # fsolve solves the entire column at once
        sol = fsolve(obj, np.concatenate([guess_x, guess_y]))
        
        x_final = sol[N-1] 
        removal = (xF - x_final) / xF * 100
        return removal

       




    stages = np.arange(1,21)
    feed_conc = np.linspace(0.05,xcf,20)

    Z_counter = np.zeros((len(feed_conc),len(stages)))

    for i,xF in enumerate(feed_conc):

        for j,N in enumerate(stages):

            Z_counter[i,j] = solve_countercurrent(N,xF,F,S)



    X,Y = np.meshgrid(stages,feed_conc)



    fig = plt.figure(figsize=(8,6))

    ax = fig.add_subplot(111,projection='3d')

    surf = ax.plot_surface(
        X,
        Y,
        Z_counter,
        cmap='viridis',
        edgecolor='none'
    )

    ax.set_xlabel("Number of Stages")
    ax.set_ylabel("Feed Concentration $x_C$")
    ax.set_zlabel("Removal (%)")

    ax.set_title(f"Counter-current Extraction\nF={F}, S={S}, xcf={xcf}")

    ax.view_init(elev=25,azim=-60)

    fig.colorbar(surf, shrink=0.5, aspect=10, label="% Removal")

    plt.show()
    #plot 2


    import pandas as pd
    feed_conc = np.linspace(0.05,xcf,20)
    data = {
        "Feed Flowrate": [F]*len(feed_conc),
        "Solvent Flowrate": [S]*len(feed_conc),
        "xcf": feed_conc
    }

    for j, N in enumerate(stages):
        data[f"Stage_{N}"] = Z[:, j]

    df = pd.DataFrame(data)

    import os

    file = "crosscurrent_results.xlsx"

    if not os.path.exists(file):
        df.to_excel(file, index=False)
    else:
        with pd.ExcelWriter(file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df.to_excel(writer,
                        startrow=writer.sheets["Sheet1"].max_row,
                        index=False,
                        header=False)


    feed_conc=np.linspace(0.05,xcf,20)
    data2={
        "Feed Flowrate": [F]*len(feed_conc),
        "Solvent Flowrate": [S]*len(feed_conc),
        "xcf": feed_conc
    }

    for j, N in enumerate(stages):
        data2[f"Stage_{N}"] = Z_counter[:, j]
    df2 = pd.DataFrame(data2)


    file = "countercurrent_results.xlsx"

    if not os.path.exists(file):
        df2.to_excel(file, index=False)
    else:
        with pd.ExcelWriter(file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df2.to_excel(writer,
                        startrow=writer.sheets["Sheet1"].max_row,
                        index=False,
                        header=False)

F_values = [800, 1000, 1200, 1500]
S_values = [1000, 1300, 1600]
xcf_values = [0.25, 0.30, 0.35]

xA = [0.9845, 0.9545, 0.8580, 0.7570, 0.6780, 0.5500, 0.4290];
xB = [0.0155, 0.0170, 0.0250, 0.0380, 0.0600, 0.1220 ,0.2250];
xC = [0.0000, 0.0285, 0.1170, 0.2050, 0.2620, 0.3280, 0.3460];

# Extract phase (MIBK phase)
yA = [0.0212, 0.0280, 0.0540, 0.0920, 0.1450, 0.2200, 0.3100];
yB = [0.9788, 0.9533, 0.8570, 0.7350, 0.6090, 0.4720, 0.3540];
yC = [0.0000, 0.0187, 0.0890, 0.1730, 0.2460, 0.3080, 0.3360];


for F in F_values:
    for S in S_values:
        for xcf in xcf_values:


            run_simulation(F, S, xcf, xB, xC, yB, yC)

            print(f"Finished simulation: F={F}, S={S}, xcf={xcf}")


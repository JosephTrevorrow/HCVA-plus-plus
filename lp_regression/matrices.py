import numpy as np
import pandas as pd

def PMatrix(df_row, list_of_prefs):
    """Reconstructs the P matrix of the formalisation. The P matrix is the preference matrix."""
    """P-Matrix example:
    [[st, st], [st,co], [st,se], [st,op],
    [co, st], [co, co], [co, se], [co, op]]
    , etc. etc.,]] """
    # 1. Find the number of values (that becomes width/height of matrix)
    num_vals = 4 # TODO: Make this automatically generate: num of possible comparisons is n(n-1)/2
    p_matrix = np.zeros((num_vals, num_vals))
    # 2. Iterate over every row that starts with a "P__" (so it is a preference), and store appropriately.
    k = 1 # Set at 1 to avoid col 1: "country" # TODO: Debug to make sure this works
    for i in range(num_vals):
        for j in range(num_vals):
            p_matrix[i][j] = df_row[k]
            k+=1
    if __debug__:
        print("P-matrix: ", p_matrix, "\n")
    return p_matrix

def JMatrixs(df_row, list_of_actions):
    """This function reconstructs the J matrices. A J matrix is a matrix of action judgements
    INPUT: pd.DataFrame object
    RETURN: J+ and J- matrices. J+ is positives only, J- is negatives
    """
    """Note that:  n_val=J_list[0][0].shape[0]
                   n_actions=J_list[0][0].shape[1]"""
    """ Example J matrix: (1 row per value, actions in order)
    J_p = [
        [df_row['a_adp_rel'], df_row['a_div_rel']],
        [df_row['a_adp_nonrel'], df_row['a_div_nonrel']]
    ]
    J_n = [
        [-df_row['a_adp_rel'], -df_row['a_div_rel']],
        [-df_row['a_adp_nonrel'], -df_row['a_div_nonrel']]
    ]
    """
    size = len(list_of_actions)
    J_p = np.zeros((size, size))
    J_n = np.zeros((size, size))

    k = 0
    for i in range(size):
        for j in range(size):
            J_p[i][j] = df_row[k]
            J_n[i][j] = -df_row[k]
            k+=1
    if __debug__:
        print("J_p: ", J_p, "\nJ_n: ", J_n, "\n")
    return J_p, J_n

def Weights(df, n_countries, weights=0):
    """
    This function computes a weights vector. We do not consider weights in this work.
    # TODO: Check if this can be removed safely.
    """
    w = []
    n_total = 0
    # no weights, i.e. w = 1
    for i in range(n_countries):
        w.append(1)
    return np.array(w)

def FormalisationObjects(filename='value_systems.csv', delimiter=',', weights=0):
    """
    This function computes the matrices P, J+ and J-, and the weight vector of the formalisation.
    INPUT: filename -- str ; delimiter -- str ;
           weights -- int (weights' arg is always 0).
    RETURN: np.array with weights (array only contains 1's)
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    n_agents = df.shape[0]  # number of rows
    J_list = []
    P_list = []
    country_dict = {}
    list_of_prefs = [col for col in df.columns if 'P__' in col]
    # Note that this is a list of all matrices, not a sum of all matrices.
    for i in range(n_agents):
        country = df.iloc[i]['country']
        country_dict.update({i: country})
        P = PMatrix(df.iloc[i], list_of_prefs)
        try:
            # J_n is only used in the creation of a BVector.
            J_p, J_n = JMatrixs(df.iloc[i])
            J_list.append((J_p, J_n))
        except:
            # This will happen if you are not aggregating action judgements, just preferences.
            if __debug__:
                print("Could not find JMatrix, do you only have preference value_systems?")
            pass
        P_list.append(P)

    w = Weights(df, n_agents, weights)
    return P_list, J_list, w, country_dict

def Vectorisation(M):
    """
    This function vectorises any matrix.
    INPUT: M (matrix)
    RETURN: np.array shape = dim x dim
    """
    vector = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            vector.append(M[i][j])
    return vector


def BMatrix(w, n_val=2, n_actions=2, p=2):
    """
    This function computes the B matrix.
    INPUT: w (weights), n_val -- int (number of values), n_actions -- int (number of actions),
           p -- int
    RETURN: np.array shape = 2·n_val·n_actions·n_countres x 2·n_val·n_actions
    """
    I = np.identity(2 * n_val * n_actions)
    B = np.array((w[0] ** (1 / p)) * I)
    for i in range(1, len(w)):
        B = np.concatenate((B, (w[i] ** (1 / p)) * I))
    return B


def BVector(J_list, w, p=2):
    """
    This function computes the b vector.
    INPUT: J_list (list of J matrices), w (weights), p -- int
    RETURN: np.array shape = 2·n_val·n_actions·n_countres x 1
    """
    b = []
    for i in range(len(w)):
        j_p = Vectorisation(J_list[i][0])
        j_n = Vectorisation(J_list[i][1])
        for k in range(len(j_p)):
            b.append((w[i]**(1 / p)) * j_p[k])
        for k in range(len(j_n)):
            b.append((w[i]**(1 / p)) * j_n[k])
    return np.array(b)


def CMatrix(w, n_val=2, p=2):
    """
    This function computes the C matrix.
    INPUT: w (weights), n_val -- int (number of values), p -- int
    RETURN: np.array shape = n_val·n_val·n_countres x n_val·n_val
    """
    I = np.identity(n_val * n_val)
    C = np.array((w[0] ** (1 / p)) * I)
    for i in range(1, len(w)):
        C = np.concatenate((C, (w[i] ** (1 / p)) * I))
    return C


def CVector(P_list, w, p=2):
    """
    This function computes the c vector.
    INPUT: P_list (list of P matrices), w (weights), p -- int
    RETURN: np.array shape = n_val·n_val·n_countres x 1
    """
    c = []
    for i in range(len(w)):
        pref = Vectorisation(P_list[i])
        for k in range(len(pref)):
            c.append((w[i] ** (1 / p)) * pref[k])

    return np.array(c)


def FormalisationMatrix(P_list, J_list, w, p=2, v=True):
    """
    This function computes the A matrix and b vector of the lp-regression problem,
    i.e. minimizing ||Ax-b||_p problem.
    INPUT: P_list (list of P matrices), J_list (list of J matrices), w (weights),
           p -- int, v -- boolean (parameter, when v = True, we solve the preference aggregation
           over moral values, when v = False, we solve the aggregation of moral values)
    RETURN: A,b
    """
    if v:
        A = CMatrix(w, n_val=P_list[0][0].shape[0], p=p)
        b = CVector(P_list, w, p=p)
    else:
        A = BMatrix(w, n_val=J_list[0][0].shape[0],
                    n_actions=J_list[0][0].shape[1], p=p)
        b = BVector(J_list, w, p=p)
    """
    In this case, A and b correspond to the matrices described in Theorem 5.1 of the paper.
    A = $[w_1^{1/p}*I...w_N^{1/p}*I]$
    b = $[w_1^{1/p}*T_1...w_N^{1/p}*T_N]$
    """
    return A, b


if __name__ == '__main__':
    P_list, J_list, w, country_dict = FormalisationObjects()
    A, b = FormalisationMatrix(P_list, J_list, w)

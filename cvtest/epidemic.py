import numpy as np

def get_state(N, dates_save, day):
    state = np.zeros(N,dtype=np.int8)
    exp = (dates_save["date_exposed"] <= day)
    state[exp] = 1
    infect = (dates_save["date_infectious"]<=day)
    sympt = (dates_save["date_symptomatic"]<=day)
    nonsympt = infect & (~sympt)

    neversympt = np.isnan(dates_save["date_symptomatic"])
    asymp = (nonsympt & neversympt)
    presymp = nonsympt & (~neversympt)
    state[asymp] = 2
    state[presymp] = 3

    state[sympt] = 4
    severe = (dates_save["date_severe"] <= day)
    critical = (dates_save["date_critical"]<= day)
    state[severe] = 5
    state[critical] = 6

    state[dates_save["date_recovered"]<=day] = 7
    state[dates_save["date_dead"]<=day] = 8

    return state
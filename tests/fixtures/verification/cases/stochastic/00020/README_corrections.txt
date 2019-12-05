Corrections to The immigration-death process test '00020', '002-01'

Initial 00020 model
    Reactions:
        Immigration: -> X, @Alpha
        Death: X -> , @Mu

    Initial amounts (molecules):
        Species type X = 0

    Parameters:
        Alpha = 1
        Mu = 0.1

Since the initial population is 0, this must be modeled in a wc lang abstact compartment, which means V(t) = V(0) = V.

Because Immigration is order 0, its rate @ time=0 in wc sim is:
    = Alpha * V * NA
Set this equal to the 00020 model rate, and solve for V
    Alpha * V * NA = Alpha
    V = 1/NA

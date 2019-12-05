Corrections to Dimerization test '00030', '003-01'

Test 00030 assumes constant but unknown volume, doesn't specify specie type molecular weights and provides rate constants with units of 1/sec. But wc sim assumes density = mass/V, species types have molecular weights, and rate constants have units of M^(1-n)*(1/sec) for order n reactions.

Initial 00030 model
    Reactions:
        Dimerization: 2P -> P2, @k1 * P * (P - 1)) / 2
        Disassociation: P2 -> 2P, @k2 * P2

    Initial amounts (molecules):
        Species type P = 100
        Species type P2 = 0

    Parameters:
        k1 = 0.001
        k2 = 0.01

Corrected 00030 model:

Solve for V and species types molecular weights s.t.

    rate(Dimerization) @ time=0 = k1 * P(0) * (P(0) - 1)) / 2 = 4.95
    rate(Disassociation) @ time=0 = k2 * P2(0) = 0 [not helpful]

    Species type molecular weights for mass balance and constant volume:
        P(MW) = P2(MW)/2

In WC model:

    Assume given values of k1 & k2:
    
        rate(Dimerization) @ time=0 = k1 * P(0) * (P(0) - 1)) / (2 * V * NA) = 4.95
        rate(Disassociation) @ time=0 = 0

        solve rate(Dimerization) for V
            P(0) = 100
            k1 * P(0) * (P(0) - 1)) / (2 * V * NA) = 4.95
            -> 4.95 * 1/(V * NA) = 4.95
            -> V(cell, 0) = V(cell) = 1/NA

        Determine molecular weights:
            By reaction mass conservation, 
            mass(cell, 0) = mass(cell) = 100*P(MW)
            V(cell, 0) = mass(cell)/density(0)
            density(0) = mass(cell)/V(cell) = 100*P(MW)/(1/NA) = 100 * NA * P(MW)
            -> molecular weights are underconstrained
            -> just select MWs that satisfy P(MW) = P2(MW)/2

#!/usr/bin/env python3
"""
Implementation of a genetic algorithm minimizing search
"""
import numpy as np

def genetic(func, x0, lowerbounds, upperbounds):
    """Implementation of a genetic algorithm minimizing search
    
    Using a genetic algorithm, searches for the minimizing set of variables for
    an objective multivariable function given a set of upper and lower
    boundaries on those variables and an initial guess.
    
    Practical Genetic Algorithms by Haupt & Haupt (2003)
    
    Args:
        func: The multivariable function to minimize.
        x0: A list of values for the variables to be used as an initial guess
            at the minimizing values.
        lowerbounds: A list of lower bounds on each variable of the function.
        upperbounds: A list of upper bounds on each variable of the function.
    
    Returns:
        A tuple containing the minimizing values of each variable and the value
        of the function i.e. (xmin, func(xmin)).
    """

    # Number of variables in the multivariate function
    Nvars = len(x0)

    # Stopping criteria
    generations = 1000 # Max number of iterations

    Npop = 12 # Set population size
    mutrate = 0.5 # Set mutation rate

    selection = 0.5 # Set fraction of population that survives per generation
    keep = np.floor(selection * Npop) # Number of surviving members

    # The best members are more likely to crossover than other members
    prob = np.arange(1, keep+1) / np.sum(np.arange(1, keep+1))
    odds = np.cumsum([0.0] + list(reversed(prob)))

    # Mutation
    nmut = np.ceil((Npop - 1) * Nvars * mutrate)
    Nmatings = np.ceil((Npop - keep) / 2.0)


    #Initialize the population
    generation = 0 # Generation counter

    # Generate a random population within the variable bounds
    population = np.array([[upperbounds[ivar] +
                            (lowerbounds[ivar]-upperbounds[ivar])*member[ivar]
                            for ivar in range(Nvars)]
                           for member in np.random.rand(Npop, Nvars)])
    # Include the initial guess
    population[0][:] = x0

    # Evaluate the population's initial fitness
    fitness = np.zeros(Npop)
    for index in range(Npop):
        fitness[index] = func(population[index][:])

    # Sort population members by fitness
    sortorder = np.argsort(fitness)
    population = population[sortorder][:] # The best member is now first

    # Preallocate
    ma = np.ones(3, dtype='int')
    pa = np.ones(3, dtype='int')

    while generation < generations:
        generation += 1 # Increment the generation counter

        # Reproduction / Crossover
        # Choose parents
        parent1 = np.random.rand(Nmatings)
        parent2 = np.random.rand(Nmatings)

        # ma and pa represent the indicies of genes that will crossover
        ic = 1
        while ic <= Nmatings:
            for idd in np.arange(2, keep+2):
                if (parent1[ic-1] <= odds[idd-1] and
                    parent1[ic-1] > odds[idd-2]):
                    ma[ic-1] = idd-2

                if (parent2[ic-1] <= odds[idd-1] and
                    parent2[ic-1] > odds[idd-2]):
                    pa[ic-1] = idd-2

            ic += 1
        # end while

        # Single point crossover
        ix = np.arange(1, keep+1, 2) - 1 # index of parent 1
        xp = np.ceil(np.random.rand(Nmatings) * Nvars) - 1 # crossover point
        r = np.random.rand(Nmatings) # mixing parameter
        for ic in range(int(Nmatings)):
            xy = population[ma[ic]][xp[ic]] - population[pa[ic]][xp[ic]] # ma and pa mate
            population[keep+ix[ic]][:] = population[ma[ic]][:] # 1st offspring
            population[keep+ix[ic]+1][:] = population[pa[ic]][:] # 2nd offspring
            population[keep+ix[ic]][xp[ic]] = population[ma[ic]][xp[ic]] - r[ic]*xy
            # 1st
            population[keep+ix[ic]+1][xp[ic]]=population[pa[ic]][xp[ic]] + r[ic]*xy
            # 2nd
            if xp[ic] < Nvars: # crossover when last variable not selected
                population[keep+ix[ic]][:] = np.concatenate([population[keep+ix[ic]][0:xp[ic]],
                                                             population[keep+ix[ic]+1][xp[ic] + 0:Nvars]])
                population[keep+ix[ic]+1][:] = np.concatenate([population[keep+ix[ic]+1][0:xp[ic]],
                                                               population[keep+ix[ic]][xp[ic] + 0:Nvars]])
            # end if
        # end for

        # Mutation
        mrow = np.sort(np.ceil(np.random.rand(nmut) * (Npop-1)))
        mcol = np.ceil(np.random.rand(nmut) * Nvars) - 1

        for ii in range(int(nmut)):
            mutant = population[mrow[ii]][:]

            varhi = upperbounds[mcol[ii]]
            varlo = lowerbounds[mcol[ii]]

            # Small chance for a fully random mutation
            if np.random.rand() < (mutrate / 3.0):
                # Make completely random choice within variable bounds
                mutant[mcol[ii]] = ((varhi - varlo)*np.random.rand()) + varlo
            else:
                # Normally distributed random change
                mutant[mcol[ii]] = mutant[mcol[ii]] * (1 + np.random.randn())
                # Ensure the values are within the variable bounds
                if mutant[mcol[ii]] > varhi:
                    mutant[mcol[ii]] = varhi
                elif mutant[mcol[ii]] < varlo:
                    mutant[mcol[ii]] = varlo

            population[mrow[ii]][:] = mutant

        # Re-evaluate population fitness
        for index in range(Npop):
            fitness[index] = func(population[index][:])

        # Sort population members by fitness
        sortorder = np.argsort(fitness)
        population = population[sortorder][:] # The best member is now first
    # end while

    xmin = population[0][:]
    fval = func(xmin)

    return (xmin, fval)

# Example
if __name__ == '__main__':
    from time import clock
    testfunction = lambda x: (x[0]**2) + (x[1]**4) + (x[2]**6) + (x[3]**8)
    xguess = np.array([3.0, 3.0, 3.0, 3.0])
    LB = np.array([-5.0, -5.0, -5.0, -5.0])
    UB = np.array([5.0, 5.0, 5.0, 5.0])
    start = clock()
    print(genetic(testfunction, xguess, LB, UB))
    print(str(clock() - start) + ' s')

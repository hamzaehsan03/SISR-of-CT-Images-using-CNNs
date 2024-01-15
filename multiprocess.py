from multiprocessing import Pool

def parallel_process(function, arguments, processors=None):

    '''
    generic parallelisation function
    function = function to be processed in parallel
    arguments = argument tuples
    processors = number of parallel processes to use
    '''
    
    with Pool(processes=processors) as pool:
        results = pool.starmap(function, arguments)
    return results

from multiprocessing import Pool

def parallel_process(function, arguments, processors=None):

    with Pool(processes=processors) as pool:
        results = pool.starmap(function, arguments)
    return results

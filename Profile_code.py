import cProfile
import pstats

# Profile the FrozenLakeSW.py script
def profile_frozenlake():
    # Create a profiler
    profiler = cProfile.Profile()
    
    # Run the FrozenLakeSW.py script under the profiler
    profiler.run('exec(open("FrozenLakeSW.py").read())')
    
    # Create a Stats object to sort and display profiling results
    stats = pstats.Stats(profiler)
    stats.strip_dirs()  # Remove extraneous path info
    stats.sort_stats(pstats.SortKey.TIME)  # Sort by time spent in functions
    
    # Save the profiling results to a file
    stats.dump_stats("FrozenLakeSW.prof")
    print("Profiling results saved to FrozenLakeSW.prof")

if __name__ == "__main__":
    profile_frozenlake()
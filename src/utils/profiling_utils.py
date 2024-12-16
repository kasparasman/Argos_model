import time
import cProfile
import pstats
import torch
import logging
import os
from functools import wraps

class SadTalkerProfiler:
    def __init__(self, log_dir='./profiling_logs'):
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'profiling.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SadTalker_Profile')
        self.events = {}
        
    def start_event(self, name):
        """Start timing an event"""
        self.events[name] = {
            'start': time.perf_counter(),
            'cuda_start': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        if self.events[name]['cuda_start']:
            self.events[name]['cuda_start'].record()
            
    def end_event(self, name):
        """End timing an event and log results"""
        if name not in self.events:
            return
        
        cpu_time = time.perf_counter() - self.events[name]['start']
        
        cuda_time = None
        if self.events[name]['cuda_start'] and torch.cuda.is_available():
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_end.record()
            torch.cuda.synchronize()
            cuda_time = self.events[name]['cuda_start'].elapsed_time(cuda_end) / 1000
            
        self.logger.debug(f"{name} - CPU Time: {cpu_time:.3f}s" + 
                         (f", CUDA Time: {cuda_time:.3f}s" if cuda_time is not None else ""))
        
        # Log memory stats if CUDA is available
        if torch.cuda.is_available():
            self.logger.debug(f"{name} - GPU Memory: "
                            f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB, "
                            f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        
        return cpu_time, cuda_time

def profile_generator(func):
    """Decorator for profiling generator functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats('./profiling_logs/generator_profile.stats')
        return result
    return wrapper
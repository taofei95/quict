import multiprocessing

from .redis_controller import RedisController


class MasterNodeManager:
    def __init__(self):
        # Node API connection
        # Node list
        pass

    def start_all_processor(self):
        # Start Pending Job Processor
        
        # Start Running Job Processor
        
        # Start Killed Job Processor
    
        # Start Cluster Status Processor
        pass


class PendingJobProcessor(multiprocessing.Process):
    def __init__(self):
        super().__init__()

    def start(self):
        pass


class RunningJobProcessor(multiprocessing.Process):
    def __init__(self):
        super().__init__()

    def start(self):
        pass


class KilledJobProcessor(multiprocessing.Process):
    def __init__(self):
        super().__init__()

    def start(self):
        pass


class ClusterStatusProcessor(multiprocessing.Process):
    def __init__(self):
        super().__init__()

    def start(self):
        pass


if __name__ == "__main__":
    manager = MasterNodeManager()
    manager.start_all_processor()

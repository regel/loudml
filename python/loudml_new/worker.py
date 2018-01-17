"""
LoudML worker
"""

import logging
import signal

g_worker = None

class Worker:
    """
    LoudML worker
    """

    def __init__(self, msg_queue):
        self._msg_queue = msg_queue
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def run(self, job_id, func_name, *args, **kwargs):
        """
        Run requested task and return the result
        """

        self._msg_queue.put({
            'type': 'job_state',
            'job_id': job_id,
            'state': 'running',
        })
        return getattr(self, func_name)(*args, **kwargs)



    """
    # Example
    #
    def do_things(self, value):
        if value:
        import time
        time.sleep(value)
        return {'value': value}
    else:
        raise Exception("no value")
    """


def init_worker(msg_queue):
    global g_worker
    g_worker = Worker(msg_queue)

def run(job_id, func_name, *args, **kwargs):
    global g_worker
    return g_worker.run(job_id, func_name, *args, **kwargs)

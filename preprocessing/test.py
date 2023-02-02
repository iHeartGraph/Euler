from multiprocessing import Manager, Pool
from time import sleep 

def mp_task(lock,idx):
    with lock:
        sleep(1)
        print("yay",idx)


with Manager() as manager:
    p = Pool(processes=4)
    lock = manager.Lock()
    tasks = [
        p.apply_async(mp_task, (lock, i))
        for i in range(8)
    ]
    [t.wait() for t in tasks]
    p.close()
    p.join()


import psutil
import testtsp
import multiprocessing as mp
import time

# communicate
worker_conn, master_conn = mp.Pipe()
threshold = 0.1

p = mp.Process(target=testtsp.looptest, args=[worker_conn])
terminate = False
p.start()

#for i in range(1000):
total_ite = 0

while total_ite <= 1000:
	if terminate:
		terminate = False
		p.join()
		print('main process is not terminated yet, launching new process')
		time.sleep(5)
		worker_conn, master_conn = mp.Pipe()
		p = mp.Process(target=testtsp.looptest, args=[worker_conn])
		p.start()

	master_conn.send(('rollout',None))
	mem = master_conn.recv()
	available_mem = psutil.virtual_memory().available

	print(mem[0] / (0.0 + available_mem))
	if mem[0] / (0.0 + available_mem) >= threshold:
		master_conn.send(('terminate',None))
		terminate = True

	total_ite += 1

	print(total_ite)

if not terminate:
	master_conn.send(('terminate',None))

#p.join()


#print('main process is not terminated yet, launching new process')
#time.sleep(100)
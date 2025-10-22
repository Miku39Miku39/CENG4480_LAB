from sense_hat import SenseHat
import multiprocessing
import random
import time, os

def proc_read_temp(sense, pipe):
    while True:
        temp = sense.get_temperature()
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        pipe.send([temp, color])
        print('Read temperature: {:.2f}'.format(temp))
        time.sleep(5)

def proc_display_temp(sense, pipe):
    while True:
        temp, color = pipe.recv()
        disp_msg = '{:.2f}'.format(temp)
        sense.show_message(disp_msg, text_colour=color)

if __name__ == '__main__':
    sense = SenseHat()
    sense.clear()
    pipe = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=proc_read_temp, args=(sense, pipe[0]))
    p2 = multiprocessing.Process(target=proc_display_temp, args=(sense, pipe[1]))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
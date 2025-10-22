from sense_hat import SenseHat
from time import sleep
import copy
import random

def pos2idx(pos):
    return pos[0]*8+pos[1]

def move(head_pos,body,direction,eating):
    body_list=body
    body_list.insert(0,copy.deepcopy(head_pos))
    #print(head_pos,body)
    if direction == 'up':
        head_pos[1]-=1
        if head_pos[1]<0:
            head_pos[1]=7
    elif direction == 'down':
        head_pos[1]+=1
        if head_pos[1]>7:
            head_pos[1]=0
    elif direction == 'left':
        head_pos[0]-=1
        if head_pos[0]<0:
            head_pos[0]=7
    elif direction == 'right':
        head_pos[0]+=1
        if head_pos[0]>7:
            head_pos[0]=0
    if not eating:
        body_list.pop()
    #print(head_pos,body_list)
    
    # add eating logic
    return head_pos, body_list

def gen_food(body):
    ########################################
    # Question 2. Implement the gen_food(body) function
    # Input: body (List)
    # Output: food (List with 2 elements, i.e. a 2D vector)
    # Please make sure the food can will not be generated on the snake body pixel
    ########################################
    # Answer 2 start Here !!!

    
if __name__ == "__main__": 
    head_pos= [4,4]
    body=[[4,3],[4,2]]
    food = gen_food(body)

    eating=False
    direction='down'

    sense = SenseHat()

    red = (255, 0, 0)
    green = (0, 200, 0)
    blue = (0, 0, 255)
    black = (0,0,0)

    while True:
        sense.clear()
        if eating:
            food = gen_food(body)
        sense.set_pixel(food[0],food[1],(255,255,0))
        head_pos, body=move(head_pos,body,direction,eating)
        # print(head_pos,body)
        if head_pos in body:
            sense.clear(red)
            sleep(0.5)
            sense.clear()
            sleep(0.5)
            sense.clear(red)
            sleep(0.5)
            sense.show_letter('G',red)
            sleep(1)
            
            head_pos= [4,4]
            body=[[4,3],[4,2]]
            food = gen_food(body)
            eating=False
            direction='down'

        if head_pos==food:
            eating=True
        else:
            eating=False

        ########################################
        # Question 1. Use set_pixel function in sense to display the snake head, the head color should be red
        ########################################
        # Answer 1 start Here !!!
        
        for b in body:
            sense.set_pixel(b[0],b[1],green)
            
        sleep(0.5)
        for event in sense.stick.get_events()[::-1]:
            if event.action=='pressed':
                print("[{:.2f}] {}".format(event.timestamp, event.direction))
                if event.direction == 'middle': # reset
                    head_pos= [4,4]
                    body=[[4,3],[4,2]]
                    food = gen_food(body)

                    eating=False
                    direction='down'
                
                ########################################
                # Question 3. Use the joystick to control snake direction
                ########################################
                # Answer 3 start Here !!!
        


                break


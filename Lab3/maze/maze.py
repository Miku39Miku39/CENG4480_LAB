from sense_hat import SenseHat
from time import sleep
from maze_lib import *

MAP_NO = 1


def move_marble(pitch, roll, x, y, W):
    new_x = x
    new_y = y
    if 1 < pitch < 179 and x != 0:
        new_x -= 1
    elif 359 > pitch > 179 and x != 7 :
        new_x += 1
    if 1 < roll < 179 and y != 7:
        new_y += 1
    elif 359 > roll > 179 and y != 0 :
        new_y -= 1
    x,y = check_wall(x, y, new_x, new_y, W)
    return x,y

def check_wall(x, y, new_x, new_y, W):
    if maze[new_y][new_x] != W:
        return new_x, new_y
    elif maze[new_y][x] != W:
        return x, new_y
    elif maze[y][new_x] != W:
        return new_x, y
    return x,y

def check_win(x, y, dst):
    if maze[y][x] == dst:
        sense.clear()
        sense.show_message('You Win')
        return True
    return False

if __name__ == '__main__':   
    x = 1
    y = 1
    maze = get_maze(MAP_NO)
    
    sense = SenseHat()
    sense.clear()
    win = False

    while True:
        ########################################
        # Question 1. Read sensor to get orientation
        ########################################
        # Fill the following two lines

        # sense.get_orientation() returns a dictionary 
        # with the keys 'pitch', 'roll' and 'yaw'
        # For this maze game, we only need to use pitch and roll, 
        # values are obtained using the keys 'pitch' and 'roll'
        
        pitch = sense.get_orientation()['pitch']
        roll = sense.get_orientation()['roll']
        
        # print(sense.get_orientation())
        # Move marble
        x,y = move_marble(pitch, roll, x, y, W)
        # Check whether you arrive destination
        win = check_win(x, y, D)
        if win:
            sleep(5)
            sense.clear()
            break
        maze[y][x] = B
        
        ########################################
        # Question 2. Display the new map
        ########################################
        # Your Answer 2 starts here !!!
        
        # sense.set_pixels() allows the assigning of each pixel on the LED display, 
        # since the function accepts a list of 64 smaller lists of RGB values, 
        # we can then refer to the provided maze orientation from `maze_lib` 
        # and iterate over the whole array of coloured pixels.
        
        sense.set_pixels([pixel for row in maze for pixel in row])
        
        
        sleep(0.05)
        maze[y][x] = P
        
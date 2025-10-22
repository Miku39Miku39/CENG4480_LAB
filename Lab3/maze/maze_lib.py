W = (50, 0, 50)    # Wall Color
P = (0, 0, 0)      # Path Color
B = (255, 0, 0)    # Your Ball Color
D = (0, 255, 0)    # Destination Color

def get_maze(maze_no=0):
    if maze_no == 0:
        maze = [[W, W, W, W, W, W, W, W],
                [W, P, W, P, P, P, P, W],
                [W, P, W, P, W, W, D, W],
                [W, P, W, P, W, W, W, W],
                [W, P, W, P, P, P, P, W],
                [W, P, W, W, W, W, P, W],
                [W, P, P, P, P, P, P, W],
                [W, W, W, W, W, W, W, W]]
    elif maze_no == 1:
        maze = [[W, W, W, W, W, W, W, W],
                [W, P, P, P, P, P, P, W],
                [W, W, W, W, W, W, P, W],
                [W, D, P, P, W, P, P, W],
                [W, W, P, W, W, P, W, W],
                [W, W, P, W, P, P, W, W],
                [W, P, P, P, P, P, P, W],
                [W, W, W, W, W, W, W, W]]
    elif maze_no == 2:
        maze = [[W, W, W, W, W, W, W, W],
                [W, P, W, P, P, P, W, W],
                [W, P, W, P, W, P, W, W],
                [W, P, W, P, W, P, W, W],
                [W, P, W, P, W, P, W, W],
                [W, P, W, P, W, P, W, W],
                [W, P, P, P, W, P, D, W],
                [W, W, W, W, W, W, W, W]]
    elif maze_no == 66:
        maze = [[W, W, W, W, W, W, W, W],
                [W, P, P, P, W, W, W, W],
                [W, P, P, P, W, W, W, W],
                [W, P, P, P, W, W, W, W],
                [W, W, W, W, W, W, W, W],
                [W, W, W, W, W, P, P, W],
                [W, W, W, W, W, P, D, W],
                [W, W, W, W, W, W, W, W]]
    elif maze_no == 99:
        maze = [[W, W, W, W, W, W, W, W],
                [W, P, W, W, W, W, W, W],
                [W, W, W, W, W, W, W, W],
                [W, W, W, W, W, W, W, W],
                [W, W, W, W, W, W, W, W],
                [W, W, W, W, W, W, W, W],
                [W, W, W, W, W, W, W, W],
                [W, W, W, W, W, W, W, W]]
    else:
        raise "Wrong Map Number"
    
    return maze

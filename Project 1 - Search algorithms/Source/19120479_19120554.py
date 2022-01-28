from os import system, name
import matplotlib.pyplot as plt
import math
import numpy as np

#------------------------------------utility functions------------------------------------
def visualize_maze(matrix, bonus, start, end, route=None, visited=None, fig_size=(6.4, 4.8)):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    plt.show(block=True)
    ax=plt.figure(dpi=100, figsize=fig_size).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green') 

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')
    
    if visited:
        plt.scatter([i[1] for i in visited],[-i[0] for i in visited],
                    marker='.',s=100,color='blue', alpha= 0.1)

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='blue')

    plt.text(end[1],-end[0],'EXIT',color='red',
         horizontalalignment='center',
         verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    
    for _, point in enumerate(bonus):
        print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')
        
def read_file(file_name: str = 'maze.txt'):
    f=open(file_name,'r')
    n_bonus_points = int(next(f)[:-1])
    bonus_points = []
    for i in range(n_bonus_points):
        x, y, reward = map(int, next(f)[:-1].split(' '))
        bonus_points.append((x, y, reward))

    text=f.read()
    matrix=[list(i) for i in text.splitlines()]
    f.close()
    return bonus_points, matrix
        
#------------------------------------class------------------------------------
class Node:
    def __init__(self, x: int, y: int, bonus=0):
        self.x = x 
        self.y = y
        self.bonus = bonus # bonus point (negative)
        self.next = [] # save adjacency nodes
        self.pos = -1 # ordinal number (positive)
        self.wall = False # check if the node is a wall 'x'

    def __repr__(self):
        neighbors = [(item.x, item.y) for item in self.next]
        return f'({self.x},{self.y}) -> {neighbors}'

class Maze:
    def __init__(self, matrix, bonus_points= []):
        self.height = len(matrix)
        self.width = len(matrix[0])
        self.V = 0 # no. of non-wall nodes
        self.graph = [] # adjacency list
        self.bonuses = [] # list of bonus nodes

        # make each item in the matrix be a node itself 
        graph = []
        for i in range(self.height):
            for j in range(self.width):
                if matrix[i][j] != 'x': # not wall
                    node = Node(i,j)
                    self.V += 1

                    if matrix[i][j] == '+':
                        node.bonus = [item[-1] for item in bonus_points if (item[0], item[1]) == (i, j)][0] # bonus line structure: <x y neg_score>
                        self.bonuses.append(node)
                    elif matrix[i][j] in ['S', 's', '*']: #detect start point
                        self.start = node 
                    elif i in [0,self.height-1] or j in [0,self.width-1]: #detect end point
                        self.end = node
                else: # wall
                    node = Node(-1,-1)
                    node.wall = True
                node.pos = i * self.width + j
                graph.append(node) 
        # sort bonus points based on their scores (ascending)
        self.bonuses.sort(key= lambda x: x.bonus)
              
        # create adj list
        for item in graph:
            if item.wall == False:
                x,y = item.x, item.y
                if x + 1 < self.height and matrix[x+1][y] != 'x':
                    item.next.append(graph[(x+1) * self.width + y])
                if matrix[x-1][y] != 'x':
                    item.next.append(graph[(x-1) * self.width + y])
                if y + 1 < self.width and matrix[x][y+1] != 'x': 
                    item.next.append(graph[x * self.width + y+1])
                if matrix[x][y-1] != 'x':
                    item.next.append(graph[x * self.width + y-1])
            self.graph.append(item)
    
    def print_graph(self):
        for node in self.graph:
            print(node)        
        
#------------------------------------BFS------------------------------------
def bfs(maze, start: Node, end: Node):
    queue = [start]
    vis = [start]
    node_init = Node(-2,-2)
    par = [node_init] * (maze.height * maze.width)

    while queue:
        v = queue.pop(0)

        if v == end:
            break
        for adj_node in v.next:
            if adj_node not in vis:
                queue.append(adj_node)
                vis.append(adj_node)
                par[adj_node.pos] = v 

    # construct a path
    path = [end]
    cur = end
    total_cost = 0
    
    while path[-1].pos != start.pos:
        if cur in maze.bonuses:
            total_cost += cur.bonus
        total_cost += 1
        parent = par[cur.pos]
        path.append(parent)
        cur = parent
        
    path.reverse()
    path = [(item.x, item.y) for item in path]
    vis = [(item.x, item.y) for item in vis]
    return path, vis, total_cost

#------------------------------------DFS------------------------------------
def dfs(maze: Maze, start: Node, end: Node):
    stack = [start]
    vis = [start]
    node_init = Node(-2,-2)
    par = [node_init] * (maze.height * maze.width)

    while stack:
        v = stack.pop(0)

        if v.x == end.x and v.y == end.y:
            break
        for adj_node in v.next:
            i = 0

            if adj_node not in vis:
                stack.insert(i, adj_node)
                vis.append(adj_node)
                par[adj_node.pos] = v 
                i += 1
    
    # construct a path
    path = [end]
    cur = end
    total_cost = 0
    
    while path[-1].pos != start.pos:
        
        if cur in maze.bonuses:
            total_cost += cur.bonus
        total_cost += 1
        parent = par[cur.pos]
        path.append(parent)
        cur = parent
        
    path.reverse()
    path = [(item.x, item.y) for item in path]
    vis = [(item.x, item.y) for item in vis]
    return path, vis, total_cost        
        
#------------------------------------heuristic functions------------------------------------  
# euclid distance between 2 nodes
def dist(node1: Node, node2: Node):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
      
def heuristic_euclid(maze: Maze, end: Node):
    heuristic = [float("inf")] * (maze.height*maze.width)
    for node in maze.graph:
        if not node.wall:
            heuristic[node.pos] = math.sqrt((node.x - end.x)**2 + (node.y - end.y)**2)
    return heuristic # list

def heuristic_manhattan(maze: Maze, end: Node):
    heuristic = [float("inf")] * (maze.height*maze.width)
    for node in maze.graph:
        if not node.wall:
            heuristic[node.pos] = abs(node.x - end.x) + abs(node.y - end.y)
    return heuristic        
                    
def manhattan_custom(maze: Maze, end: Node):
    h = [float("inf")] * (maze.height*maze.width)
    
    for node in maze.graph:
        if not node.wall:
            for bonus in maze.bonuses:
                closest_bonus = [(item, dist(node, item)) for item in maze.bonuses]
                closest_bonus.sort(key= lambda tup: tup[1])
                closest_bonus = closest_bonus[0][0]
                
                h_tmp = dist(node, closest_bonus) + closest_bonus.bonus + dist(closest_bonus, maze.end)
                expected_len = dist(node, end)
                h[node.pos] = min(h_tmp, expected_len)
    return h 
                           
#------------------------------------Greedy with no bonus------------------------------------
# parameter h: heuristic
def gbfs(maze: Maze, start: Node, end: Node, h):
    vis = [start]
    node_init = Node(float("inf"), float("inf"))
    par = [node_init] * (maze.height * maze.width)
    # f(n) = h(n)  evaluated function
    f = [float("inf")] * (maze.height * maze.width)
    f[start.pos] = h[start.pos]
    # open list - new added content will be ascending order (sort by height of heuristic)
    stack = [(start, f[start.pos])]

    while stack:
        cur = stack.pop(0)[0]

        newNeighbor = []

        if cur.x == end.x and cur.y == end.y:
            break
        for neighbor in cur.next:
            d = 1 # d(cur, neighbor)
            if neighbor in vis:
                f_new = d + h[neighbor.pos] # recalculate f[neighbor]
                if f_new < f[neighbor.pos]: # if a better path with smaller f is found
                    par[neighbor.pos] = cur
            else: # not visited
                vis.append(neighbor)
                f[neighbor.pos] = h[neighbor.pos]
                par[neighbor.pos] = cur
                if (neighbor, f[neighbor.pos]) not in stack:
                    newNeighbor.append((neighbor, f[neighbor.pos]))

        newNeighbor.sort(key = lambda item: item[1])
        
        i = 0
        for neighbor in newNeighbor:
            stack.insert(i, neighbor)
            i += 1
                    
    # construct a path
    path = [end]
    cur = end
    total_cost = 0
    
    while path[-1].pos != start.pos:
        
        if cur in maze.bonuses:
            total_cost += cur.bonus
        total_cost += 1
        parent = par[cur.pos]
        path.append(parent)
        cur = parent
        
    path.reverse()
    path = [(item.x, item.y) for item in path]
    vis = [(item.x, item.y) for item in vis]
    return path, vis, total_cost

#------------------------------------ A* with no bonus ------------------------------------  
# parameter h: heuristic      
def a_star(maze: Maze, start: Node, end: Node, h):
    vis = [start]
    node_init = Node(float("inf"), float("inf"))
    par = [node_init] * (maze.height * maze.width)
    # g(n): weight from start node to node 'n' 
    g = [float("inf")] * (maze.height * maze.width)
    g[start.pos] = 0
    # f(n) = h(n) + g(n): evaluated function
    f = [float("inf")] * (maze.height * maze.width)
    f[start.pos] = h[start.pos]
    # open list (always sort in f-descending order)
    queue = [(start, f[start.pos])]

    while queue:
        queue.sort(key = lambda item: item[1]) # sort by f, f = h + g
        cur = queue.pop(0)[0]

        if cur.x == end.x and cur.y == end.y:
            break
        for neighbor in cur.next:
            d = 1 # d(cur, neighbor)
            if neighbor in vis:
                f_new = g[cur.pos] + d + h[neighbor.pos] # recalculate f[neighbor]
                if f_new < f[neighbor.pos]: # if a better path with smaller f is found
                    g[neighbor.pos] = g[cur.pos] + d
                    par[neighbor.pos] = cur
            else: # not visited
                vis.append(neighbor)
                g[neighbor.pos] = g[cur.pos] + d
                f[neighbor.pos] = g[neighbor.pos] + h[neighbor.pos]
                par[neighbor.pos] = cur
                if (neighbor, f[neighbor.pos]) not in queue:
                    queue.append((neighbor, f[neighbor.pos]))
                    
    # reconstruct a path
    total_cost = 0
    path = [end]
    cur = end
    # trace back
    while path[-1].pos != start.pos:
        if cur in maze.bonuses:
            total_cost += cur.bonus
        total_cost += 1
        parent = par[cur.pos]
        path.append(parent)
        cur = parent
        
    path.reverse()
    path = [(item.x, item.y) for item in path]
    vis = [(item.x, item.y) for item in vis]
    return path, vis, total_cost        
        
#------------------------------------ A* with bonus ------------------------------------ 
def custom_sort(lst, bonus_points):
    '''
        Sort a queue in f-ascending order (f=g+h: evaluation) + favor bonus points
        lst: a queue of type 'list'
        bonus_points: a list of all bonus and their info in the maze
    '''
    # first, sort by f 
    lst.sort(key = lambda item: item[1])
    # second, prioritise bonus points
    i = 0

    while i < len(lst):
        istart = i
        iend = -1
        for j in range(i+1, len(lst)):
            if lst[j][-1] != lst[i][-1]:
                iend = j
                break
        
        if istart < iend:
            itmp = istart
            for k in range(istart, iend):
                if lst[k][0] in bonus_points:
                    lst[itmp], lst[k] = lst[k], lst[itmp]
                    itmp += 1
            i = iend
        else:
            i += 1
        
def a_star_bonus(maze: Maze, start: Node, end: Node, h):
    node_init = Node(float("inf"), float("inf"))
    # parent list
    par = [node_init] * (maze.height * maze.width)
    # current node to traverse
    cur = node_init
    # g(n): weight from start node to node 'n' 
    g = [float("inf")] * (maze.height * maze.width)
    g[start.pos] = 0
    # f(n) = h(n) + g(n): evaluated function
    f = [float("inf")] * (maze.height * maze.width)
    f[start.pos] = h[start.pos]
    # open list (always sort in f-descending order)
    queue = [[start, f[start.pos]]]
    # close list: save visited nodes
    vis = [start]
    # find const
    bonus_points = [abs(bonus_node.bonus) for bonus_node in maze.bonuses]
    # bonus_points.append(1)
    const = max(bonus_points) + 1
    
    while queue and cur != end:
        # queue.sort(key = lambda item: item[1]) # sort by f, f = h + g
        custom_sort(queue, maze.bonuses)
        cur = queue.pop(0)[0]
        
        for neighbor in cur.next:
            d = const + neighbor.bonus # tránh trọng số âm
            
            if neighbor in vis:
                f_new = g[cur.pos] + d + h[neighbor.pos] # recalculate f[neighbor]
                if f_new < f[neighbor.pos]: # if a better path with smaller f is found
                    g[neighbor.pos] = g[cur.pos] + d
                    par[neighbor.pos] = cur

            else: # not visited
                vis.append(neighbor)
                g[neighbor.pos] = g[cur.pos] + d
                f[neighbor.pos] = g[neighbor.pos] + h[neighbor.pos]
                par[neighbor.pos] = cur
                if [neighbor, f[neighbor.pos]] not in queue: 
                    queue.append([neighbor, f[neighbor.pos]])
        
    # reconstruct a path
    total_cost = 0
    path = [end]
    cur = end
    
    while path[-1].pos != start.pos:
        if cur in maze.bonuses:
            total_cost += cur.bonus
        total_cost += 1
        parent = par[cur.pos]
        path.append(parent)
        cur = parent
        
    path.reverse()
    path = [(item.x, item.y) for item in path]
    vis = [(item.x, item.y) for item in vis]
    return path, vis, total_cost	
        
#------------------------------------ Greedy with bonus ------------------------------------ 
def gbfs_bonus(maze: Maze, start: Node, end: Node, h):
    node_init = Node(float("inf"), float("inf"))
    par = [node_init] * (maze.height * maze.width)
    # f(n) = h(n): evaluated function
    f = [float("inf")] * (maze.height * maze.width)
    f[start.pos] = h[start.pos]
    # open list (always sort in f-descending order)
    stack = [[start, f[start.pos]]]
    # clost list: save visited nodes
    vis = [start]
    # find const
    bonus_points = [abs(bonus_node.bonus) for bonus_node in maze.bonuses]
    # bonus_points.append(1)
    const = max(bonus_points) + 1
    
    while stack:
        cur = stack.pop(0)[0]

        newNeighbor = []
        
        if (cur.x, cur.y) == (end.x, end.y):
            break
        
        for neighbor in cur.next:
            d = const + neighbor.bonus # tránh trọng số âm
                    
            if neighbor in vis:
                f_new = d + h[neighbor.pos] 

            else: # not visited
                vis.append(neighbor)
                f[neighbor.pos] = h[neighbor.pos] + d
                par[neighbor.pos] = cur
                if (neighbor, f[neighbor.pos]) not in stack:
                    newNeighbor.append((neighbor, f[neighbor.pos]))

        custom_sort(newNeighbor, maze.bonuses)
        
        i = 0
        for neighbor in newNeighbor:
            stack.insert(i, neighbor)
            i += 1
    
    # reconstruct a path
    total_cost = 0
    path = [end]
    cur = end
    
    while path[-1].pos != start.pos:
        if cur in maze.bonuses:
            total_cost += cur.bonus
        total_cost += 1
        parent = par[cur.pos]
        path.append(parent)
        cur = parent
        
    path.reverse()
    path = [(item.x, item.y) for item in path]
    vis = [(item.x, item.y) for item in vis]
    return path, vis, total_cost        
       
#------------------------------------main------------------------------------
def clrscr():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux (here, os.name is 'posix')
    else:
        _ = system('clear')
        
menu = ['\tMenu', '1. Map with no bonus', '2. Map with bonus', '3. Exit the program']

while 1:
    for item in menu:
        print(item)
    print('-----------------------')
    option = int(input('Your choice: '))
    clrscr()
    
    if option == 3:
        break

    while 1:
        if option == 1:
            search_methods = ['BFS', 'DFS', 'A*', 'Greedy Best First Search', 'Back to choosing file']
            heuristic_methods = ['Manhattan', 'Euclid']
            files = ['map_no_bonus0' + str(i + 1) + '.txt' for i in range(5)]
            files.append('Back to menu')
            
            # input file
            for i in range(len(files)):
                print(f'{i+1}, {files[i]}')
            print('-----------------------')
            file_index = int(input('Enter a file (1 -> 6): '))
            
            if file_index == 6: # exit
                clrscr()
                break
            
            # read data from file
            file_path = './testcases/' + files[file_index - 1]
            bonus_points, matrix = read_file(file_path)
            maze = Maze(matrix, bonus_points)
            start = (maze.start.x, maze.start.y)
            end = (maze.end.x, maze.end.y)
                
            #input search method
            while 1:
                clrscr()
                print(f'File: {file_path}')
                for i in range(len(search_methods)):
                    print(f'{i+1}, {search_methods[i]}')
                print('-----------------------')    
                search_index = int(input('Enter a search method (1 -> 5): '))
                solution = None
                
                if search_index == 5:
                    clrscr()
                    break
                
                # run search
                if search_index == 1: # bfs
                    solution = bfs(maze, maze.start, maze.end)
                elif search_index == 2: # dfs
                    solution = dfs(maze, maze.start, maze.end)
                elif search_index == 3: # A*
                    # input a heuristic
                    print('\nChoose 1 of 2 heuristic funtions')
                    for i in range(len(heuristic_methods)):
                        print(f'{i+1}, {heuristic_methods[i]}')
                    print('-----------------------')    
                    heuristic_index = int(input('Enter a heuristic (1 or 2): ')) 
                    
                    heuristic = None
                    if heuristic_index == 1:
                        heuristic = heuristic_manhattan(maze, maze.end)
                    else:
                        heuristic = heuristic_euclid(maze, maze.end)
                        
                    solution = a_star(maze, maze.start, maze.end, heuristic)
                elif search_index == 4: # greedy
                    # input a heuristic
                    print('\nChoose 1 of 2 heuristic funtions')
                    for i in range(len(heuristic_methods)):
                        print(f'{i+1}, {heuristic_methods[i]}')
                    print('-----------------------')    
                    heuristic_index = int(input('Enter a heuristic (1 or 2): ')) 
                    
                    heuristic = None
                    if heuristic_index == 1:
                        heuristic = heuristic_manhattan(maze, maze.end)
                    else:
                        heuristic = heuristic_euclid(maze, maze.end)
                        
                    solution = gbfs(maze, maze.start, maze.end, heuristic)
                path, vis, cost = solution
                
                # visualize
                print('----Maze information and total cost:')
                visualize_maze(matrix, bonus_points, start, end, route=path, fig_size=(12,7)) #, (12,15), (10,6)
                print(f'Total cost = {cost}')   
                
                tmp = input('Press ENTER to escape')  
    
        else: # option == 2
            search_methods = ['A*', 'Greedy Best First Search', 'Back to choosing file']
            heuristic_methods = ['Manhattan', 'Euclid', 'Manhattan custom']
            files = ['map_bonus0' + str(i + 1) + '.txt' for i in range(3)]
            files.append('Back to menu')
            
            # input file
            for i in range(len(files)):
                print(f'{i+1}, {files[i]}')
            print('-----------------------')
            file_index = int(input('Enter a file (1 -> 4): ')) 
            
            if file_index == 4: # exit
                clrscr()
                break
                
            # read data from file
            file_path = './testcases/' + files[file_index - 1]
            bonus_points, matrix = read_file(file_path)
            maze = Maze(matrix, bonus_points)
            start = (maze.start.x, maze.start.y)
            end = (maze.end.x, maze.end.y) 
                
            #input search method
            while 1:
                clrscr()
                print(f'File: {file_path}')
                for i in range(len(search_methods)):
                    print(f'{i+1}, {search_methods[i]}')
                print('-----------------------')    
                search_index = int(input('Enter a search method (1 -> 3): '))
                solution = None
                
                if search_index == 3:
                    clrscr()
                    break
                
                # run search
                if search_index == 1: # A*
                    # input a heuristic
                    print('\nChoose 1 of 3 heuristic funtions')
                    for i in range(len(heuristic_methods)):
                        print(f'{i+1}, {heuristic_methods[i]}')
                    print('-----------------------')    
                    heuristic_index = int(input('Enter a heuristic (1 -> 3): ')) 
                    heuristic = None
                    
                    if heuristic_index == 1:
                        heuristic = heuristic_manhattan(maze, maze.end)
                    elif heuristic_index == 2:
                        heuristic = heuristic_euclid(maze, maze.end)
                    else:
                        heuristic = manhattan_custom(maze, maze.end)
                        
                    solution = a_star_bonus(maze, maze.start, maze.end, heuristic)
                else: # greedy
                    # input a heuristic
                    print('\nChoose 1 of 3 heuristic funtions')
                    for i in range(len(heuristic_methods)):
                        print(f'{i+1}, {heuristic_methods[i]}')
                    print('-----------------------')    
                    heuristic_index = int(input('Enter a heuristic (1 -> 3): ')) 
                    heuristic = None
                    
                    if heuristic_index == 1:
                        heuristic = heuristic_manhattan(maze, maze.end)
                    elif heuristic_index == 2:
                        heuristic = heuristic_euclid(maze, maze.end)
                    else:
                        heuristic = manhattan_custom(maze, maze.end)
                        
                    solution = gbfs_bonus(maze, maze.start, maze.end, heuristic)
                path, vis, cost = solution
                
                # visualize
                print('----Maze information and total cost:')
                visualize_maze(matrix, bonus_points, start, end, route=path, fig_size=(12,7)) #, (12,15), (10,6)
                print(f'Total cost = {cost}')   
                
                tmp = input('Press ENTER to escape')  

               
            
            
            
            
            
            
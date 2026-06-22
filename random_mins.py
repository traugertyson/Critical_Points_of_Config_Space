import random
import numpy as np
import math
import matplotlib.pyplot as plt

n = 4
d = 2
w = 2.5
h = 10
scalar = 5
hardness = 30

step_size = 0.001
error = 0.001
"""
Am doing gradient descent. However since this is constrained I am at first doing it with penalty bc thats the easiest
way off the bat
"""

def get_random_ball(w,h,d):
    num = []
    for i in range(d-1): #width only deals with first d dims
        num.append(random.uniform(.5,w-.5)) #centers only so need to minus radius of 1/2
    num.append(random.uniform(.5,h-.5))
    return np.array(num)

"""
RENAME THIS ACUTALLY GETS SQUARED DISTANCE
"""
def get_distance(x,y):
    total = 0
    for i in range(len(x)):
        total += (x[i] - y[i])**2
    return total

    
def check_valid(balls,new_ball):
    for i in balls:
        val = i - new_ball
        if np.dot(val,val) < 1:
            return False
    return True

def get_random_point(n,w,h,d):
    count = 0
    balls = []
    while count < n:
        new_ball = get_random_ball(w,h,d)
        if check_valid(balls,new_ball):
            balls.append(new_ball)
            count+=1
        else: #failed so completely resample
            balls = []
            count = 0

    return balls

def get_energy_val(x):
    return sum([p[-1] for p in x])

"""
#penalty function: sum over distances of 1/dist^10
def get_penalty_val(x):
    total = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if j == i:
                continue
            dist = get_distance(x[i],x[j])
            total += scalar/(dist**10)
    for i in range(len(x)):
        for dim in range(d-1):
            total += scalar/((x[i][dim]-w - .5)**10) + scalar/((x[i][dim]+.5)**10)

        total += scalar/((x[i][dim]+.5)**10) #for the y! it cannot go through the floor as well

    return total
"""
def get_penalty_val(x):
    total = 0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            #if j == i:
            #    continue
            dist = get_distance(x[i],x[j])
            total += 1/((dist)**(2*hardness))
        for dim in range(d-1):
            #total += 1/((x[i][dim]-w-.5))**(2*hardness) + 1/((x[i][dim] + .5)**(2*hardness))
            total += 1/(math.exp(-1*(x[i][dim]-w+.5))**(2*hardness)) + 1/(math.exp(x[i][dim] - .5)**(2*hardness))
        #total += 1/((x[i][d-1] + .5)**(2*hardness))
        total += 1/(math.exp(x[i][d-1] - .5)**(2*hardness))
    return total

def get_total_energy(x):
    return get_energy_val(x) + get_penalty_val(x)

#for quickly protopying penalty functions; just does (e(x + h) - e(x))/h on each val
def get_lazy_grad(x):
    h = 0.0000001
    curr_val = get_total_energy(x)
    grad = [0 for i in range(n*d)]
    for i in range(n*d):
        new_x = [np.copy(x[i]) for i in range(len(x))]
        ball = i//d
        index = i%d
        new_x[ball][index] += h
        grad[i] = (get_total_energy(new_x) - curr_val)/h
    return grad

def get_grad(x):
    grad = [0 for x in range(n*d)]
    for i in range(n):
        grad[d*(i+1)-1] = 1 #getting grad of f

    #grad of penalty (assuming my calc is right lmao)
    for i in range(n*d):
        ball = i//d
        index = i%d
        for j in range(ball+1,n):
            dist = get_distance(x[ball],x[j])
            added_val = -2*hardness*(dist)**(-2*hardness-1)*2*(x[ball][index] - x[j][index])
            grad[i] +=added_val
            grad[j*d + index] +=added_val #+=? Maybe -=? 
        if index != d-1:
            grad[i] += -2*hardness*(x[ball][index] + .5)**(-2*hardness-1) + -2*hardness*(x[ball][index] - w - .5)**(-2*hardness-1)
        else:
            grad[i] += -2*hardness*(x[ball][index] + .5)**(-2*hardness-1)
    return grad




"""
def get_grad(x):
    grad = [0 for x in range(n*d)]
    for i in range(n):
        grad[d*(i+1)-1] = 1 #getting grad of f

    #grad of penalty (assuming my calc is right lmao)
    for i in range(n*d):
        ball = i//d
        index = i%d
        #grad[i] += -1.1/10*((x[ball][index] - w-.5)**(-11) + (x[ball][index]+.5)**(-11) + 2*sum([get_distance(x[ball],x[j])**(-11)*(x[ball][index] - x[j][index]) if j != ball else 0 for j in range(n)]))
        if index != d-1:
            grad[i] += -1*scalar/10*((x[ball][index] - w-.5)**(-11) + (x[ball][index]+.5)**(-11) + 2*sum([get_distance(x[ball],x[j])**(-11)*(x[ball][index] - x[j][index]) if j != ball else 0 for j in range(n)]))
        else:
            grad[i] += -1*scalar/10*((x[ball][index]+.5)**(-11) + 2*sum([get_distance(x[ball],x[j])**(-11)*(x[ball][index] - x[j][index]) if j != ball else 0 for j in range(n)]))

    return grad
"""

def update_point(x,grad,step_size):
    new_pt = []
    #print(step_size)
    for i in range(n):
        #print("added_vals: ", [step_size * grad[j] for j in range(d*i,d*(i+1))])
        new_pt.append(x[i] - np.array([step_size * grad[j] for j in range(d*i,d*(i+1))]))
    return new_pt

def get_total_dist(x,y):
    total = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            total+= abs(x[i][j] - y[i][j])
    return total
def grad_descent(x,step_size):
    change = 1000
    old_val = get_total_energy(x)
    while change > .00001:
        grad = get_lazy_grad(x)
        old_x = x
        #print("current point: ", x)
        #print("current_grad:", grad)
        x = update_point(x,grad,step_size)
        val = get_total_energy(x)
        change = get_total_dist(old_x,x)
        #if change > 1000:
        #    print(x)
        #    return 0
        #print(x)
    return x

def plot_balls(x):
    circles = []
    for i in range(n):
        circles.append(plt.Circle(x[i],.5))
    fig,ax = plt.subplots()
    for i in range(n):
        ax.add_patch(circles[i])

    x1 = (0,0)
    y1 = (0,max(w,n))
    x2 = (w,0)
    y2 = (w,max(w,n))
    plt.axline(x1,y1)#,(x2,y2))    
    plt.axline(x2,y2)#,(x2,y2))    
    plt.xlim([0,max(w,n)])
    plt.ylim([0,max(n,w)])
    plt.show()
    return fig

"""
adj mat of form 
distances
n is left wall
n+1 is floor
n+2 is right wall
"""
def get_contact_graph(x):
    epsilon = .15
    adj_mat = [[0 for i in range(n + 3)] for j in range(n + 3)]
    for i in range(n):
        for j in range(n):
            if get_distance(x[i],x[j]) < 1 + epsilon:
                adj_mat[i][j] = 1
        if x[i][0] < .5 + epsilon:
            adj_mat[i][n] = 1
            adj_mat[n][i] = 1
        if x[i][0] > w-.5 - epsilon:
            adj_mat[i][n+2] = 1
            adj_mat[n+2][i] = 1
        if x[i][1] < .5 + epsilon:
            adj_mat[i][n+1] = 1
            adj_mat[n+1][i] = 1

    return adj_mat

"""
i is permutation number:
0 is identity
1 is (1 2)
2 is (1 3)
3 is (2 3)
4 is (1 2 3)
5 is (1 3 2)
"""
def permute_adj_mat(i,adj_mat):
    adj_mat_perm = np.array([[adj_mat[i][j] for j in range(len(adj_mat[i]))] for i in range(len(adj_mat))])
    if i == 0:
        return adj_mat_perm
    if i == 1: 
        adj_mat_perm[:,[1,0]] = adj_mat_perm[:,[0,1]]
        adj_mat_perm[[0,1],:] = adj_mat_perm[[1,0],:]
    elif i == 2:
        adj_mat_perm[:,[2,0]] = adj_mat_perm[:,[0,2]]
        adj_mat_perm[[0,2],:] = adj_mat_perm[[2,0],:]
    elif i == 3:
        adj_mat_perm[:,[2,1]] = adj_mat_perm[:,[1,2]]
        adj_mat_perm[[1,2],:] = adj_mat_perm[[2,1],:]
    elif i == 4:
        adj_mat_perm[:,[0,1,2]] = adj_mat_perm[:,[1,2,0]]
        adj_mat_perm[[0,1,2],:] = adj_mat_perm[[1,2,0],:]
    elif i == 5:
        adj_mat_perm[:,[0,1,2]] = adj_mat_perm[:,[2,0,1]]
        adj_mat_perm[[0,1,2],:] = adj_mat_perm[[2,0,1],:]
    return adj_mat_perm

def get_ball_order(xs):
    orderr = [[xs[i],i] for i in range(len(xs))]
    orderr = sorted(orderr, key=lambda x: x[0][0])
    return [x[1] for x in orderr]

def permute_matrix(col_order, adj_mat):
    #adj_mat_perm = np.zeros((len(adj_mat[0]),len(adj_mat)))#np.array([[0 for j in range(len(adj_mat[i]))] for i in range(len(adj_mat))])
    adj_mat_2 = np.array([np.array([adj_mat[i][j] for j in range(len(adj_mat[i]))]) for i in range(len(adj_mat))])
    adj_mat_perm = np.array([np.array([adj_mat[i][j] for j in range(len(adj_mat[i]))]) for i in range(len(adj_mat))])
    #swaps columns
    for i in range(len(col_order)):
        adj_mat_perm[:,i] = adj_mat_2[:,col_order[i]]
        #adj_mat_perm[i,:] = adj_mat_2[col_order[i],:]

    #swaps rows; have to do this after swapping rows or else it doesn't really swap rows
    adj_mat_2 = np.array([np.array([adj_mat_perm[i][j] for j in range(len(adj_mat[i]))]) for i in range(len(adj_mat))])
    for i in range(len(col_order)):
        adj_mat_perm[i,:] = adj_mat_2[col_order[i],:]
    return adj_mat_perm



def is_equal(a,b):
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] != b[i][j]:
            #if abs(a[i][j] - b[i][j]) < error:
                return False
    return True

"""
Currently assumes that for each adj_mat, there can only be 1 grav. balanced config
"""
def is_in(xs,adj_mat, list_adjs,totals):
    order = get_ball_order(xs) 
    adj_mat_2 = permute_matrix(order,adj_mat)
    print(np.matrix(adj_mat))
    print(adj_mat_2)
    for j in range(len(list_adjs)):
        if is_equal(adj_mat_2,list_adjs[j]):
            totals[j]+=1
            return j
    list_adjs.append(np.array(adj_mat_2))
    totals.append(1)
    return len(totals)
    """
    for j in range(len(list_adjs)):
        for i in range(6):
            if is_equal(permute_adj_mat(i,adj_mat),list_adjs[j]):
                totals[j] +=1
                return j
    list_adjs.append(np.array(adj_mat))
    totals.append(1)
    return len(totals)
"""

#for i in range(100):
#    print(get_random_point(n,w,h,d))
#x = [np.array([1.0,1.0]),np.array([1.1,3.0]),np.array([1.0,5.0])]
#print(get_penalty_val(x))
#print(get_grad(x))
#print(get_lazy_grad(x))
totals = []
adj_mats = []

while True:
    x = get_random_point(n,w,h,d)
    print(x)
    #try:
    x = grad_descent(x,step_size)
    #print(x)
    adj_mat = get_contact_graph(x)
    #print(np.array(adj_mat))
    print("RUNNNNNNNNNNNNNNNNN")
    ans = is_in(x,adj_mat,adj_mats,totals)
    print("Ans ", ans, len(totals))
    #print(totals)
    if ans == len(totals):
        plot_balls(x)
    #print(permute_adj_mat(1,adj_mat))
    #except e:
    #    continue
"""
x = [np.array([.6875,2]),np.array([1.375,1.0]),np.array([2.0625,2.0])]
x = grad_descent(x,step_size)
plot_balls(x)
"""

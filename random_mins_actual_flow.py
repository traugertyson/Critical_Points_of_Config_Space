import scipy
import numpy as np
import math
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import random

n = 7
d = 2 #THIS CODE IS ALWAYS ASSUMING d=2 FOR NOW!
w = 3

n_e = np.array([-1*(i%2) for i in range(d*n)])

tol = 10**(-6)


def get_rig_mat(x,contacts):
    mat = []
    for i in contacts:
        new_row = [0 for i in range(n*d)]
        ball_1 = i[0]
        ball_2 = i[1]
        if ball_2 >= 0 and ball_1 >= 0:
            offset_1 = d*ball_1
            offset_2 = d*ball_2
            for j in range(d):
                new_row[offset_1 + j] = x[offset_1 + j] - x[offset_2 + j]
                new_row[offset_2 + j] = x[offset_2 + j] - x[offset_1 + j]
            mat.append(new_row)
        elif ball_1 < 0:
            print("DONT PUT THE WALL AS A FIRST ARGUMENT DUMMY")
            return -1
        elif ball_2 < 0:
            offset_1 = d*ball_1
            #pos_ball_2 = -1*ball_2 #ball_2 is negative so making it positive to do math with; the negative is just an indicator
            if ball_2 == -1:
                    new_row[offset_1] = 1#x[offset_1 + pos_ball_2//2]+ .5
            elif ball_2 == -2:
                    new_row[offset_1 + 1] = 1#x[offset_1 + pos_ball_2//2]+ .5
            elif ball_2 == -3:
                    new_row[offset_1] = -1#x[offset_1 + pos_ball_2//2]+ .5
            else:
                raise Exception("invalid Wall")
            mat.append(new_row)
        else:
            print("Should literally never occur")

            """
            if ball_2%2 != 0:
                    new_row[offset_1 + pos_ball_2//2] = .5#x[offset_1 + pos_ball_2//2]+ .5
            else:
                    new_row[offset_1 + pos_ball_2//2] = -.5#x[offset_1 + pos_ball_2//2] - .5 + w #this could be wrong
            mat.append(new_row)
        else:
            print("Should literally never occur")
            """
    return mat

def obj(x):
    def objj(pt):
        return 0
        #return get_dot([x[i]-pt[i] for i in range(n*d)],[x[i] - pt[i] for i in range(n*d)])
    return objj

def find_feasible_point(x,contacts):
    """
    return {"x":x}
    """
    constraints = []
    for contact in contacts:
            constraints.append({'type':'ineq','fun':lambda pt: get_distance(pt,contact[0],contact[1])-1})
    objective = obj(x)
    result = scipy.optimize.minimize(objective,x,method="SLSQP",constraints=constraints)
    return result

def get_jac_mat(x,rig_mat):
    mat = []
    for row in range(n*d):
        new_row = [0 for s in range(n*d)]
        new_row[row] = 2
        e_i = [0 for s in range(n*d)]
        e_i[row] = 1
        Rx_e_i = rig_mat.dot(e_i)
        for extra in Rx_e_i.tolist()[0]:
            new_row.append(extra)
        mat.append(new_row)
    for row in range(len(rig_mat)):
        new_row = rig_mat[row,:].tolist()[0]
        print(new_row)
        for extra in range(len(rig_mat)):
            new_row.append(0)
        print(new_row)
        mat.append(new_row)
    return mat

def make_res_vec(contacts):
    vec = [-1*(i%2) for i in range(n*d)]
    for i in range(len(contacts)):
        vec.append(0)
    return vec


def get_push(x,contacts):
    if contacts == []:
        return n_e
    eye = np.eye(n*d)
    p = cp.Variable(n*d)
    rig_mat = np.matrix(get_rig_mat(x,contacts))
    prob = cp.Problem(cp.Minimize(cp.quad_form(p,eye) - n_e.T @ p),
                       [rig_mat @ p >= 0])
    prob.solve()
    return p.value

    """
    rig_mat = np.matrix(get_rig_mat(x,contacts))
    jac_mat = np.matrix(get_jac_mat(x,rig_mat))
    res_vec = make_res_vec(contacts)
    ans = scipy.linalg.solve(jac_mat,res_vec)
    return ans[0:n*d]

    if contacts == []:
        return n_e
    rig_mat = get_rig_mat(x,contacts)
    print(np.matrix(rig_mat))
    pinv_rig_mat = scipy.linalg.pinv(rig_mat)
    #pinv_rig_mat = np.matrix(rig_mat).T
    weights = scipy.optimize.lsq_linear(pinv_rig_mat,n_e,bounds=(0,np.inf))
    print(weights)
    return pinv_rig_mat.dot(weights['x'])
    """

def get_ball_i(x,i):
        return x[d*i:d*(i+1)]

def is_in_contacts(i,j,contacts):
    for s in contacts:
        if (s[0] == i and s[1] == j) or (s[0] == j and s[1] == i):
            return True
    return False

def get_total_distance_squared(a,b):
    total = 0
    for i in range(len(a)):
        total+= (a[i]-b[i])**2
    return total

def get_total_distance(a,b):
    return get_total_distance_squared(a,b)**.5

def get_dot(a,b):
    total = 0
    for i in range(len(a)):
        total+=a[i]*b[i]
    return total


def get_distance(x,ball_i,ball_j):
    i = ball_i
    j = ball_j
    if ball_i < 0 and ball_j < 0:
        raise Exception("Two walls should not be checked for distance")
    if ball_i < 0: #making it so ball_j is negative if either are
        tmp = ball_i
        ball_i = ball_j
        ball_j = tmp
    if ball_j >= 0:
        return get_total_distance(get_ball_i(x,i),get_ball_i(x,j))
    elif ball_j == -1:
        return get_ball_i(x,i)[0]+.5
    elif ball_j == -2:
        return get_ball_i(x,i)[1] + .5
    elif ball_j == -3:
        return -1*get_ball_i(x,i)[0]+w+.5
    else:
        raise Exception("Invalid ball given")

"""
Know that push pushes ball_i into ball_j. Thus we want to solve for when d^2((x+t*p)_i,(x+t*p)_j) = 1, which is a quadratic equation with respect to t. Further, we know that there are 2 times where this would happen: when it first pushes them into each other, and then it eventually pushes them all the way through the other. Thus these will be the two times, and the smaller of the two will be the one we want
"""
def get_step(x,push,i,j):
    if i < 0:
        raise Exception("NO NEGS IN FIRST IDIOT")
    if j >= 0:
        ball_i = get_ball_i(x,i)
        ball_j = get_ball_i(x,j)
        push_i = get_ball_i(push,i)
        push_j = get_ball_i(push,j)
        x_i_minus_x_j = [ball_i[s] - ball_j[s] for s in range(d)]
        p_i_minus_p_j = [push_i[s] - push_j[s] for s in range(d)]
        x_i_squared = get_dot(x_i_minus_x_j,x_i_minus_x_j) - 1 #minus 1 comes from this eventually being the c term in quadratic formula, so adding in the -1 to really make it the c term
        p_i_squared = get_dot(p_i_minus_p_j,p_i_minus_p_j)
        mixed = get_dot(x_i_minus_x_j,p_i_minus_p_j)
        discriminant = mixed**2 - 4*x_i_squared*p_i_squared
        return (-mixed-discriminant**.5)/(2*p_i_squared)
    else:
        ball_i = get_ball_i(x,i)
        push_i = get_ball_i(push,i)
        if j == -1:
            return (.5-ball_i[0])/push_i[0]
        elif j == -2:
            return (.5 - ball_i[1])/push_i[1]
        elif j == -3:
            return (w-.5-ball_i[0])/push_i[0]


"""
Checks to see if any new contacts are going to be made this step. If so, returns these new contacts, and the size needed to make the first new contact. If none, then returns contacts = [] and dist_moved = step_size
"""
def check_new_contacts(x,contacts,push,step_size):
    new_contacts = []
    new_pt = [x[i] + step_size*push[i] for i in range(n*d)]
    for i in range(n):
        for j in range(-3,n): #-3,-2,-1 are all walls
            if i == j or is_in_contacts(i,j,contacts):
                continue
            if ((j >= 0) and (j < i)):
                continue
            if get_distance(new_pt,i,j) <= 1+tol:
                #new_contacts.append(i,j)
                new_step_size = get_step(x,push,i,j)

                if new_step_size < step_size:
                    new_contacts = [[i,j]]
                    step_size = new_step_size
                    new_pt = [x[i] + step_size*push[i] for i in range(n*d)]
                elif new_step_size == step_size:
                    new_contacts.append([i,j])

    return new_contacts, step_size

"""
Checks to see if we need to push off our current stratification. If yes, deletes it from contacts. If not then doesnt do anything
"""
"""
Bad current implementation: what if
|ooo|
|o o|
-----
(all ontop of each other) with push down in the middle ball; this is in Rx_p = 0 space, but no push actually keeps it on the same strata.
"""
def check_break_contacts(x,push,contacts):
    tol = 0.00000001
    rig_mat = get_rig_mat(x,contacts)
    r_x_p = rig_mat @ p
    n_cont = []
    for i in range(len(r_x_p)):
        if abs(r_x_p[i]) < tol:
            n_cont.append(contacts[i])
    return n_cont







def get_contacts(x):
    contacts = []
    for i in range(n):
        for j in range(i,n):
            if i == j:
                continue
            if get_distance(x,i,j) <= 1 + tol:
                contacts.append([i,j])
    for i in range(n):
        for j in [-3,-2,-1]:
            if get_distance(x,i,j) <= 1 + tol:
                contacts.append([i,j])
    return contacts

def update_point(x,contacts,grad,step_size):

    #first, prune grad; if any values < tol then just set it to 0
    grad = [i if abs(i) > tol else 0 for i in grad]
    #check to see if any are going to be pushed into each other
    #dist_moved = how far need to in grad direction in order to get them to contact
    new_contacts, step_size = check_new_contacts(x,contacts,grad,step_size)
    new_x = [x[i] + step_size*grad[i] for i in range(n*d)]#update_point(x,grad,step_size)
    contacts = get_contacts(new_x)#check_break_contacts(new_x,x,contacts,step_size)
    #new_x = find_feasible_point(new_x,contacts)['x']
    """new idea here; just run the shit for current contacts and if push off then who cares"""
    #contacts = contacts + new_contacts
    return new_x,contacts,step_size,new_contacts

def get_total_energy(x):
    total = 0
    for i in range(n):
        total+= x[i*d+1]
    return total

def get_grad(x,contacts):
    push = get_push(x,contacts)
    norm = math.sqrt(push.T @ push)
    return push/norm
def grad_descent(x,contacts,step_size):
    iters = 0
    change = 1000
    old_val = get_total_energy(x)
    while change > 0.00001:
        grad = get_grad(x,contacts)
        #print(grad)
        old_x = x
        x, contacts,stepped_size,new= update_point(x,contacts,grad,step_size)
        if iters % 100 == 0:
            plot_balls(x,iters==0)
        val = get_total_energy(x)
        change = old_val - val
        old_val = val
        if new != []:
            print(new,contacts,stepped_size,step_size)
            change = 1
        iters+=1
    return x


def plot_balls(x,first):
    print(x)
    if not first:
        print("Clearing")
        plt.clf()
    circles = []
    for i in range(n):
        circles.append(plt.Circle(get_ball_i(x,i),.5))
    ax = plt.axes()
    for i in range(n):
        ax.add_patch(circles[i])

    print(ax)
    x1 = (0,0)
    y1 = (0,max(w,n))
    x2 = (w,0)
    y2 = (w,max(w,n))
    plt.axline(x1,y1)#,(x2,y2))    
    plt.axline(x2,y2)#,(x2,y2))    
    plt.xlim([0,max(w,n)])
    plt.ylim([0,max(n,w)])
    #plt.show(block=False)
    plt.pause(0.001)
    return fig


def check_valid(balls,new_ball):
    for i in balls:
        val = i - new_ball
        if np.dot(val,val) < 1:
            return False
    return True

def get_random_ball(w,h,d):
    num = []
    for i in range(d-1): #width only deals with first d dims
        num.append(random.uniform(.5,w-.5)) #centers only so need to minus radius of 1/2
    num.append(random.uniform(.5,h-.5))
    return np.array(num)

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

"""
#x = [.5,.5,.5+math.sqrt(2)/2,.5+math.sqrt(2)/2,1,3]
x = get_random_point(n,w,12,d)
x = [a for xs in x for a in xs]
print(x)

contacts = [[0,-1],[0,-2],[0,1]]

fig = plt.subplot()
x = grad_descent(x,contacts,0.001)
plot_balls(x,False)
"""

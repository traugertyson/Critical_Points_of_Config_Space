"""
Idea is to flow with the quick method until it finishes. Then to go to the actual flow to find the minimum around there
"""
import random_mins#.py
import random_mins_actual_flow#.py

n = 10
w = 3
h = 10
d = 2 #MOST OF THIS ASSUMES d=2!

step_size = 0.001
tol = 10**(-6)



"""
Consider two minimums the same if their geometry is the same; 
"""
def is_in_mins(x,contacts,rig_mat,mins):
   #crude rn just checks if pts are close 
    x = transform_x_to_balls_x(x)
    x = sorted(x)
    x = [a for xs in x for a in xs]
    for i in range(len(mins['x'])):
        if get_dist(x,mins['x'][i]) < tol*n:
            return i
    return len(mins['x'])

def get_dist(a,b):
    total = 0
    for i in range(len(a)):
        total+= (a[i]-b[i])**2
    return total
def get_ball_order(xs):
    orderr = [[xs[i],i] for i in range(len(xs))]
    orderr = sorted(orderr, key=lambda x: x[0][0])
    return [x[1] for x in orderr]

"""
Takes x = [a_1,a_2,b_1,b_2,...] to x = [[a_1,a_2],[b_1,b_2],...]
"""
def transform_x_to_balls_x(x):
    ret = []
    for i in range(n):
        new_ball = []
        for j in range(d):
            new_ball.append(x[i*d+j])
        ret.append(new_ball)
    return ret


def run_random_point():
    print("Start", i)
    #x = random_mins.run_rand_start_grad_desc(n,w,h,d,step_size)
    x = random_mins.get_random_point(n,w,h,d)
    #random_mins.plot_balls(x)
    print("End")
    x = [a for xs in x for a in xs]
    print("Start 2")
    x = random_mins_actual_flow.run_grad_desc(n,w,h,d,step_size,x,True)
    print("End 2",i)
    return x


def trim_contacts(contacts):
    return contacts

def sort_x(x):
    if type(x[0]) == list: #if already a list
        return sorted(x)
    else:
        x = transform_x_to_balls_x(x)
        x = sorted(x)
        return [a for xs in x for a in xs]
totals = []
mins = {'x':[],'rig_mat':[],'contacts':[]}
for i in range(1000):
    x = run_random_point()

    contacts = random_mins_actual_flow.get_contacts(x)
    contacts = trim_contacts(contacts)
    rig_mat = random_mins_actual_flow.get_rig_mat(x,contacts)
    number = is_in_mins(x,contacts,rig_mat,mins)
    if number != len(totals):
        totals[number]+=1
    else:
        totals.append(1)
        mins['x'].append(sort_x(x))
        mins['rig_mat'].append(rig_mat)
        mins['contacts'].append(contacts)
        random_mins_actual_flow.plot_balls(x,False,False)
    print(totals)


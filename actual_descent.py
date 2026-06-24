"""
Idea is to flow with the quick method until it finishes. Then to go to the actual flow to find the minimum around there
"""
import random_mins#.py
import random_mins_actual_flow#.py

n = 3
w = 2.5
h = 5
d = 2 #MOST OF THIS ASSUMES d=2!

step_size = 0.0001
tol = 10**(-6)


def is_in_mins(x,mins):
    for comp in range(len(mins)):
        for j in range(len(x)):
            if abs(x[j] - mins[comp][j]) > tol:
                continue
        return comp
    return len(mins)

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

totals = []
mins = []

for i in range(1000):
    print("Start")
    x = random_mins.run_rand_start_grad_desc(n,w,h,d,step_size)
    #random_mins.plot_balls(x)
    print("End")
    x = [a for xs in x for a in xs]
    print("Start 2")
    x = random_mins_actual_flow.run_grad_desc(n,w,h,d,step_size,x,False)
    print("End 2")
    #random_mins_actual_flow.plot_balls(x,False,False)
    x = transform_x_to_balls_x(x)
    x = sorted(x)#,key = lambda x: x[0])
    ans = is_in_mins(x,mins)
    if ans == len(mins):
        random_mins.plot_balls(x)
        totals.append(1)
        mins.append([x[i] for i in range(len(x))])




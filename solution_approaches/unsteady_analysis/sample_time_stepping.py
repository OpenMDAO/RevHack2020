import openmdao.api as om 


p = om.Problem()

p.model.add_subsystem('calc', om.ExecComp('dx_dt = 5*t'), promotes=['*'])

p.setup()

# assume a time-step of 1 second

x =[0.,]
t =[0,]
delta_t = 1

for i in range(10): 
    p['t'] = t[i]
    p.run_model()
    
    x.append(delta_t*p['dx_dt'][0])
    t.append(t[i]+1)  

print(x)
print(t)




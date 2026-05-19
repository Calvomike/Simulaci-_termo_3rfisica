
import numpy as np
from vpython import *

# Web VPython 3.2

# Hard-sphere gas.

# Bruce Sherwood


win = 500

Natoms = 500  # change this to have more or fewer atoms

# Typical values
iteration = 0
L = 1  # container is a cube L on a side
gray = color.gray(0.7)  # color of edges of container
mass = 4e-3 / 6e23  # helium mass
Ratom = 0.02  # wildly exaggerated size of helium atom
# Ratom = 0.02
k = 1.4e-23  # Boltzmann constant
T = 300  # around room temperature
dt = 1e-5
g_accel = 9.8*4e5  # Acceleration magnitude
gravity_force = mass * g_accel * dt
APPLY_GRAVITY = False
APPLY_ANDERSEN = False

grav_iters = [0,1,1e5,2e5,5e5,1e6,2e6,5e6,1e7]
simulation_steps = 500

def vel_maxwell_boltzmann(T, massa, n=1):
    kB = 1.380649e-23  # constante de Boltzmann (J/K)
    
    sigma = np.sqrt(kB * T / massa)
    
    vx = np.random.normal(0, sigma, n)
    vy = np.random.normal(0, sigma, n)
    vz = np.random.normal(0, sigma, n)
    
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    return v

animation = canvas(width=win, height=win, align="left")
animation.range = L
animation.title = 'A "hard-sphere" gas'
s = """  Theoretical and averaged speed distributions (meters/sec).
  Initially all atoms have the same speed, but collisions
  change the speeds of the colliding atoms. One of the atoms is
  marked and leaves a trail so you can follow its path.
"""
animation.caption = s

d = L / 2 + Ratom
r = 0.005
boxbottom = curve(color=gray, radius=r)
boxbottom.append(
    [
        vector(-d, -d, -d),
        vector(-d, -d, d),
        vector(d, -d, d),
        vector(d, -d, -d),
        vector(-d, -d, -d),
    ]
)
boxtop = curve(color=gray, radius=r)
boxtop.append(
    [
        vector(-d, d, -d),
        vector(-d, d, d),
        vector(d, d, d),
        vector(d, d, -d),
        vector(-d, d, -d),
    ]
)
vert1 = curve(color=gray, radius=r)
vert2 = curve(color=gray, radius=r)
vert3 = curve(color=gray, radius=r)
vert4 = curve(color=gray, radius=r)
vert1.append([vector(-d, -d, -d), vector(-d, d, -d)])
vert2.append([vector(-d, -d, d), vector(-d, d, d)])
vert3.append([vector(d, -d, d), vector(d, d, d)])
vert4.append([vector(d, -d, -d), vector(d, d, -d)])

Atoms = []
p = []
apos = []
pavg = sqrt(2 * mass * 1.5 * k * T)  # average kinetic energy p**2/(2mass) = (3/2)kT

for i in range(Natoms):
    x = L * random() - L / 2
    y = L * random() - L / 2
    z = L * random() - L / 2
    if i == 0:
        Atoms.append(
            sphere(
                pos=vector(x, y, z),
                radius=Ratom,
                color=color.cyan,
                make_trail=True,
                retain=100,
                trail_radius=0.3 * Ratom,
            )
        )
    else:
        Atoms.append(sphere(pos=vector(x, y, z), radius=Ratom, color=gray))
    apos.append(vec(x, y, z))
    theta = pi * random()
    phi = 2 * pi * random()
    px = pavg * sin(theta) * cos(phi)
    py = pavg * sin(theta) * sin(phi)
    pz = pavg * cos(theta)
    p.append(vector(px, py, pz))

deltav = 100  # binning for v histogram
hist_max = 4000
hist_bins = int(hist_max // deltav + 1)


def barx(v):
    return int(v / deltav)  # index into bars array


nhisto = int(4500 / deltav)
histo = []
for i in range(nhisto):
    histo.append(0.0)
histo[barx(pavg / mass)] = Natoms

gg = graph(
    width=win,
    height=0.4 * win,
    xmax=3000,
    align="left",
    xtitle="speed, m/s",
    ytitle="Number of atoms",
    # ymax=Natoms * deltav / 1000,
)

theory = gcurve(color=color.blue, width=2)
dv = 10
for v in range(0, 3001 + dv, dv):  # theoretical prediction
    theory.plot(
        v,
        (deltav / dv)
        * Natoms
        * 4
        * pi
        * ((mass / (2 * pi * k * T)) ** 1.5)
        * exp(-0.5 * mass * (v**2) / (k * T))
        * (v**2)
        * dv,
    )

accum = []
for i in range(int(3000 / deltav)):
    accum.append([deltav * (i + 0.5), 0])
vdist = gvbars(color=color.red, delta=deltav)


def interchange(v1, v2):  # remove from v1 bar, add to v2 bar
    barx1 = barx(v1)
    barx2 = barx(v2)
    if barx1 == barx2:
        return
    if barx1 >= len(histo) or barx2 >= len(histo):
        return
    histo[barx1] -= 1
    histo[barx2] += 1

def checkCollisions():
    hitlist = []
    r2 = 2 * Ratom
    r2 *= r2
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i):
            aj = apos[j]
            dr = ai - aj
            if mag2(dr) < r2:
                hitlist.append([i, j])
    return hitlist

def get_histogram_heights(data):
    if not data:
        return []

    # 1. Determine how many bins we need based on the largest value
    # Example: if max is 250, we need 3 bins (0-99, 100-199, 200-299)
    num_bins = hist_bins
    
    # 2. Create our "columns" initialized at zero height
    columns = [0] * num_bins
    
    # 3. Fill the columns
    for value in data:
        val = (value.x ** 2 + value.y ** 2 + value.z ** 2)**(1/2) / mass
        bin_index = min(int(val // 100),num_bins-1)
        columns[bin_index] += 1
    return columns
        

for grav in grav_iters:
    gravity_force = mass * grav*9.8 * dt
    iteration = 0

    open(f"pos/{Ratom}_{grav:1g}.csv","w").close()
    for i in range(Natoms):
        x = L * random() - L / 2
        y = L * random() - L / 2
        z = L * random() - L / 2
        apos[i] = vector(x,y,z)
        Atoms[i].pos = apos[i]
        theta = pi * random()
        phi = 2 * pi * random()
        px = pavg * sin(theta) * cos(phi)
        py = pavg * sin(theta) * sin(phi)
        pz = pavg * cos(theta)
        p[i] = vector(px, py, pz)
    nhisto = 0  # number of histogram snapshots to average
    for it in range(simulation_steps):
        rate(300)
        # Accumulate and average histogram snapshots
        for i in range(len(accum)):
            accum[i][1] = (iteration * accum[i][1] + histo[i]) / (iteration + 1)

        # for i in range(len(accum2)):
        #     # histo2[i] = 10
        #     accum2[i][1] = (iteration * accum2[i][1] + histo2[i]) / (iteration + 1)
        if iteration % 10 == 0:
            # print(histo)
            # print(histo2)
            # print(accum)
            # print(accum2)
            vdist.data = accum
            # vdist2.data = accum2

        if APPLY_GRAVITY:
            for i in range(Natoms):
                p[i].z -= gravity_force
        # Update all positions
        for i in range(Natoms):
            apos[i] = apos[i] + (p[i] / mass) * dt
            Atoms[i].pos = apos[i]

        # Check for collisions
        hitlist = checkCollisions()

        # If any collisions took place, update momenta of the two atoms
        for ij in hitlist:
            i = ij[0]
            j = ij[1]
            ptot = p[i] + p[j]
            posi = apos[i]
            posj = apos[j]
            vi = p[i] / mass
            vj = p[j] / mass
            vrel = vj - vi
            a = vrel.mag2
            if a == 0:
                continue  # exactly same velocities
            rrel = posi - posj
            if rrel.mag > Ratom:
                continue  # one atom went all the way through another

            # theta is the angle between vrel and rrel:
            dx = dot(rrel, vrel.hat)  # rrel.mag*cos(theta)
            dy = cross(rrel, vrel.hat).mag  # rrel.mag*sin(theta)
            # alpha is the angle of the triangle composed of rrel, path of atom j, and a line
            #   from the center of atom i to the center of atom j where atome j hits atom i:
            alpha = asin(dy / (2 * Ratom))
            d = (2 * Ratom) * cos(
                alpha
            ) - dx  # distance traveled into the atom from first contact
            deltat = (
                d / vrel.mag
            )  # time spent moving from first contact to position inside atom

            posi = posi - vi * deltat  # back up to contact configuration
            posj = posj - vj * deltat
            mtot = 2 * mass
            pcmi = p[i] - ptot * mass / mtot  # transform momenta to cm frame
            pcmj = p[j] - ptot * mass / mtot
            rrel = norm(rrel)
            pcmi = pcmi - 2 * pcmi.dot(rrel) * rrel  # bounce in cm frame
            pcmj = pcmj - 2 * pcmj.dot(rrel) * rrel
            p[i] = pcmi + ptot * mass / mtot  # transform momenta back to lab frame
            p[j] = pcmj + ptot * mass / mtot
            apos[i] = posi + (p[i] / mass) * deltat  # move forward deltat in time
            apos[j] = posj + (p[j] / mass) * deltat

        xocs = 0
        for i in range(Natoms):
            loc = apos[i]
            if abs(loc.x) > L / 2:
                xocs +=1
                if loc.x < 0:
                    p[i].x = abs(p[i].x)
                    apos[i].x = - L/2
                else:
                    p[i].x = -abs(p[i].x)
                    apos[i].x = + L/2

            if abs(loc.y) > L / 2:
                xocs +=1
                if loc.y < 0:
                    p[i].y = abs(p[i].y)
                    apos[i].y = - L/2
                else:
                    p[i].y = -abs(p[i].y)
                    apos[i].y = + L/2

            if abs(loc.z) > L / 2:
                xocs +=1
                if loc.z < 0:
                    p[i].z = abs(p[i].z)
                    apos[i].z = - L/2
                else:
                    p[i].z = -abs(p[i].z)
                    apos[i].z = L/2
            # if APPLY_GRAVITY:
            #     p[i].y -= gravity_force
            # mod_e = norm(p[i]) / (mass * 2)
        prob = xocs / Natoms * 50
        if APPLY_ANDERSEN:
            for i in range(Natoms):
                if random() > prob:
                    continue
                new_vel = vel_maxwell_boltzmann(T,mass)
                vel_mod = sqrt(p[i].x*p[i].x+p[i].y*p[i].y+p[i].z*p[i].z) / mass
                new_scalar = new_vel/vel_mod
                # p[i].x =p[i].x* new_scalar
                # p[i].y =p[i].y* new_scalar
                # p[i].z = p[i].z* new_scalar
                p[i] = p[i] * new_scalar[0]
                # print(new_scalar,type(new_scalar))
        columns = get_histogram_heights(p)
        histo = columns

        # columns2 = get_histogram_heights2(apos)
        # print("Columnes 2:",columns2)
        # z_pos = []
        # for i in range(Natoms):
        #     z_pos.append(apos[i].z)
        # z_pos = np.array(z_pos)
        # print("Average",np.mean(z_pos))
        # print("STD:",np.std(z_pos))

        data_array = np.array([[v.x, v.y, v.z] for v in apos])
        # with open(f"pos/{Ratom}_{grav:1g}.csv", 'a') as f:
        #     # We use a comma delimiter to keep it as a CSV
        #     np.savetxt(f, data_array, delimiter=',')
            # f.write(str(data_array))
            # f.write("\n")
        
        
        # histo2 = columns2
        iteration +=1
        animation.caption = f"""\tIteració: {iteration}
        Prob Andersen: {str(prob*100) + "%" if APPLY_ANDERSEN else "DISABLED"}
        Gravetat: {grav if APPLY_GRAVITY else "DISABLED"} G"""
        # print(p[-3:])
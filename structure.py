import meep as mp
import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt



"defining the light source characteristics from subwavelength"

wvl_min = 0.5          
wvl_max = 1.5         
fmin = 1/wvl_max        
fmax = 1/wvl_min        

fcen = 0.5*(fmin+fmax)  
df = fmax-fmin                          

sy = 10
src_pt = mp.Vector3(-1.75,0,0)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=src_pt, size=mp.Vector3(y=sy))]

"now defining the cell characteristics"
resolution = 30  
sx = 7.5
sy = 10
pml_layers = [mp.PML(thickness=1,direction=mp.X)]
cell_size = mp.Vector3(sx,sy,0)
glass = mp.Medium(index=1.5)

"setting up the simulation"

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    default_material=glass,
                    sources=sources)
nfreq = 21

"setting up the monitor screen to get the flux"
mon_pt = mp.Vector3(1.25,0,0)
flux_mon = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy)))

f = plt.figure(dpi=120)
sim.plot2D(ax=f.gca())
plt.show()

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-4))

"resetting the simulation and defining the geometry of the grating"
sim.reset_meep()
input_flux = mp.get_fluxes(flux_mon)


geometry = [mp.Block(material=mp.Medium(index=1.5), size=mp.Vector3(3,mp.inf), 
                     center=mp.Vector3(-2.25,0,0)),
            mp.Block(material=mp.Medium(index=1.5), size=mp.Vector3(1,3),
                     center=mp.Vector3(-0.5,0,0))]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    geometry=geometry,
                    boundary_layers=pml_layers,
                    sources=sources)

mode_mon = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy)))

f2 = plt.figure(dpi=120)
sim.plot2D(ax=f2.gca())
plt.show()

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-4))

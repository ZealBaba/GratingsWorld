
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import argparse


"Then, I defined all the parameters of my grating structure that are constant"
resolution = 60         

dpml = 1.0              
dsub = 2.0              
dpad = 3.0              
gp = 10.0              
gh = 0.5 

"Then specify the characteristics of my source"

wvl_min = 0.5
wvl_max = 1.5          
fmin = 1/wvl_max        
fmax = 1/wvl_min
fcen = 0.5*(fmin+fmax)  
df = fmax-fmin
nfreq = 100              

k_point = mp.Vector3(0,0,0)

glass = mp.Medium(index=1.5)

"Then I define a function called grating, which takes variables gdc and oddz. "
"I also define my cell size and the position of the source. "
"Then I set the simulation, after which I set the geometry of the gratings and re-run the simulation." 
"The flux monitor output the wavelength, the transmittance for 4 different "
"grating duty cycle (0.1, 0.2, 0.3 and 0.4) and the phase of the output flux."

def grating(gdc,oddz):
  sx = dpml+dsub+gh+dpad+dpml
  sy = gp

  cell_size = mp.Vector3(sx,sy,0)
  pml_layers = [mp.PML(thickness=dpml,direction=mp.X)]

  src_pt = mp.Vector3(-1.75,0,0)
  sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez if oddz else mp.Hz,size=mp.Vector3(y=sy), center=src_pt, )]

  symmetries=[mp.Mirror(mp.Y, phase=+1 if oddz else -1)]

  sim = mp.Simulation(resolution=resolution,
                      cell_size=cell_size,
                      boundary_layers=pml_layers,
                      k_point=k_point,
                      default_material=glass,
                      sources=sources,
                      symmetries=symmetries)

  mon_pt = mp.Vector3(1.25,0,0)
  flux_mon = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0,sy,0)))

  sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-4))

  input_flux = mp.get_fluxes(flux_mon)

  sim.reset_meep()

  geometry = [mp.Block(material=glass, size=mp.Vector3(dpml+dsub,mp.inf,mp.inf), center=mp.Vector3(-2.25,0,0)),
              mp.Block(material=glass, size=mp.Vector3(gh,gdc*gp,mp.inf), center=mp.Vector3(-0.5,0,0))]

  sim = mp.Simulation(resolution=resolution,
                      cell_size=cell_size,
                      boundary_layers=pml_layers,
                      geometry=geometry,
                      k_point=k_point,
                      sources=sources,
                      symmetries=symmetries)

  mode_mon = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0,sy,0)))

  sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-4))

  freqs = mp.get_eigenmode_freqs(mode_mon)
  res = sim.get_eigenmode_coefficients(mode_mon, [1], eig_parity=mp.ODD_Z+mp.EVEN_Y if oddz else mp.EVEN_Z+mp.ODD_Y)
  coeffs = res.alpha

  mode_wvl = [1/freqs[nf] for nf in range(nfreq)]
  mode_tran = [abs(coeffs[0,nf,0])**2/input_flux[nf] for nf in range(nfreq)]
  mode_phase = [np.angle(coeffs[0,nf,0]) for nf in range(nfreq)]

  return mode_wvl, mode_tran, mode_phase

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-oddz', action='store_true', default=False, help='oddz? (default: False)')
  args = parser.parse_args()

  gdc = np.arange(0.1,0.4,0.1)
  mode_tran = np.empty((gdc.size,nfreq))
  mode_phase = np.empty((gdc.size,nfreq))
  for n in range(gdc.size):
    mode_wvl, mode_tran[n,:], mode_phase[n,:] = grating(gdc[n],args.oddz)
    
#The plot for transmittance and phase information for four grating duty cycle values

  plt.figure(dpi=150)

  plt.subplot(1,2,1)
  plt.pcolormesh(mode_wvl, gdc, mode_tran, cmap='hot_r', shading='gouraud', vmin=0, vmax=mode_tran.max())
  plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
  plt.xlabel("wavelength (μm)")
  plt.xticks([t for t in np.arange(wvl_min,wvl_max+0.1,0.1)])
  plt.ylabel("grating duty cycle")
  plt.yticks([t for t in np.arange(gdc[0],gdc[-1]+0.1,0.1)])
  plt.title("transmittance")
  cbar = plt.colorbar()
  cbar.set_ticks([t for t in np.arange(0,1.2,0.2)])
  cbar.set_ticklabels(["{:.1f}".format(t) for t in np.arange(0,1.2,0.2)])

  plt.subplot(1,2,2)
  plt.pcolormesh(mode_wvl, gdc, mode_phase, cmap='RdBu', shading='gouraud', vmin=mode_phase.min(), vmax=mode_phase.max())
  plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
  plt.xlabel("wavelength (μm)")
  plt.xticks([t for t in np.arange(wvl_min,wvl_max+0.1,0.1)])
  plt.ylabel("grating duty cycle")
  plt.yticks([t for t in np.arange(gdc[0],gdc[-1]+0.1,0.1)])
  plt.title("phase (radians)")
  cbar = plt.colorbar()
  cbar.set_ticks([t for t in range(-3,4)])
  cbar.set_ticklabels(["{:.1f}".format(t) for t in range(-3,4)])

  plt.tight_layout()
  plt.show()

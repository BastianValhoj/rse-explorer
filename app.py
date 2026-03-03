import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

# Load data
@st.cache_resource
def load_data():
    return h5py.File("RSE_data.h5", 'r')

data = load_data()

# create header for interactive parameters
st.sidebar.header("Parameters")

# create dropdown menu for N
n_options = np.asanyarray(data.attrs['N'])
selected_N = st.sidebar.selectbox("Tiling (N)", options=n_options)
current_N = data[f'N_{selected_N}']

# create dropdown for energy 
E_options = np.asanyarray(data.attrs["E"]).round(6)
selected_E = st.sidebar.selectbox("Energy (E)", options=E_options)

# create dropwdown for eta
#eta_options = np.asanyarray(data.attrs["ETA"])
eta_keys = sorted(
    [k for k in current_N.keys() if k.startswith('eta_')],
    key=lambda x: float(x.split('_')[1])
)

# Use format_func so the user sees "1.0e-01" instead of "eta_1.0e-01"
selected_eta_key = st.sidebar.selectbox(
    "Select eta", 
    options=eta_keys,
    format_func=lambda x: f"{float(x.split('_')[1]):.1e}"
)

# create slider for site to investigate
selected_site = st.sidebar.slider("Site Index", min_value=0, max_value=selected_N-1, value=0)

# create multiple choice for imaginary or real part of RSE
part = st.sidebar.radio("Part", options=["Imag","Real"]).lower()

# create slider for how many next nearest neighbours to include 
max_nn = np.max([selected_site, (selected_N - 1 - selected_site)])
selected_nn = st.sidebar.slider("Num NN", min_value=0, max_value=max_nn, value=0)

# create checkbox for left plot: for a given NN, show only one or both coupling values
show_nn_pair = st.sidebar.checkbox("Show paris of NN")

# create checkbox for global min max
use_global_minmax = st.sidebar.checkbox("Use global min/max")

# from selected N and eta, define plotting parameters
xyz = np.asanyarray(current_N["xyz"])
x,y,z = xyz.T

E_idx = np.argwhere(E_options == 0)[0,0] # find the index where E==0 (or the first instance)
RSE = current_N[selected_eta_key][E_idx,...]

if use_global_minmax:
    vmin = data.attrs["vmin"]
    vmax = data.attrs["vmax"]
    vmax = np.max(np.abs([vmin, vmax])) # find the greatest magnitude
    vmin = -vmax # set min as the negative greatest magnitude
elif not use_global_minmax:
    vmin = np.min([np.real(RSE), np.imag(RSE)])
    vmax = np.max([np.real(RSE), np.imag(RSE)])
    vmax = np.max(np.abs([vmin, vmax]))
    vmin = -vmax



# --- Plotting logic ---
def render_plot(N, part, site, nn, nn_pair):
    # find imag/real RSE for specific site
    part = part.lower()
    if (part == "real"): #or (("r" in part) and ("i" not in part)):
        RSE_part = np.real(RSE)
    elif (part == "imag"): #or (("i" in part) and ("r" not in part)):
        RSE_part = np.imag(RSE)
    site_coupling = RSE_part[site, :]
    
    
    # calculate distances
    all_coords = xyz
    site_coord = xyz[[site],:]
    electrode_coords = xyz[:N, :]
    site_mask = np.delete(np.arange(0, xyz.shape[0]), site)
    diff = all_coords[:, None, :] - site_coord[None, :, :] # (M, 1, N) - (1, K, N) = (M, K, L)
    diff_site = site_coord[:, None, :] - electrode_coords[None, :, :] # (1, 1, N) - (1, M, N) = (1, M, N)
    dists = np.linalg.norm(diff, axis=2).squeeze()
    dists_site = np.linalg.norm(diff_site, axis=2).squeeze()
   
    # set ylim for for left plot (with a slight offset so data does not touch y-axis
    ylim = (vmin*(1-2e-2), 
            vmax*(1+2e-1)
    )
    # calcualte xlim as the maximum distance to last atom on the side (from idx 0 to N-1)
    xlim = (0, np.linalg.norm(xyz[0, :] - xyz[N, :])*(1+1e-1)
    )
    
    # start figure
    fig, axes = plt.subplots(1,2, figsize=(10, 4), dpi=150)
    
    ## Left figure: scatter plot of distance and coupling 
    axes[0].scatter(dists[site_mask][N:], site_coupling[site_mask][N:], marker='.', color="r", s=15) # plot points NOT in first electrode side
    axes[0].scatter(dists[site_mask][:N], site_coupling[site_mask][:N], marker='+', color="b", s=30) # plot points IN the first electrode side
    axes[0].axhline(0, color="k",linestyle="--", linewidth=.8)
    
    # left axis params
    axes[0].set(xlabel=r"$r$ [$\mathrm{\AA}$]", 
                xlim=xlim,
                ylabel=rf"$\Sigma_{{{site}, j}}$", 
                ylim=ylim,
                title="Coupling Vs. distance"
    )
    
    # plot position colored by coupling
    sc = axes[1].scatter(x, y, c=site_coupling, vmin=vmin, vmax=vmax, cmap="RdBu", s=100/N)
    
    # right axis params
    axes[1].set(xlabel="$x$", 
                xticklabels="",
                ylabel="$y$",
                yticklabels="",
                title=rf"Coupling in 2D, $\Sigma_{{{site}, j}}$",
    )
    
    # Plot site on right plot, and NN on both
    # nearest_neighbours = nn
    farthest_dist = 0
    for s in np.arange(site-nn, site+nn+1, 1):
        if (0 <= s) and (s < N): # s has to be in valid range 
            if s != site: # if it is not the site, plot on both
                farthest_dist=np.max([farthest_dist, dists[s]])
                axes[1].text(x=x[s], y=y[s]-1, s=s, horizontalalignment="right", verticalalignment="top")   
            elif s == site: # if it is site, plot with arrow on right
                axes[1].annotate(site, xy=(x[site], y[site]), xycoords="data", xytext=(x[site]+2*N/9, y[site]+3*N/9), arrowprops={"color":"k", "width":0.10, "headwidth":3.0, "headlength": 4.0}, )
    # if nn:
    farthest_nn_mask = np.nonzero(np.isclose(dists_site, farthest_dist, atol=1e-2))[0]
    if not nn_pair:
        farthest_nn_mask = farthest_nn_mask[[0]]
    
    axes[0].axvline(farthest_dist, color='k', linestyle="--", lw=1, alpha=0.5)
    
    coupling_for_farthest_nn = site_coupling[farthest_nn_mask]
    for coup in coupling_for_farthest_nn:
        axes[0].annotate(text=f" {coup:.1f}", xy=(farthest_dist, coup), xytext=(farthest_dist, coup))
        
        
        
                
    axes[1].axis("off")
    axes[0].grid(True, which="both", linestyle="-", lw=0.2)
    
    # Add colorbar (using divider to match the rescaled box)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(sc, cax)
    
    
    st.pyplot(fig)

# run plot using parameters specified by interactive elements 
render_plot(selected_N, part, selected_site, selected_nn, show_nn_pair)

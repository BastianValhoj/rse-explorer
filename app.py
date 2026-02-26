import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import joblib

@st.cache_resource
def load_data():
    return joblib.load("RSE_data_tiling_conv.pkl")

data = load_data()

st.sidebar.header("Parameters")

n_options = sorted([int(k) for k in data.keys() if k.isdigit()])
selected_N = st.sidebar.selectbox("Tiling (N)", options=n_options)

current = data[str(selected_N)]
energies = current["E"]
xyz = current["xyz"]
x,y,z = xyz.T

# E0_idx = np.searchsorted(E, 0) # find the index where E==0 (or the first instance)
RSE_re = current["RSE_re"]

RSE_E0 = RSE_re[0, ...]

selected_site = st.sidebar.slider("Site Index", min_value=0, max_value=selected_N-1, value=0)

part = st.sidebar.radio("Part", options=["Imag","Real"]).lower()

max_nn = np.max([selected_site, (selected_N - 1 - selected_site)])
selected_nn = st.sidebar.slider("Num NN", min_value=0, max_value=max_nn, value=0)
show_nn_pair = st.sidebar.checkbox("Show paris of NN")


# --- Plotting logic ---

def render_plot(current_data, N, part, site, nn, nn_pair):
    part = part.lower()


    
    if (part == "real"): #or (("r" in part) and ("i" not in part)):
        RSE_E0_part = np.real(RSE_E0)
    elif (part == "imag"): #or (("i" in part) and ("r" not in part)):
        RSE_E0_part = np.imag(RSE_E0)
        
    site_coupling = RSE_E0_part[site, :]
    
    
    vmax = np.max(np.abs(RSE_E0_part))
    vmin = -vmax
    
    # calculate distance  
    all_coords = xyz
    site_coord = xyz[[site],:]
    electrode_coords = xyz[:N, :]
    site_mask = np.delete(np.arange(0, xyz.shape[0]), site)
    diff = all_coords[:, None, :] - site_coord[None, :, :] # (M, 1, N) - (1, K, N) = (M, K, L)
    diff_site = site_coord[:, None, :] - electrode_coords[None, :, :] # (1, 1, N) - (1, M, N) = (1, M, N)
    dists = np.linalg.norm(diff, axis=2).squeeze()
    dists_site = np.linalg.norm(diff_site, axis=2).squeeze()
    
    ylim = (np.min(RSE_E0_part)*(1-2e-2), 
            np.max(RSE_E0_part)*(1+2e-1)
    )
    # calcualte xlim as the maximum distance to last atom on the side (from idx 0 to N-1)
    xlim = (0, np.linalg.norm(xyz[0, :] - xyz[N, :])*(1+1e-1)
    )
    
    # start figure
    fig, axes = plt.subplots(1,2, figsize=(10, 4), dpi=150)
    
    # Left figure: scatter plot of distance and coupling 
    axes[0].scatter(dists[site_mask][N:], site_coupling[site_mask][N:], marker='.', color="r", s=15) # plot points not in first electrode side
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
    if nn:
        farthest_nn_mask = np.nonzero(np.isclose(dists_site, farthest_dist, atol=1e-2))[0]
        if not nn_pair:
            farthest_nn_mask = farthest_nn_mask[[0]]
        
        # print(f"mask = {farthest_nn_mask}")
        # farthest_nn_mask = farthest_nn_mask[farthest_nn_mask < N]
        # print(f"mask = {farthest_nn_mask}")
        # farthest_nn_mask = np.unique(farthest_nn_mask)
        # print(f"mask = {farthest_nn_mask}")
        
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
    
render_plot(current, selected_N, part, selected_site, selected_nn, show_nn_pair)

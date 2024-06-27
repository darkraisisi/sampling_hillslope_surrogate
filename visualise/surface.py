import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc
from matplotlib.ticker import FormatStrFormatter

def plot_lines(st_eq, un_eq, ax):
    for unstable in un_eq:
        ax.plot(unstable[:,0], unstable[:,1], zs=0, zdir='z', linestyle = 'dashed',
                linewidth=2, color = 'black', zorder=10)

    for stable in st_eq:
        ax.plot(stable[:,0], stable[:,1], zs=0, zdir='z', linestyle = 'solid',
                linewidth=2, color = 'black', zorder=11)


def eq_lines(contour, gradient, B_lim, D_lim, n_sq):
  # Find for what regions the equilibrium is stable
  grad_stab = gradient < 0

  dashed_lines = []
  solid_lines = [] 
  lines = contour.allsegs[0]
  # print(lines)
  for line in lines:
    if len(line) < 1:
      continue
    indices = (np.array(line)//np.array([(B_lim+1E-5)/n_sq, (D_lim+1E-6)/n_sq])).astype(int)
    stability = grad_stab[indices[:,0], indices[:,1]]
    current_line = line[0]

    for i in range(len(line)-1):
      if (stability[i] != stability[i+1]):
        midpoint = (line[i] + line[i+1])/2
        current_line = np.vstack([current_line, midpoint])
        if stability[i]:
          solid_lines.append(current_line)
        else:
          dashed_lines.append(current_line)
        current_line = midpoint
      current_line = np.vstack([current_line, line[i+1]])

    if stability[-1]:
          solid_lines.append(current_line)
    else:
      dashed_lines.append(current_line)

  return solid_lines, dashed_lines


def show(D_grid, B_grid, dD_dt, dB_dt, D_lim, B_lim):
    grad_B, _ = np.gradient(dB_dt)
    _, grad_D = np.gradient(dD_dt)

    # Set the parameters
    scale_surface = 2
    my_cmap = plt.cm.jet
    desaturation = 0.8
    jet_colors = my_cmap(np.arange(my_cmap.N))
    jet_colors_hsv = mc.rgb_to_hsv(jet_colors[:, :3])
    jet_colors_hsv[:, 1] *= desaturation
    jet_colors_desaturated = mc.hsv_to_rgb(jet_colors_hsv)
    my_cmap_desaturated = mc.ListedColormap(jet_colors_desaturated)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,6), subplot_kw={"projection": "3d"})

    ax[0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0]), np.diag([1, 1, 0.5, 1]))
    ax[1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1]), np.diag([1, 1, 0.5, 1]))
    ax[0].zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax[0].set_zlabel('Biomass net\ngrowth ($kg/m^2/yr$)', labelpad=34)
    ax[1].set_zlabel('Soil depth\nincrease (m/yr)', labelpad=36)
    plt.tight_layout()

    for ax_ in ax:
        ax_.xaxis.set_major_locator(plt.MaxNLocator(6, prune='lower'))
        ax_.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax_.tick_params(axis='z', pad=15)
        ax_.set_xlim(B_lim,0)
        ax_.set_ylim(0,D_lim)
        ax_.set_xlabel('Biomass ($kg/m^2$)', labelpad=20)
        ax_.set_ylabel('Soil depth ($m$)', labelpad=24)

    # ax[0]
    ax[0].plot_surface(B_grid, D_grid, dB_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black', 
                    alpha=1, shade=False ,rstride=scale_surface, cstride=scale_surface)

    dB_dt_0 = ax[0].contour3D(X=B_grid, Y=D_grid, Z=dB_dt, levels = [0.0], linewidths=0)
    st_eq_B, un_eq_B = eq_lines(dB_dt_0, grad_B, B_lim, D_lim, len(dB_dt))
    plot_lines(st_eq_B, un_eq_B, ax[0])

    # ax[1]
    ax[1].plot_surface(B_grid, D_grid, dD_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black', 
                alpha=1, shade=False ,rstride=scale_surface, cstride=scale_surface)

    dD_dt_0 = ax[1].contour3D(X=B_grid, Y=D_grid, Z=dD_dt, levels = [0.0], linewidths=0)
    st_eq_B, un_eq_B = eq_lines(dD_dt_0, grad_D, B_lim, D_lim, len(dD_dt))
    plot_lines(st_eq_B, un_eq_B, ax[1])


def pdf(D_grid, B_grid, dD_dt, dB_dt, D_pdf, B_pdf, D_lim, B_lim):
    # Set the parameters
    scale_surface = int(B_grid.shape[0] / 20)
    my_cmap = plt.cm.jet
    desaturation = 0.8
    jet_colors = my_cmap(np.arange(my_cmap.N))
    jet_colors_hsv = mc.rgb_to_hsv(jet_colors[:, :3])
    jet_colors_hsv[:, 1] *= desaturation
    jet_colors_desaturated = mc.hsv_to_rgb(jet_colors_hsv)
    my_cmap_desaturated = mc.ListedColormap(jet_colors_desaturated)
    
    fig, ax = plt.subplots(2, 2, figsize=(20,12), subplot_kw={"projection": "3d"})

    for i in range(2):
        ax[i,0].set_zlabel('Biomass net\ngrowth ($kg/m^2/yr$)', labelpad=34)
        ax[i,0].zaxis.set_major_formatter(FormatStrFormatter('%.6f'))

    for i in range(2):
        ax[i,1].set_zlabel('Soil depth\nincrease (m/yr)', labelpad=36)
        ax[i,1].zaxis.set_major_formatter(FormatStrFormatter('%.6f'))

    ax[0,0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0,0]), np.diag([1, 1, 0.5, 1]))
    ax[0,1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0,1]), np.diag([1, 1, 0.5, 1]))

    ax[1,0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1,0]), np.diag([1, 1, 0.5, 1]))
    ax[1,1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1,1]), np.diag([1, 1, 0.5, 1]))

    plt.tight_layout()

    for row_ax in ax:
        for ax_ in row_ax:
            ax_.xaxis.set_major_locator(plt.MaxNLocator(6, prune='lower'))
            ax_.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax_.tick_params(axis='z', pad=15)
            ax_.set_xlim(B_lim,0)
            ax_.set_ylim(0,D_lim)
            ax_.set_xlabel('Biomass ($kg/m^2$)', labelpad=20)
            ax_.set_ylabel('Soil depth ($m$)', labelpad=24)
    # Rate of change
    ## dB_dt
    ax[0, 0].plot_surface(B_grid, D_grid, dB_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black', 
                    alpha=1, shade=False ,rstride=scale_surface, cstride=scale_surface)

    # dD_dt
    ax[0, 1].plot_surface(B_grid, D_grid, dD_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black', 
                alpha=1, shade=False ,rstride=scale_surface, cstride=scale_surface)
    
    # B_pdf
    ax[1, 0].plot_surface(B_grid, D_grid, B_pdf, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black', 
                alpha=1, shade=False ,rstride=scale_surface, cstride=scale_surface)
    # ax[1, 0].set_zlim(0.0001, 0)
    # ax[1, 0].set_zlim(1e-5, 1E-4)
    
    # D_pdf
    ax[1, 1].plot_surface(B_grid, D_grid, D_pdf, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black', 
                    alpha=1, shade=False ,rstride=scale_surface, cstride=scale_surface)
    # ax[1, 1].set_zlim(0, 1E-4)
    # ax[1, 1].set_zlim(1e-5, 1E-4)
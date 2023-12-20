import numba
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.core import Structure, Lattice, Element
from jupyter_jsmol.pymatgen import quick_view

Z = ["Place holder, even He is diff with empty", "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]

P = ["Place holder, even He is diff with empty", "He","Ne","Ar","Kr","Xe","Rn","Fr","Cs","Rb","K","Na","Li","Ra","Ba","Sr","Ca","Yb","Eu","Y","Sc","Lu","Tm","Er","Ho","Dy","Tb","Gd","Sm","Pm","Nd","Pr","Ce","La","Lr","No","Md","Fm","Es","Cf","Bk","Cm","Am","Pu","Np","U","Pa","Th","Ac","Zr","Hf","Ti","Nb","Ta","V","Mo","W","Cr","Tc","Re","Mn","Fe","Os","Ru","Co","Ir","Rh","Ni","Pt","Pd","Au","Ag","Cu","Mg","Hg","Cd","Zn","Be","Tl","In","Al","Ga","Pb","Sn","Ge","Si","B","Bi","Sb","As","P","Po","Te","Se","S","C","At","I","Br","Cl","N","O","F","H"]

GA = ["Place holder, even He is diff with empty", "He","Ne","Ar","At","Rn","Fr","Es","Fm","Md","No","Lr","Kr","Xe","Pm","Cs","Rb","K","Na","Li","Ra","Ba","Sr","Ca","Eu","Yb","Lu","Tm","Y","Er","Ho","Dy","Tb","Gd","Sm","Nd","Pr","Ce","La","Ac","Am","Cm","Bk","Cf","Pu","Np","U","Th","Pa","Sc","Zr","Hf","Ti","Nb","Ta","V","Cr","Mo","W","Re","Tc","Os","Ru","Ir","Rh","Pt","Pd","Au","Ag","Cu","Ni","Co","Fe","Mn","Mg","Zn","Cd","Hg","Be","Al","Ga","In","Tl","Pb","Sn","Ge","Si","B","C","N","P","As","Sb","Bi","Po","Te","Se","S","O","I","Br","Cl","F","H"]

Pm = ["Place holder, even He is diff with empty", "He","Ne","Ar","Kr","Xe","Rn","Fr","Cs","Rb","K","Na","Li","Ra","Ba","Sr","Ca","Eu","Yb","Lu","Tm","Y","Er","Ho","Dy","Tb","Gd","Sm","Pm","Nd","Pr","Ce","La","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Sc","Zr","Hf","Ti","Ta","Nb","V","Cr","Mo","W","Re","Tc","Os","Ru","Ir","Rh","Pt","Pd","Au","Ag","Cu","Ni","Co","Fe","Mn","Mg","Zn","Cd","Hg","Be","Al","Ga","In","Tl","Pb","Sn","Ge","Si","B","C","N","P","As","Sb","Bi","Po","Te","Se","S","O","At","I","Br","Cl","F","H"]

ngrid = 8
kgauss = 32.515

def index_3d_to_1d(index_3d,nmax):
    index_1d=index_3d[0]*nmax**2+index_3d[1]*nmax+index_3d[2]
    return index_1d

def index_1d_to_3d(index_1d,nmax):
    index_3d=np.zeros((3),dtype=int)
    index_3d[0]=int(index_1d/nmax**2)
    index_3d[1]=int((index_1d-index_3d[0]*nmax**2)/nmax)
    index_3d[2]=index_1d-index_3d[0]*nmax**2-index_3d[1]*nmax
    return index_3d

def get_grid(ngrid):
    x=[]; y=[]; z=[]
    for ix in range(ngrid):
        xcoord = ix / ngrid
        for iy in range(ngrid):
            ycoord = iy / ngrid
            for iz in range(ngrid):
                zcoord = iz / ngrid
                x.append(xcoord); y.append(ycoord); z.append(zcoord) 
    return x,y,z

def compute_length(axis_val):
    lr = LinearRegression()
    axis_val[0:16] = -axis_val[0:16]
    lr = LinearRegression().fit(np.arange(-16,16).reshape(-1,1)/16, axis_val)
    return lr.coef_[0]

def compute_angle(ri,rj,rij):
    cos_theta = (ri**2 + rj**2 - rij**2) / (2*ri*rj)
    theta = np.arccos(-cos_theta) * 180/np.pi # angle in deg.
    return theta

@numba.jit(nopython=True, parallel=False)
def fast_sum(img):
    ret = np.sum(img, axis=0)
    return ret

@numba.jit(nopython=True, parallel=False)
def atom2img_lst(Z, frac_coords):
    natom = len(Z)
    img = np.zeros((natom, ngrid,ngrid,ngrid), dtype=np.complex128)
    nmid = int(ngrid/2)
    sorted_indices = np.argsort(Z)
    sorted_Z = Z[sorted_indices]
    sorted_frac_coords = frac_coords[sorted_indices]
    for iatom in range(natom):
        fi = sorted_Z[iatom]
        xi, yi, zi = sorted_frac_coords[iatom]
        for ht, h in enumerate(range(-nmid,nmid)):
            for kt, k in enumerate(range(-nmid,nmid)):
                for lt, l in enumerate(range(-nmid,nmid)):
                    # img[ht, kt, lt] += fi * np.exp(-(h**2+k**2+l**2)/ngrid) * np.exp(2 * np.pi * 1j * (h * xi + k * yi + l * zi))
                    img[iatom, ht, kt, lt] += fi * np.exp(2 * np.pi * 1j * (h * xi + k * yi + l * zi))
    return img

@numba.jit(nopython=True, parallel=False)
def atom_points2img(Z, frac_coords):
    natom = len(Z)
    img = np.zeros((ngrid,ngrid,ngrid), dtype=np.complex128)
    nmid = int(ngrid/2)

    for ht, h in enumerate(range(-nmid,nmid)):
        for kt, k in enumerate(range(-nmid,nmid)):
            for lt, l in enumerate(range(-nmid,nmid)):

                for iatom in range(natom):
                    fi = Z[iatom]
                    xi, yi, zi = frac_coords[iatom]
                    # img[ht, kt, lt] += fi * np.exp(-(h**2+k**2+l**2)/ngrid) * np.exp(2 * np.pi * 1j * (h * xi + k * yi + l * zi))
                    img[ht, kt, lt] += np.exp(2 * np.pi * 1j * (h * xi + k * yi + l * zi))
    return img/natom

@numba.jit(nopython=True, parallel=False)
def atom2img(Z, frac_coords):
    natom = len(Z)
    img = np.zeros((ngrid,ngrid,ngrid), dtype=np.complex128)
    nmid = int(ngrid/2)

    for ht, h in enumerate(range(-nmid,nmid)):
        for kt, k in enumerate(range(-nmid,nmid)):
            for lt, l in enumerate(range(-nmid,nmid)):

                for iatom in range(natom):
                    fi = Z[iatom]
                    xi, yi, zi = frac_coords[iatom]
                    # img[ht, kt, lt] += fi * np.exp(-(h**2+k**2+l**2)/ngrid) * np.exp(2 * np.pi * 1j * (h * xi + k * yi + l * zi))
                    img[ht, kt, lt] += fi * np.exp(2 * np.pi * 1j * (h * xi + k * yi + l * zi))
    return img

def error_atom(Z, frac_coords, img):
    return np.sum(
        np.abs(img - atom2img(Z, frac_coords.reshape(len(Z),3)))**2)

def error_latt(latt_params, img):
    guess_lattice = Lattice.from_parameters(*latt_params)
    return np.sum(
        np.abs(img - latt2img(guess_lattice.reciprocal_lattice_crystallographic.metric_tensor))**2)

def guess_atom(img, verbose = False):
    nmid = int(ngrid/2)
    density = np.fft.ifftn(img)

    iatom = 0
    
    indexes = index_1d_to_3d(np.argsort(np.abs(density.flatten()))[-1], ngrid)
    guess_frac_coords = np.array([[(1-indexes[ii]/ngrid)%1 for ii in range(3)]])
    guess_Z = np.array([np.abs(density)[indexes[0], indexes[1], indexes[2]]*kgauss])
    res=minimize(lambda x: error_atom(guess_Z, x, img),
                 guess_frac_coords[iatom].flatten(), tol=1e-8)
    guess_frac_coords[iatom] = res.x
    
    guess_density = density - np.fft.ifftn(atom2img(np.array(guess_Z), np.array(guess_frac_coords)))
    if verbose:
        print("atomic positions detected:")
        print("%3d %7.3f %9.6f %9.6f %9.6f" % (0, guess_Z[0], guess_frac_coords[0][0], guess_frac_coords[0][1], guess_frac_coords[0][2]))
        print("maximum of remaining density:", np.max(np.abs(guess_density.flatten()))*kgauss)
        print()

    while np.max(np.abs(guess_density.flatten()))*kgauss>0.5:
        iatom += 1
        indexes = index_1d_to_3d(np.argsort(np.abs(guess_density.flatten()))[-1], ngrid)
        guess_frac_coords = np.append(guess_frac_coords, np.array(
            [[(1-indexes[ii]/ngrid)%1 for ii in range(3)]])).reshape(iatom+1, 3)
        guess_Z = np.append(guess_Z, np.array(
            [np.abs(guess_density)[indexes[0], indexes[1], indexes[2]]*kgauss]))
        res=minimize(lambda x: error_atom(guess_Z, np.append(guess_frac_coords[0:iatom],x), img),
                     guess_frac_coords[iatom].flatten(), tol=1e-3)
        guess_frac_coords[iatom] = (res.x).copy()
        res=minimize(lambda x: error_atom(x, guess_frac_coords.flatten(), img),
                     guess_Z, tol=1e-3)
        guess_Z = (res.x).copy()
        guess_density = density - np.fft.ifftn(atom2img(np.array(guess_Z), np.array(guess_frac_coords)))
        if verbose:
            print("atomic positions detected:")
            for ii in range(iatom+1):
                print("%3d %7.3f %9.6f %9.6f %9.6f" % (ii, guess_Z[ii], guess_frac_coords[ii][0], guess_frac_coords[ii][1], guess_frac_coords[ii][2]))
            print("maximum of remaining density:", np.max(np.abs(guess_density.flatten()))*kgauss)
            print()

    res=minimize(lambda x: error_atom(x, guess_frac_coords.flatten(), img),
                 guess_Z,tol=1e-8)
    guess_Z = np.round(res.x).astype(int)        

    return guess_Z, guess_frac_coords

def refine_atom(current_Z, current_frac_coords, img, verbose = False):
    natom = len(current_Z)

    new_frac_coords = current_frac_coords.copy()
    for iat in range(natom):
        if verbose:
            print("optimizing atomic position # ",iat)
        if iat == 0:
            res=minimize(
                lambda x: error_atom(current_Z, np.append(x,new_frac_coords[1:].flatten()),img),
                new_frac_coords[iat].flatten(), tol=1e-8)
        elif iat == natom-1:
            res=minimize(
                lambda x: error_atom(current_Z, np.append(new_frac_coords[0:natom-1].flatten(),x),img),
                new_frac_coords[iat].flatten(), tol=1e-8)
        else:
            res=minimize(
                lambda x: error_atom(current_Z, np.append(new_frac_coords[0:iat].flatten(),np.append(x,new_frac_coords[iat+1:].flatten())),img),
                new_frac_coords[iat].flatten(), tol=1e-8)
        new_frac_coords[iat] = res.x
        if verbose:
            error = res.fun
            print("error:", error)
        
    return new_frac_coords

@numba.jit(nopython=True, parallel=False)
def latt2img(gstar):
    #### because numba.jit can not handle lattice directly, so... copy some code from d_hkl here... ####
    img = np.zeros((ngrid, ngrid, ngrid))
    nmid = int(ngrid/2)
    for ht, h in enumerate(range(-nmid,nmid)):
        for kt, k in enumerate(range(-nmid,nmid)):
            for lt, l in enumerate(range(-nmid,nmid)):
                if h==0 and k==0 and l==0:
                    img[ht, kt, lt] = (1/(np.dot(gstar[0],np.cross(gstar[1],gstar[2]))))**(1/6)
                else:
                    hkl = np.array([float(h), float(k), float(l)])
                    img[ht, kt, lt] = 1 / ((np.dot(np.dot(hkl, gstar), hkl.T)) ** (1 / 2))
    return img

def img2atom(img, error_tol=1e-04, coords_tol=1e-05, verbose = False):
    guess_Z, guess_frac_coords = guess_atom(img, verbose = verbose)
    error = error_atom(guess_Z, guess_frac_coords.flatten(), img)
    coords_diff = 1
    if verbose:
        print("error:", error)
    while error > error_tol and coords_diff > coords_tol:
        new_frac_coords = refine_atom(guess_Z, guess_frac_coords, img, verbose = verbose)
        guess_frac_coords = new_frac_coords.copy()
        error = error_atom(guess_Z, guess_frac_coords.flatten(), img)
        coords_diff = np.max(guess_frac_coords-new_frac_coords)
        if verbose:
            print("error:", error, "change in the positions:", coord_diff)
    return guess_Z, guess_frac_coords

def img2latt(img):
    img_center = int(ngrid/2)
    ra = img[img_center,img_center,img_center]
    rb = ra
    rc = ra
    alpha = 90
    beta = 90
    gamma = 90
    res=minimize(lambda x: error_latt(x,img), (ra,rb,rc,alpha,beta,gamma), tol=1e-8)
    print()
    return tuple(res.x[ii] for ii in range(6))

def img2struc(img_atom, img_latt):
    latt_params = img2latt(img_latt)
    lattice = Lattice.from_parameters(*latt_params)
    Z, frac_coords = img2atom(img_atom)
    symbols = [Element.from_Z(Z[i]).symbol for i in np.arange(len(Z))]
    return Structure(lattice, symbols, frac_coords)

class Crystal(object):
    def __init__(self, crystal_structure):
        self.structure = crystal_structure
        self.natom = len(self.structure.sites)
        self.pt1d = GA
        frac_coords = []
        Z = []
        for site in self.structure:
            for sp, occu in site.species.items():
                frac_coords.append(site.frac_coords)
                Z.append(self.pt1d.index(sp.symbol)) # sp.Z)

        self.frac_coords = np.array(frac_coords)
        self.Z = np.array(Z)

        self.grid_atom = get_grid(ngrid)
        self.grid_latt = get_grid(ngrid)

        self.img_atom_lst = atom2img_lst(self.Z, self.frac_coords)
        self.img_atom = fast_sum(self.img_atom_lst)
        self.img_atom_avg = atom_points2img(self.Z, self.frac_coords) # fast_sum(self.img_atom_lst)
        self.img_latt = latt2img(self.structure.lattice.reciprocal_lattice_crystallographic.metric_tensor)
        
    def show_struc(self):
        view = quick_view(self.structure)
        display(view)
        return
    
    def show_img_atom_3D(self, isomin = 0.2):
        x = self.grid_atom[0]; y = self.grid_atom[1];  z = self.grid_atom[2];
        fig = go.Figure(
            data = go.Isosurface(
                x = x, y = y, z = z,
                value = self.img_atom.flatten(),
                isomin = isomin
                )
            )
        fig.show()
        return
    
    def show_img_atom_2D(self, zmin=0, zmax=1):
        nmid = int(ngrid/2)
        cuts = np.zeros((3, ngrid, ngrid))
        cuts[0,:,:] = np.abs(self.img_atom[nmid,:,:])
        cuts[1,:,:] = np.abs(self.img_atom[:,nmid,:])
        cuts[2,:,:] = np.abs(self.img_atom[:,:,nmid])

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('x', 'y', 'z'))
        for ii in range(3):
            showscale = True
            if ii != 0:
                showscale = False
            fig.add_trace(go.Heatmap(
                z=np.abs(cuts[ii,:,:]), zmin=0, zmax=1,
                showscale = showscale, colorbar=dict(orientation="h")),
                ii+1, 1)
        fig.update_layout(width=400, height=1200)
        fig.show()
        return

    def show_img_latt_3D(self, isomin = 0.5):
        x = self.grid_latt[0]; y = self.grid_latt[1];  z = self.grid_latt[2];
        fig = go.Figure(
            data = go.Isosurface(
            x = x, y = y, z = z,
            value = self.img_latt.flatten(),
            isomin = isomin
            )
        )
        fig.show()
        return

    def show_img_latt_2D(self, zmin=0, zmax=1):
        ngrid = self.img_latt.shape[0]
        nmid = int(ngrid/2)
        cuts = np.zeros((3, ngrid, ngrid))
        cuts[0,:,:] = self.img_latt[nmid,:,:]
        cuts[1,:,:] = self.img_latt[:,nmid,:]
        cuts[2,:,:] = self.img_latt[:,:,nmid] 
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('x', 'y', 'z'))
        for ii in range(3):
            showscale = True
            if ii != 0:
                showscale = False
            fig.add_trace(go.Heatmap(
                z=cuts[ii,:,:], zmin=zmin, zmax=zmax,
                showscale = showscale, colorbar=dict(orientation="h")),
                ii+1, 1)
        fig.update_layout(width=400, height=1200)
        fig.show()
        return

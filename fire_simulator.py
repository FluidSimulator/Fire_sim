"""
fire_simulator.py
═════════════════
2D Eulerian Fire Simulator with ML-Accelerated Pressure Solver
and ML Fire Spread Classifier — SINGLE FILE EXECUTABLE.

Requirements: pip install taichi numpy
Run:          python fire_simulator.py

Controls:
  Click+drag    Move the red sphere obstacle
  Space         Reset simulation
  ESC           Quit

ML Systems:
  1. Neural Pressure Solver  – MLP regression (9→32→16→1)
  2. Fire Spread Classifier  – MLP classification (20→32→3, softmax)
"""

import time
import numpy as np
import taichi as ti
from numpy.lib.stride_tricks import sliding_window_view

# ═════════════════════════════════════════════════════════════════════════════
#  1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════
GRID_W, GRID_H   = 128, 256
DISPLAY_SCALE     = 2
DT                = 0.032
SOURCE_RADIUS     = 30
SOURCE_DENSITY_STR = 6.0
SOURCE_TEMP_STR   = 18.0
SOURCE_NOISE      = 0.50
BUOYANCY_STRENGTH = 2.8
VORTICITY_STRENGTH = 0.50
DISSIPATION_DENSITY = 0.975
DISSIPATION_TEMP  = 0.993
COOLING_FIRE      = 0.978
COOLING_SMOKE     = 0.970
FIRE_THRESHOLD    = 1.5
JACOBI_ITERS      = 30
OVER_RELAXATION   = 1.00
OBSTACLE_CX, OBSTACLE_CY, OBSTACLE_RADIUS = 64, 80, 16
ML_COLLECT_FRAMES = 40
ML_TRAIN_EPOCHS   = 40
ML_JACOBI_AFTER   = 15
ML_FIRE_COLLECT   = 60
ML_FIRE_EPOCHS    = 50
ML_FIRE_INFER_EVERY = 8

# ═════════════════════════════════════════════════════════════════════════════
#  2. TAICHI INIT + FIELDS
# ═════════════════════════════════════════════════════════════════════════════
ti.init(arch=ti.cpu, cpu_max_num_threads=16, default_fp=ti.f32, fast_math=True)

W, H = GRID_W, GRID_H
vel_x     = ti.field(ti.f32, shape=(W+1, H))
vel_y     = ti.field(ti.f32, shape=(W, H+1))
vel_x_tmp = ti.field(ti.f32, shape=(W+1, H))
vel_y_tmp = ti.field(ti.f32, shape=(W, H+1))
pressure  = ti.field(ti.f32, shape=(W, H))
pressure_b = ti.field(ti.f32, shape=(W, H))
divergence = ti.field(ti.f32, shape=(W, H))
density    = ti.field(ti.f32, shape=(W, H))
density_tmp = ti.field(ti.f32, shape=(W, H))
temperature = ti.field(ti.f32, shape=(W, H))
temperature_tmp = ti.field(ti.f32, shape=(W, H))
curl       = ti.field(ti.f32, shape=(W, H))
obstacle   = ti.field(ti.f32, shape=(W, H))
pixels     = ti.Vector.field(3, ti.f32, shape=(W, H))
ml_class   = ti.Vector.field(3, ti.f32, shape=(W, H))
obs_cx_f   = ti.field(ti.f32, shape=())
obs_cy_f   = ti.field(ti.f32, shape=())
obs_r_f    = ti.field(ti.f32, shape=())

_DT, _BUOY, _VORT = DT, BUOYANCY_STRENGTH, VORTICITY_STRENGTH
_DISS_D, _DISS_T = DISSIPATION_DENSITY, DISSIPATION_TEMP
_COOL_F, _COOL_S, _FTHR = COOLING_FIRE, COOLING_SMOKE, FIRE_THRESHOLD
_SOR, _NOISE = OVER_RELAXATION, SOURCE_NOISE

# ═════════════════════════════════════════════════════════════════════════════
#  3. TAICHI KERNELS
# ═════════════════════════════════════════════════════════════════════════════
@ti.kernel
def init_obstacle_circle(cx: int, cy: int, radius: int):
    for i, j in ti.ndrange(W, H):
        if (float(i)-float(cx))**2 + (float(j)-float(cy))**2 < float(radius*radius):
            obstacle[i, j] = 1.0

@ti.kernel
def clear_obstacle():
    for i, j in ti.ndrange(W, H):
        obstacle[i, j] = 0.0

@ti.func
def bilerp_vx(px: float, py: float) -> float:
    xi = int(ti.floor(px)); yf = py - 0.5; yi = int(ti.floor(yf))
    tx = px - ti.floor(px); ty = yf - ti.floor(yf)
    xi = ti.max(0, ti.min(xi, W-1)); yi = ti.max(0, ti.min(yi, H-2))
    return (1-tx)*(1-ty)*vel_x[xi,yi] + tx*(1-ty)*vel_x[xi+1,yi] + (1-tx)*ty*vel_x[xi,yi+1] + tx*ty*vel_x[xi+1,yi+1]

@ti.func
def bilerp_vy(px: float, py: float) -> float:
    xf = px - 0.5; xi = int(ti.floor(xf)); yi = int(ti.floor(py))
    tx = xf - ti.floor(xf); ty = py - ti.floor(py)
    xi = ti.max(0, ti.min(xi, W-2)); yi = ti.max(0, ti.min(yi, H-1))
    return (1-tx)*(1-ty)*vel_y[xi,yi] + tx*(1-ty)*vel_y[xi+1,yi] + (1-tx)*ty*vel_y[xi,yi+1] + tx*ty*vel_y[xi+1,yi+1]

@ti.func
def bilerp_cell(field: ti.template(), px: float, py: float) -> float:
    xf = px-0.5; yf = py-0.5; xi = int(ti.floor(xf)); yi = int(ti.floor(yf))
    tx = xf-ti.floor(xf); ty = yf-ti.floor(yf)
    xi = ti.max(0, ti.min(xi, W-2)); yi = ti.max(0, ti.min(yi, H-2))
    return (1-tx)*(1-ty)*field[xi,yi] + tx*(1-ty)*field[xi+1,yi] + (1-tx)*ty*field[xi,yi+1] + tx*ty*field[xi+1,yi+1]

@ti.func
def vel_at_center(i: int, j: int) -> ti.Vector:
    return ti.Vector([0.5*(vel_x[i,j]+vel_x[i+1,j]), 0.5*(vel_y[i,j]+vel_y[i,j+1])])

@ti.kernel
def advect_velocity():
    for i, j in ti.ndrange((1, W), H):
        cx, cy = float(i), float(j)+0.5
        u = vel_x[i,j]
        v = 0.25*(vel_y[i-1,j]+vel_y[i-1,j+1]+vel_y[i,j]+vel_y[i,j+1])
        px = ti.max(1.0, ti.min(cx-_DT*u, float(W)-1.0))
        py = ti.max(0.5+1e-4, ti.min(cy-_DT*v, float(H)-0.5-1e-4))
        vel_x_tmp[i,j] = bilerp_vx(px, py)
    for i, j in ti.ndrange(W, (1, H)):
        cx, cy = float(i)+0.5, float(j)
        u = 0.25*(vel_x[i,j-1]+vel_x[i+1,j-1]+vel_x[i,j]+vel_x[i+1,j])
        v = vel_y[i,j]
        px = ti.max(0.5+1e-4, ti.min(cx-_DT*u, float(W)-0.5-1e-4))
        py = ti.max(1.0, ti.min(cy-_DT*v, float(H)-1.0))
        vel_y_tmp[i,j] = bilerp_vy(px, py)

@ti.kernel
def swap_velocity_buffers():
    for i, j in ti.ndrange(W+1, H):
        vel_x[i,j] = vel_x_tmp[i,j]; vel_x_tmp[i,j] = 0.0
    for i, j in ti.ndrange(W, H+1):
        vel_y[i,j] = vel_y_tmp[i,j]; vel_y_tmp[i,j] = 0.0

@ti.kernel
def advect_and_cool_scalars():
    for i, j in ti.ndrange(W, H):
        cx, cy = float(i)+0.5, float(j)+0.5
        uv = vel_at_center(i, j)
        px = ti.max(0.5+1e-4, ti.min(cx-_DT*uv[0], float(W)-0.5-1e-4))
        py = ti.max(0.5+1e-4, ti.min(cy-_DT*uv[1], float(H)-0.5-1e-4))
        d = bilerp_cell(density, px, py) * _DISS_D
        t = bilerp_cell(temperature, px, py) * _DISS_T
        t *= ti.select(t > _FTHR, _COOL_F, _COOL_S)
        heat_ratio = ti.min(1.0, t / (_FTHR + 0.5))
        d *= 0.96 + 0.04 * heat_ratio
        if d < 0.01: d = 0.0
        if t < 0.005: t = 0.0
        density_tmp[i,j] = d; temperature_tmp[i,j] = t

@ti.kernel
def swap_scalar_buffers():
    for i, j in ti.ndrange(W, H):
        density[i,j] = density_tmp[i,j]; temperature[i,j] = temperature_tmp[i,j]

@ti.kernel
def apply_buoyancy_force():
    for i, j in ti.ndrange(W, (1, H)):
        vel_y[i,j] += _BUOY * 0.5 * (temperature[i,j-1]+temperature[i,j]) * _DT

@ti.kernel
def compute_curl_field():
    for i, j in ti.ndrange((1, W-1), (1, H-1)):
        curl[i,j] = 0.5*(vel_y[i+1,j]-vel_y[i-1,j]) - 0.5*(vel_x[i,j+1]-vel_x[i,j-1])

@ti.kernel
def apply_vorticity_confinement():
    for i, j in ti.ndrange((1, W-1), (1, H-1)):
        eta_x = 0.5*(ti.abs(curl[i+1,j])-ti.abs(curl[i-1,j]))
        eta_y = 0.5*(ti.abs(curl[i,j+1])-ti.abs(curl[i,j-1]))
        mag = ti.sqrt(eta_x*eta_x+eta_y*eta_y)+1e-6
        omega = curl[i,j]
        fx = _VORT*_DT*(eta_y/mag)*omega; fy = -_VORT*_DT*(eta_x/mag)*omega
        vel_x[i,j] += fx*0.5; vel_x[i+1,j] += fx*0.5
        vel_y[i,j] += fy*0.5; vel_y[i,j+1] += fy*0.5

@ti.kernel
def compute_divergence():
    for i, j in ti.ndrange(W, H):
        divergence[i,j] = vel_x[i+1,j]-vel_x[i,j]+vel_y[i,j+1]-vel_y[i,j]

@ti.kernel
def jacobi_iteration():
    for i, j in ti.ndrange((1, W-1), (1, H-1)):
        p_sum = pressure[i-1,j]+pressure[i+1,j]+pressure[i,j-1]+pressure[i,j+1]
        p_new = (p_sum-divergence[i,j])*0.25
        pressure_b[i,j] = pressure[i,j]+_SOR*(p_new-pressure[i,j])

@ti.kernel
def swap_pressure_buffers():
    for i, j in ti.ndrange(W, H): pressure[i,j] = pressure_b[i,j]

@ti.kernel
def subtract_pressure_gradient():
    for i, j in ti.ndrange((1, W), H): vel_x[i,j] -= pressure[i,j]-pressure[i-1,j]
    for i, j in ti.ndrange(W, (1, H)): vel_y[i,j] -= pressure[i,j]-pressure[i,j-1]

@ti.kernel
def enforce_boundary_conditions():
    for j in range(H): vel_x[0,j]=0.0; vel_x[W,j]=0.0
    for i in range(W): vel_y[i,0]=0.0; vel_y[i,H]=0.0
    for i, j in ti.ndrange(W, H):
        if obstacle[i,j] > 0.5:
            density[i,j]=0.0; temperature[i,j]=0.0; pressure[i,j]=0.0
            vel_x[i,j]=0.0; vel_x[i+1,j]=0.0; vel_y[i,j]=0.0; vel_y[i,j+1]=0.0
    for i, j in ti.ndrange(W, H):
        if obstacle[i,j] < 0.5:
            if i > 0 and obstacle[i-1,j] > 0.5: vel_x[i,j] = vel_x[i+1,j] if i+1<=W else 0.0
            if i < W-1 and obstacle[i+1,j] > 0.5: vel_x[i+1,j] = vel_x[i,j]
            if j > 0 and obstacle[i,j-1] > 0.5: vel_y[i,j] = vel_y[i,j+1] if j+1<=H else 0.0
            if j < H-1 and obstacle[i,j+1] > 0.5: vel_y[i,j+1] = vel_y[i,j]

@ti.kernel
def inject_fire(cx: int, cy: int, radius: int, d_str: float, t_str: float):
    for i, j in ti.ndrange(W, H):
        if obstacle[i,j] < 0.5:
            dx = float(i)-float(cx); dy = float(j)-float(cy)
            dist = ti.sqrt(dx*dx+dy*dy)
            if dist < float(radius):
                f = 1.0-dist/float(radius); f = f*f
                n = 1.0+_NOISE*(ti.random(ti.f32)*2.0-1.0)
                s = f*n*_DT*30.0
                density[i,j] += d_str*s; temperature[i,j] += t_str*s
                vel_y[i,j+1] += t_str*0.055*f*_DT*30.0

@ti.kernel
def clamp_velocity():
    for i, j in ti.ndrange(W+1, H): vel_x[i,j] = ti.max(-100.0, ti.min(vel_x[i,j], 100.0))
    for i, j in ti.ndrange(W, H+1): vel_y[i,j] = ti.max(-100.0, ti.min(vel_y[i,j], 100.0))

@ti.kernel
def reset_pressure():
    for i, j in ti.ndrange(W, H): pressure[i,j]=0.0; pressure_b[i,j]=0.0

@ti.kernel
def reset_all_fields():
    for i, j in ti.ndrange(W+1, H): vel_x[i,j]=0.0; vel_x_tmp[i,j]=0.0
    for i, j in ti.ndrange(W, H+1): vel_y[i,j]=0.0; vel_y_tmp[i,j]=0.0
    for i, j in ti.ndrange(W, H):
        pressure[i,j]=0.0; pressure_b[i,j]=0.0; divergence[i,j]=0.0
        density[i,j]=0.0; density_tmp[i,j]=0.0
        temperature[i,j]=0.0; temperature_tmp[i,j]=0.0; curl[i,j]=0.0

# ── Renderer ──────────────────────────────────────────────────────────────────
@ti.func
def fire_color_ramp(t: float) -> ti.Vector:
    r, g, b = 0.0, 0.0, 0.0
    if t > 0.05:
        r = ti.min(1.0, t*2.0)
        g = ti.min(0.55, ti.max(0.0, (t-0.5)*0.30))
        b = ti.min(0.15, ti.max(0.0, (t-3.5)*0.08))
        if t > 6.0:
            boost = ti.min(1.0, (t-6.0)*0.12)
            g += (0.85-g)*boost; b += (0.5-b)*boost*0.3
    return ti.Vector([r, g, b])

@ti.kernel
def render_pixels():
    ocx=obs_cx_f[None]; ocy=obs_cy_f[None]; orad=obs_r_f[None]
    for i, j in pixels:
        if obstacle[i,j] > 0.5:
            dx=float(i)-ocx; dy=float(j)-ocy; dist=ti.sqrt(dx*dx+dy*dy)
            nd = dist/ti.max(orad,1.0)
            if nd < 1.0:
                nx=dx/ti.max(orad,1.0); ny=dy/ti.max(orad,1.0)
                light=ti.max(0.0,-0.5*nx+0.6*ny+0.5); spec=light*light*light*0.25
                shade=0.3+0.7*light; edge=1.0-nd*nd; mix=edge*0.7+0.3
                pixels[i,j]=ti.Vector([ti.min(1.0,0.85*shade*mix+spec), ti.min(1.0,0.12*shade*mix+spec*0.2), ti.min(1.0,0.10*shade*mix+spec*0.15)])
            else:
                pixels[i,j]=ti.Vector([0.03,0.03,0.04])
        else:
            t_val=temperature[i,j]; d_val=density[i,j]
            ml=ml_class[i,j]; p_fire=ml[1]; p_smoke=ml[2]; ml_on=(p_fire+p_smoke)>0.01
            fc=fire_color_ramp(t_val); fi=ti.min(1.0,t_val*0.45)
            hr=ti.min(1.0,t_val/(_FTHR+0.5)); sf=1.0-hr
            r,g,b = 0.0,0.0,0.0
            if ml_on:
                pf=ti.max(p_fire,0.3); r=fc[0]*fi*pf; g=fc[1]*fi*pf; b=fc[2]*fi*pf
                so=ti.min(0.30,p_smoke*0.45)*sf; r+=0.12*so; g+=0.10*so; b+=0.08*so
            else:
                so=ti.min(0.35,d_val*0.06)*sf
                r=0.12*so+fc[0]*fi; g=0.10*so+fc[1]*fi; b=0.08*so+fc[2]*fi
            pixels[i,j]=ti.Vector([ti.max(0.0,ti.min(r,1.0)), ti.max(0.0,ti.min(g,1.0)), ti.max(0.0,ti.min(b,1.0))])

# ═════════════════════════════════════════════════════════════════════════════
#  4. ML — NEURAL PRESSURE SOLVER (regression, MSE, backprop, Adam)
# ═════════════════════════════════════════════════════════════════════════════
class NeuralPressureSolver:
    def __init__(self, ps=3, h1=32, h2=16, lr=0.001):
        self.ps, self.pad = ps, ps//2
        n = ps*ps
        self.W1=(np.random.randn(n,h1)*np.sqrt(2.0/n)).astype(np.float32); self.b1=np.zeros(h1,dtype=np.float32)
        self.W2=(np.random.randn(h1,h2)*np.sqrt(2.0/h1)).astype(np.float32); self.b2=np.zeros(h2,dtype=np.float32)
        self.W3=(np.random.randn(h2,1)*np.sqrt(2.0/h2)).astype(np.float32); self.b3=np.zeros(1,dtype=np.float32)
        self.lr=lr; self.am={}; self.av={}; self.at=0
        self.X_mean=None; self.X_std=None; self.y_mean=0.0; self.y_std=1.0
        self.X_data=[]; self.y_data=[]; self.trained=False
    def _au(self,n,p,g):
        if n not in self.am: self.am[n]=np.zeros_like(p); self.av[n]=np.zeros_like(p)
        self.at+=1; self.am[n]=0.9*self.am[n]+0.1*g; self.av[n]=0.999*self.av[n]+0.001*g**2
        return p-self.lr*(self.am[n]/(1-0.9**self.at))/(np.sqrt(self.av[n]/(1-0.999**self.at))+1e-8)
    def _fwd(self,X):
        self._z1=X@self.W1+self.b1; self._a1=np.maximum(0,self._z1)
        self._z2=self._a1@self.W2+self.b2; self._a2=np.maximum(0,self._z2)
        return self._a2@self.W3+self.b3
    def _fwd_fast(self,X):
        return np.maximum(0,np.maximum(0,X@self.W1+self.b1)@self.W2+self.b2)@self.W3+self.b3
    def collect(self,div_np,pres_np,ns=1500):
        dh,ph=div_np.T.astype(np.float32),pres_np.T.astype(np.float32)
        H2,W2=dh.shape; pad=np.pad(dh,self.pad,mode='constant')
        ri,rj=np.random.randint(0,H2,ns),np.random.randint(0,W2,ns)
        ps=self.ps; P=np.empty((ns,ps*ps),dtype=np.float32); T=np.empty((ns,1),dtype=np.float32)
        for k in range(ns): P[k]=pad[ri[k]:ri[k]+ps,rj[k]:rj[k]+ps].ravel(); T[k,0]=ph[ri[k],rj[k]]
        self.X_data.append(P); self.y_data.append(T)
    def train(self,epochs=40,bs=256):
        print("\n  [ML] Training Pressure Solver...")
        t0=time.perf_counter(); X=np.vstack(self.X_data).astype(np.float32); y=np.vstack(self.y_data).astype(np.float32); N=len(X)
        self.X_mean=X.mean(0); self.X_std=X.std(0)+1e-8; self.y_mean=y.mean(); self.y_std=y.std()+1e-8
        Xn=(X-self.X_mean)/self.X_std; yn=(y-self.y_mean)/self.y_std
        for ep in range(epochs):
            perm=np.random.permutation(N); tl=0.0; nb=0
            for s in range(0,N,bs):
                idx=perm[s:s+bs]; Xb,yb=Xn[idx],yn[idx]; b=len(Xb)
                pred=self._fwd(Xb); diff=pred-yb; tl+=float(np.mean(diff**2)); nb+=1
                dl=(2.0/b)*diff; dW3=self._a2.T@dl; db3=dl.sum(0)
                da2=dl@self.W3.T*(self._z2>0); dW2=self._a1.T@da2; db2=da2.sum(0)
                da1=da2@self.W2.T*(self._z1>0); dW1=Xb.T@da1; db1=da1.sum(0)
                self.W1=self._au('W1',self.W1,dW1); self.b1=self._au('b1',self.b1,db1)
                self.W2=self._au('W2',self.W2,dW2); self.b2=self._au('b2',self.b2,db2)
                self.W3=self._au('W3',self.W3,dW3); self.b3=self._au('b3',self.b3,db3)
            if ep%10==0 or ep==epochs-1: print(f"    Epoch {ep:3d} Loss: {tl/max(nb,1):.6f}")
        self.trained=True; self.X_data.clear(); self.y_data.clear()
        print(f"    Done in {time.perf_counter()-t0:.1f}s — Pressure Solver ACTIVE\n")
    def predict(self,div_np):
        dh=div_np.T.astype(np.float32); H2,W2=dh.shape
        pad=np.pad(dh,self.pad,mode='constant'); ps=self.ps
        w=sliding_window_view(pad,(ps,ps)); P=np.ascontiguousarray(w.reshape(H2*W2,ps*ps),dtype=np.float32)
        Xn=(P-self.X_mean)/self.X_std; pn=self._fwd_fast(Xn)
        return (pn.ravel()*self.y_std+self.y_mean).reshape(H2,W2).T.astype(np.float32)
    def status(self):
        if self.trained: return "PressML-ON"
        if self.X_data: return f"PressML-col({len(self.X_data)})"
        return "PressML-idle"

# ═════════════════════════════════════════════════════════════════════════════
#  5. ML — FIRE SPREAD CLASSIFIER (softmax, cross-entropy, classification)
# ═════════════════════════════════════════════════════════════════════════════
class FireClassifier:
    def __init__(self, h=32, lr=0.002):
        n_in=20; n_out=3
        self.W1=(np.random.randn(n_in,h)*np.sqrt(2.0/n_in)).astype(np.float32); self.b1=np.zeros(h,dtype=np.float32)
        self.W2=(np.random.randn(h,n_out)*np.sqrt(2.0/h)).astype(np.float32); self.b2=np.zeros(n_out,dtype=np.float32)
        self.lr=lr; self.am={}; self.av={}; self.at=0
        self.X_data=[]; self.y_data=[]; self.trained=False
        self.X_mean=None; self.X_std=None
    def _au(self,n,p,g):
        if n not in self.am: self.am[n]=np.zeros_like(p); self.av[n]=np.zeros_like(p)
        self.at+=1; self.am[n]=0.9*self.am[n]+0.1*g; self.av[n]=0.999*self.av[n]+0.001*g**2
        return p-self.lr*(self.am[n]/(1-0.9**self.at))/(np.sqrt(self.av[n]/(1-0.999**self.at))+1e-8)
    @staticmethod
    def _sm(z):
        e=np.exp(z-z.max(1,keepdims=True)); return e/(e.sum(1,keepdims=True)+1e-8)
    def _fwd(self,X):
        self._z1=X@self.W1+self.b1; self._a1=np.maximum(0,self._z1)
        self._z2=self._a1@self.W2+self.b2; self._p=self._sm(self._z2); return self._p
    def _fwd_fast(self,X):
        a1=np.maximum(0,X@self.W1+self.b1); return self._sm(a1@self.W2+self.b2)
    def _feats(self,th,dh,vxh,vyh,ns=None):
        H2,W2=th.shape; tp=np.pad(th,1,mode='constant'); dp=np.pad(dh,1,mode='constant')
        if ns:
            ri,rj=np.random.randint(0,H2,ns),np.random.randint(0,W2,ns)
            F=np.empty((ns,20),dtype=np.float32)
            for k in range(ns):
                i,j=ri[k],rj[k]; F[k,:9]=tp[i:i+3,j:j+3].ravel(); F[k,9:18]=dp[i:i+3,j:j+3].ravel()
                F[k,18]=vxh[i,j] if i<vxh.shape[0] and j<vxh.shape[1] else 0.0
                F[k,19]=vyh[i,j] if i<vyh.shape[0] and j<vyh.shape[1] else 0.0
            return F,ri,rj
        tw=sliding_window_view(tp,(3,3)).reshape(H2*W2,9)
        dw=sliding_window_view(dp,(3,3)).reshape(H2*W2,9)
        vxf=vxh[:H2,:W2].ravel() if vxh.shape[0]>=H2 else np.zeros(H2*W2,dtype=np.float32)
        vyf=vyh[:H2,:W2].ravel() if vyh.shape[0]>=H2 else np.zeros(H2*W2,dtype=np.float32)
        return np.hstack([tw.astype(np.float32),dw.astype(np.float32),vxf.reshape(-1,1).astype(np.float32),vyf.reshape(-1,1).astype(np.float32)])
    def collect(self,t_np,d_np,vx_np,vy_np,ns=2000):
        th=t_np.T.astype(np.float32); dh=d_np.T.astype(np.float32); H2,W2=th.shape
        vxh=(vx_np[:W2,:H2] if vx_np.shape[0]>W2 else vx_np[:,:H2]).T.astype(np.float32)
        vyh=(vy_np[:W2,:H2] if vy_np.shape[1]>H2 else vy_np[:,:H2]).T.astype(np.float32)
        F,ri,rj=self._feats(th,dh,vxh,vyh,ns)
        L=np.empty(ns,dtype=np.int32)
        for k in range(ns): L[k]=1 if th[ri[k],rj[k]]>1.0 else (2 if dh[ri[k],rj[k]]>0.3 else 0)
        self.X_data.append(F); self.y_data.append(L)
    def train(self,epochs=50,bs=512):
        print("\n  [ML] Training Fire Classifier...")
        t0=time.perf_counter(); X=np.vstack(self.X_data).astype(np.float32); y=np.concatenate(self.y_data); N=len(X)
        self.X_mean=X.mean(0); self.X_std=X.std(0)+1e-8; Xn=(X-self.X_mean)/self.X_std
        yoh=np.zeros((N,3),dtype=np.float32); yoh[np.arange(N),y]=1.0
        for c in range(3): print(f"    Class {c} ({'Empty' if c==0 else 'Fire' if c==1 else 'Smoke'}): {(y==c).sum():,}")
        for ep in range(epochs):
            perm=np.random.permutation(N); tl=0.0; nb=0
            for s in range(0,N,bs):
                idx=perm[s:s+bs]; Xb,yb=Xn[idx],yoh[idx]; b=len(Xb)
                probs=self._fwd(Xb); loss=-np.mean(np.sum(yb*np.log(probs+1e-8),1)); tl+=loss; nb+=1
                dl=(probs-yb)/b; dW2=self._a1.T@dl; db2=dl.sum(0)
                da1=dl@self.W2.T*(self._z1>0); dW1=Xb.T@da1; db1=da1.sum(0)
                self.W1=self._au('W1',self.W1,dW1); self.b1=self._au('b1',self.b1,db1)
                self.W2=self._au('W2',self.W2,dW2); self.b2=self._au('b2',self.b2,db2)
            if ep%10==0 or ep==epochs-1: print(f"    Epoch {ep:3d} Loss: {tl/max(nb,1):.4f}")
        self.trained=True; self.X_data.clear(); self.y_data.clear()
        print(f"    Done in {time.perf_counter()-t0:.1f}s — Fire Classifier ACTIVE\n")
    def predict(self,t_np,d_np,vx_np,vy_np):
        th=t_np.T.astype(np.float32); dh=d_np.T.astype(np.float32); H2,W2=th.shape
        vxh=(vx_np[:W2,:H2] if vx_np.shape[0]>W2 else vx_np[:,:H2]).T.astype(np.float32)
        vyh=(vy_np[:W2,:H2] if vy_np.shape[1]>H2 else vy_np[:,:H2]).T.astype(np.float32)
        F=self._feats(th,dh,vxh,vyh); Xn=(F-self.X_mean)/self.X_std
        return np.transpose(self._fwd_fast(Xn).reshape(H2,W2,3),(1,0,2)).astype(np.float32)
    def status(self):
        if self.trained: return "FireML-ON"
        if self.X_data: return f"FireML-col({len(self.X_data)})"
        return "FireML-idle"

# ═════════════════════════════════════════════════════════════════════════════
#  6. SIMULATION LOOP
# ═════════════════════════════════════════════════════════════════════════════
def main():
    win_w, win_h = W*DISPLAY_SCALE, H*DISPLAY_SCALE
    gui = ti.GUI("Fire Sim | Drag: Move Obstacle | Space: Reset | ESC: Quit",
                 res=(win_w, win_h), fast_gui=False)

    reset_all_fields()
    clear_obstacle()
    init_obstacle_circle(OBSTACLE_CX, OBSTACLE_CY, OBSTACLE_RADIUS)

    obs_cx, obs_cy, obs_r = OBSTACLE_CX, OBSTACLE_CY, OBSTACLE_RADIUS
    emitter_x, emitter_y = W//2, 4

    ml_press = NeuralPressureSolver()
    ml_fire  = FireClassifier()
    print(f"  [ML] Pressure Solver: 9->32->16->1 | Fire Classifier: 20->32->3")
    print(f"  Collecting data for ~{ML_COLLECT_FRAMES}/{ML_FIRE_COLLECT} frames...\n")

    frame, t_fps, fps, _s = 0, time.perf_counter(), 0.0, DISPLAY_SCALE

    while gui.running:
        for ev in gui.get_events(ti.GUI.PRESS):
            if ev.key == ti.GUI.ESCAPE: gui.running = False
            elif ev.key == ti.GUI.SPACE:
                reset_all_fields(); clear_obstacle()
                init_obstacle_circle(obs_cx, obs_cy, obs_r); frame = 0

        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            obs_cx = max(obs_r+1, min(int(mx*W), W-obs_r-1))
            obs_cy = max(obs_r+1, min(int(my*H), H-obs_r-1))
            clear_obstacle(); init_obstacle_circle(obs_cx, obs_cy, obs_r)

        # Simulation step
        inject_fire(emitter_x, emitter_y, SOURCE_RADIUS, SOURCE_DENSITY_STR, SOURCE_TEMP_STR)
        apply_buoyancy_force(); compute_curl_field(); apply_vorticity_confinement(); clamp_velocity()
        enforce_boundary_conditions()

        # Pressure solve (ML or Jacobi)
        try:
            if not ml_press.trained:
                reset_pressure(); compute_divergence()
                for _ in range(JACOBI_ITERS): jacobi_iteration(); swap_pressure_buffers()
                subtract_pressure_gradient()
                if frame < ML_COLLECT_FRAMES:
                    ml_press.collect(divergence.to_numpy(), pressure.to_numpy())
                if frame == ML_COLLECT_FRAMES:
                    ml_press.train()
            else:
                reset_pressure(); compute_divergence()
                pred = np.clip(np.nan_to_num(ml_press.predict(divergence.to_numpy())), -50, 50)
                pressure.from_numpy(pred); pressure_b.from_numpy(pred)
                for _ in range(ML_JACOBI_AFTER): jacobi_iteration(); swap_pressure_buffers()
                subtract_pressure_gradient()
        except:
            reset_pressure(); compute_divergence()
            for _ in range(JACOBI_ITERS): jacobi_iteration(); swap_pressure_buffers()
            subtract_pressure_gradient()

        enforce_boundary_conditions(); clamp_velocity()
        advect_velocity(); swap_velocity_buffers(); enforce_boundary_conditions()
        advect_and_cool_scalars(); swap_scalar_buffers()

        # Fire classifier
        try:
            if not ml_fire.trained:
                if frame < ML_FIRE_COLLECT:
                    ml_fire.collect(temperature.to_numpy(), density.to_numpy(),
                                    vel_x.to_numpy(), vel_y.to_numpy())
                if frame == ML_FIRE_COLLECT: ml_fire.train()
            elif frame % ML_FIRE_INFER_EVERY == 0:
                ml_class.from_numpy(ml_fire.predict(
                    temperature.to_numpy(), density.to_numpy(),
                    vel_x.to_numpy(), vel_y.to_numpy()))
        except: pass

        obs_cx_f[None]=float(obs_cx); obs_cy_f[None]=float(obs_cy); obs_r_f[None]=float(obs_r)
        render_pixels()

        gui.set_image(np.repeat(np.repeat(pixels.to_numpy(), _s, axis=0), _s, axis=1))
        gui.show()

        frame += 1
        if frame % 60 == 0:
            now = time.perf_counter(); fps = 60.0/max(now-t_fps, 1e-9); t_fps = now
            st = f"{ml_press.status()} | {ml_fire.status()}"
            gui.title = f"Fire Sim | {fps:.0f} FPS | [{st}]"
            print(f"  Frame {frame:5d}  {fps:5.1f} FPS | {st}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
        input("\n  >>> Press Enter to close... <<<")

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

@dataclass
class WasherSpec:
    expected_outer_mm: Optional[float] = None
    expected_inner_mm: Optional[float] = None
    diameter_tolerance_mm: float = 0.6
    circularity_min: float = 0.85
    concentricity_max_ratio: float = 0.08
    dark_defect_min_area_mm2: float = 0.5

@dataclass
class Calibration:
    mm_per_pixel: Optional[float] = None
    def px_to_mm(self, px: float): return None if self.mm_per_pixel is None else px*self.mm_per_pixel
    def mm_to_px(self, mm: float): return None if self.mm_per_pixel is None else mm/self.mm_per_pixel

@dataclass
class InspectionResult:
    contour: np.ndarray
    outer_center: Tuple[float,float]
    outer_radius_px: float
    inner_center: Tuple[float,float]
    inner_radius_px: float
    circularity: float
    concentricity_ratio: float
    dark_defect_area_mm2: float
    pass_geom: bool
    pass_defect: bool

def normalize_illumination(gray):
    bg=cv2.GaussianBlur(gray,(0,0),21)
    norm=cv2.addWeighted(gray,1.5,bg,-0.5,0)
    return cv2.normalize(norm,None,0,255,cv2.NORM_MINMAX)

def threshold_washers(gray):
    norm=normalize_illumination(gray)
    _,th=cv2.threshold(255-norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),1)
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),2)
    return th

def compute_circularity(c):
    area=cv2.contourArea(c); peri=cv2.arcLength(c,True)
    return 0.0 if peri==0 else float(4.0*np.pi*area/(peri*peri))

def find_washers(mask):
    cs,h=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if h is None: return []
    h=h[0]; outs=[]
    for i,hi in enumerate(h):
        child=hi[2]
        if child!=-1:
            outer=cs[i]; inner=cs[child]
            m = cv2.moments(inner)
            if m["m00"] != 0:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                if cv2.pointPolygonTest(outer, (cx, cy), False) >= 0:
                    outs.append((outer, inner))
    return outs

def analyze_washer(outer,inner,calib,spec,gray):
    (ox,oy),orad=cv2.minEnclosingCircle(outer)
    (ix,iy),irad=cv2.minEnclosingCircle(inner)
    circ=compute_circularity(outer)
    conc=float(np.hypot(ox-ix,oy-iy))/max(orad,1.0)
    ring=np.zeros(gray.shape,np.uint8)
    cv2.drawContours(ring,[outer],-1,255,-1)
    cv2.drawContours(ring,[inner],-1,0,-1)
    blur=cv2.GaussianBlur(gray,(0,0),3)
    diff=cv2.subtract(blur,cv2.GaussianBlur(blur,(0,0),9))
    dark=cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    dark&=ring
    dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),1)
    dark_px=int(np.count_nonzero(dark))
    dark_mm2=float(dark_px*calib.mm_per_pixel**2) if calib.mm_per_pixel else float(dark_px)
    ok_geom=True
    if calib.mm_per_pixel and (spec.expected_outer_mm or spec.expected_inner_mm):
        if spec.expected_outer_mm:
            od=calib.px_to_mm(orad*2.0) or 0.0
            if abs(od-spec.expected_outer_mm)>spec.diameter_tolerance_mm: ok_geom=False
        if spec.expected_inner_mm:
            idm=calib.px_to_mm(irad*2.0) or 0.0
            if abs(idm-spec.expected_inner_mm)>spec.diameter_tolerance_mm: ok_geom=False
    if circ<spec.circularity_min: ok_geom=False
    if conc>spec.concentricity_max_ratio: ok_geom=False
    ok_def=True
    if calib.mm_per_pixel:
        if dark_mm2>spec.dark_defect_min_area_mm2: ok_def=False
    else:
        ring_area=int(cv2.contourArea(outer)-cv2.contourArea(inner))
        if ring_area>0 and dark_px/ring_area>0.002: ok_def=False
    return (outer,(ox,oy),float(orad),(ix,iy),float(irad),float(circ),float(conc),float(dark_mm2),ok_geom,ok_def)

def draw_result(frame,res,calib):
    outer_center=res[1]; outer_r=res[2]; inner_center=res[3]; inner_r=res[4]
    circ=res[5]; conc=res[6]; dark=res[7]; ok_geom=res[8]; ok_def=res[9]
    color=(0,200,0) if (ok_geom and ok_def) else (0,0,255)
    cv2.circle(frame,(int(outer_center[0]),int(outer_center[1])),int(outer_r),color,2)
    cv2.circle(frame,(int(inner_center[0]),int(inner_center[1])),int(inner_r),(255,180,0),2)
    x,y=int(outer_center[0]),int(outer_center[1])
    lines=[f"CIRC {circ:.2f}",f"CONC {conc*100:.1f}%"]
    if calib.mm_per_pixel:
        od=calib.px_to_mm(outer_r*2.0) or 0.0
        idm=calib.px_to_mm(inner_r*2.0) or 0.0
        lines+= [f"OD {od:.2f}mm",f"ID {idm:.2f}mm",f"Dark {dark:.2f}mm^2"]
    else:
        lines.append("UNCALIBRATED")
    for i,t in enumerate(lines):
        cv2.putText(frame,t,(x+10,y+20+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

def run(args):
    spec=WasherSpec(args.expected_outer,args.expected_inner,args.tol,args.circ_min,args.conc_max,args.dark_min_area)
    calib=Calibration(args.mm_per_px)
    cap=cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if args.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    if args.fps: cap.set(cv2.CAP_PROP_FPS,args.fps)
    if not cap.isOpened(): raise RuntimeError("Failed to open camera.")
    last=time.time(); fps=0.0
    while True:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask=threshold_washers(gray)
        results=[]
        for outer,inner in find_washers(mask):
            try:
                r=analyze_washer(outer,inner,calib,spec,gray)
                results.append(r); draw_result(frame,r,calib)
            except Exception: pass
        now=time.time(); dt=now-last; last=now
        if dt>0: fps=0.9*fps+0.1*(1.0/dt)
        okn=sum(1 for r in results if r[8] and r[9])
        cv2.putText(frame,f"OK {okn}/{len(results)} | FPS {fps:.1f}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(50,220,50),2,cv2.LINE_AA)
        cv2.imshow("Washer Inspection",frame)
        if args.debug: cv2.imshow("Mask",mask)
        k=cv2.waitKey(1)&0xFF
        if k in (27, ord('q')): break
        if k==ord('s'): cv2.imwrite("snapshot.png",frame)
        if k==ord('m') and results and spec.expected_inner_mm:
            r=max(results,key=lambda x:x[4])
            calib.mm_per_pixel=spec.expected_inner_mm/(r[4]*2.0)
            print(f"Calibrated from ID: {calib.mm_per_pixel:.6f} mm/px")
    cap.release(); cv2.destroyAllWindows()

def build_argparser():
    p=argparse.ArgumentParser(description="Real-time washer inspection (no AI)")
    p.add_argument("--camera",type=int,default=0)
    p.add_argument("--width",type=int,default=1920)
    p.add_argument("--height",type=int,default=1080)
    p.add_argument("--fps",type=int,default=30)
    p.add_argument("--mm-per-px",type=float,default=None)
    p.add_argument("--expected-outer",type=float,default=None)
    p.add_argument("--expected-inner",type=float,default=None)
    p.add_argument("--tol",type=float,default=0.6)
    p.add_argument("--circ-min",type=float,default=0.85)
    p.add_argument("--conc-max",type=float,default=0.08)
    p.add_argument("--dark-min-area",type=float,default=0.5)
    p.add_argument("--debug",action="store_true")
    return p

if __name__=="__main__":
    run(build_argparser().parse_args())

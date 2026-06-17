"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ADVANCED DRIVER MONITORING SYSTEM  v4.0                                  ║
║   EAR · MAR · PERCLOS · Head Pose · Multi-Level Alerts                     ║
║   + Live Web Dashboard  + Incident Recording  + Session Report             ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALL:   pip install opencv-python mediapipe numpy pygame scipy

FILES (same folder):
    incident_recorder.py   dashboard_server.py   report_generator.py

CONTROLS:  Q=quit   R=reset   S=screenshot   N=night mode

ALARM LOGIC:
    Eyes closed ≥ 3 s continuously  →  L2 loud alarm
    Eyes closed ≥ 5 s continuously  →  L3 continuous siren
    Yawns ≥ 3 in 60 s               →  L1 gentle beep
    Head down ≥ 3 s                 →  L2 alarm
    PERCLOS > 40 % in 60 s          →  L2 alarm
    Normal blink (< 0.4 s)          →  IGNORED
    No face in frame                →  monitoring paused
"""

import os, sys, csv, time, urllib.request
from collections import deque
from datetime import datetime

# ── portfolio modules (optional — graceful fallback if missing) ───────────────
try:
    from incident_recorder import IncidentRecorder
    from report_generator  import generate_report
    from dashboard_server  import DashboardServer
    _EXT = True
except ImportError as _ie:
    print(f"[WARN] Extra module missing ({_ie}) — running without dashboard/recorder")
    _EXT = False

import cv2
import numpy as np
import pygame
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks          import python as mp_python
from mediapipe.tasks.python   import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "face_landmarker.task")

def _ensure_model():
    if os.path.exists(_MODEL_PATH): return
    print("[INFO] Downloading face_landmarker.task (~5 MB) …")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[INFO] Model saved → {_MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}\nManual URL:\n  {_MODEL_URL}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
class Cfg:
    # Calibration
    CALIB_SECS        = 4
    CALIB_PCT         = 15       # percentile of open-eye EAR → threshold
    EAR_FALLBACK      = 0.21

    # Eye closure
    BLINK_MAX_SEC     = 0.40     # ≤ this → normal blink, never alarm
    DROWSY_EYE_SEC    = 3.0      # → L2
    CRITICAL_EYE_SEC  = 5.0      # → L3

    # Yawn
    MAR_THRESH        = 0.58
    YAWN_MIN_SEC      = 1.0      # sustained open ≥ this → real yawn
    YAWN_WINDOW_SEC   = 60
    YAWN_COUNT_ALARM  = 3

    # Head pose
    HEAD_DOWN_DEG     = 18       # relative pitch below this → head down
    HEAD_DOWN_SEC     = 3.0      # → L2
    HEAD_SIDE_DEG     = 30       # yaw beyond this → distracted

    # PERCLOS
    PERCLOS_WIN_SEC   = 60
    PERCLOS_ALARM     = 0.40     # 40 %

    # Display
    WIN_TITLE         = "Driver Monitoring System v4"
    PANEL_W           = 315

    # Logging
    LOG_DIR           = "dms_logs"
    LOG_FILE          = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Camera
    CAM_W, CAM_H      = 1280, 720
    FPS_SMOOTH        = 30

    # Portfolio
    DASH_PORT         = 8765
    DASH_HZ           = 10       # dashboard pushes per second
    INCIDENT_ON       = True
    REPORT_ON         = True


# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════════════════
class LM:
    L_EYE  = [362, 385, 387, 263, 373, 380]
    R_EYE  = [33,  160, 158, 133, 153, 144]
    MOUTH  = [78,  308, 82,  312, 14,  17 ]
    POSE6  = [1,   152, 33,  263, 61,  291]
    POSE3D = np.array([
        [  0.0,    0.0,    0.0],
        [  0.0, -330.0,  -65.0],
        [-225.0,  170.0, -135.0],
        [ 225.0,  170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [ 150.0, -150.0, -125.0],
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def _px(lms, ids, w, h):
    return np.array([(lms[i].x*w, lms[i].y*h) for i in ids], np.float64)

def _ear(pts):
    A=dist.euclidean(pts[1],pts[5]); B=dist.euclidean(pts[2],pts[4])
    C=dist.euclidean(pts[0],pts[3]); return (A+B)/(2.0*C+1e-6)

def _mar(pts):
    A=dist.euclidean(pts[2],pts[5]); B=dist.euclidean(pts[3],pts[4])
    C=dist.euclidean(pts[0],pts[1]); return (A+B)/(2.0*C+1e-6)

def _pose(lms, w, h):
    img = np.array([[lms[i].x*w, lms[i].y*h] for i in LM.POSE6], np.float64)
    f   = float(w)
    cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], np.float64)
    ok, rv, _ = cv2.solvePnP(LM.POSE3D, img, cam, np.zeros((4,1)),
                              flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    a,*_ = cv2.RQDecomp3x3(cv2.Rodrigues(rv)[0])
    return a[0]*360, a[1]*360, a[2]*360


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO
# ══════════════════════════════════════════════════════════════════════════════
def _beep(f1, f2=None, ms=500, vol=0.8, sr=44100):
    n = int(sr*ms/1000); t = np.linspace(0,ms/1000,n,False)
    w = np.sin(2*np.pi*f1*t) if f2 is None else \
        (np.sin(2*np.pi*f1*t)+np.sin(2*np.pi*f2*t))*0.5
    s = (w*vol*32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([s,s]))

class Audio:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self._s = {"L1":_beep(520,ms=380,vol=0.45),
                   "L2":_beep(700,900,600,0.75),
                   "L3":_beep(880,1100,900,1.0)}
        self._last = {}; self._cd = {"L1":3.0,"L2":1.8,"L3":0.9}
        self._ch3  = pygame.mixer.Channel(0)

    def play(self, lv):
        now = time.time()
        if now - self._last.get(lv,0) < self._cd[lv]: return
        if lv=="L3": self._ch3.play(self._s["L3"],loops=-1)
        else:        self._s[lv].play()
        self._last[lv]=now

    def stop_l3(self): self._ch3.stop(); self._last["L3"]=0.0
    def quit(self):    pygame.mixer.stop(); pygame.mixer.quit()


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATOR
# ══════════════════════════════════════════════════════════════════════════════
class Calibrator:
    def __init__(self, cfg):
        self._need    = int(cfg.CALIB_SECS*30)
        self._pct     = cfg.CALIB_PCT
        self._ears    = []; self._pitches = []
        self.done     = False
        self.thr      = cfg.EAR_FALLBACK
        self.neu_pitch= 0.0

    def feed(self, ear_v, pitch=0.0):
        if self.done: return True
        if ear_v > 0.18:
            self._ears.append(ear_v); self._pitches.append(pitch)
        if len(self._ears) >= self._need:
            self.thr       = float(np.clip(np.percentile(self._ears,self._pct),0.14,0.26))
            self.neu_pitch = float(np.median(self._pitches))
            self.done = True
        return self.done

    @property
    def progress(self): return min(len(self._ears)/self._need, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class State:
    def __init__(self, cfg):
        self.cfg = cfg
        # eyes
        self._eye_since  = None; self.eye_sec    = 0.0
        self.blinks      = 0;    self.long_close = 0
        # mouth
        self._mouth_since= None
        self._yawn_times : deque = deque()
        self.tot_yawns   = 0;    self._yawn_alert= False
        # head
        self._head_since = None; self.head_sec   = 0.0
        # perclos
        self._pbuf : deque = deque(); self.perclos = 0.0
        # session
        self.t0          = time.time()
        self.alerts      = {"L1":0,"L2":0,"L3":0}
        self.score       = 0.0
        self.night       = False

    # ── eyes ──────────────────────────────────────────────────────────────────
    def tick_eyes(self, closed, now):
        if closed:
            if self._eye_since is None: self._eye_since = now
            self.eye_sec = now - self._eye_since
        else:
            if self._eye_since is not None:
                dur = now - self._eye_since
                if dur >= self.cfg.BLINK_MAX_SEC: self.long_close += 1
                else:                             self.blinks     += 1
            self._eye_since = None; self.eye_sec = 0.0
        if self.eye_sec >= self.cfg.CRITICAL_EYE_SEC: return "L3"
        if self.eye_sec >= self.cfg.DROWSY_EYE_SEC:   return "L2"
        return ""

    # ── mouth ─────────────────────────────────────────────────────────────────
    def tick_mouth(self, open_, now):
        if open_:
            if self._mouth_since is None: self._mouth_since = now
        else:
            if self._mouth_since is not None:
                if now - self._mouth_since >= self.cfg.YAWN_MIN_SEC:
                    self._yawn_times.append(now); self.tot_yawns += 1
                    self._yawn_alert = False
            self._mouth_since = None
        # prune window
        cut = now - self.cfg.YAWN_WINDOW_SEC
        while self._yawn_times and self._yawn_times[0] < cut:
            self._yawn_times.popleft(); self._yawn_alert = False
        if len(self._yawn_times) >= self.cfg.YAWN_COUNT_ALARM and not self._yawn_alert:
            self._yawn_alert = True; return "L1"
        return ""

    # ── head ──────────────────────────────────────────────────────────────────
    def tick_head(self, pitch, neu, now):
        if pitch - neu < -self.cfg.HEAD_DOWN_DEG:
            if self._head_since is None: self._head_since = now
            self.head_sec = now - self._head_since
        else:
            self._head_since = None; self.head_sec = 0.0
        return "L2" if self.head_sec >= self.cfg.HEAD_DOWN_SEC else ""

    # ── perclos ───────────────────────────────────────────────────────────────
    def tick_perclos(self, closed, now):
        self._pbuf.append((now, closed))
        cut = now - self.cfg.PERCLOS_WIN_SEC
        while self._pbuf and self._pbuf[0][0] < cut: self._pbuf.popleft()
        if len(self._pbuf) < 30: self.perclos = 0.0; return ""
        self.perclos = sum(c for _,c in self._pbuf)/len(self._pbuf)
        return "L2" if self.perclos >= self.cfg.PERCLOS_ALARM else ""

    # ── drowsiness score ──────────────────────────────────────────────────────
    def update_score(self, yaw):
        s  = min(self.eye_sec  / self.cfg.CRITICAL_EYE_SEC, 1.0) * 35
        s += min(self.perclos  / self.cfg.PERCLOS_ALARM,    1.0) * 25
        s += min(len(self._yawn_times)/self.cfg.YAWN_COUNT_ALARM,1.0)*20
        s += min(self.head_sec / self.cfg.HEAD_DOWN_SEC,    1.0) * 15
        if abs(yaw) > self.cfg.HEAD_SIDE_DEG: s += 5
        self.score = min(s, 100.0); return self.score

    @property
    def recent_yawns(self): return len(self._yawn_times)

    @property
    def dur_str(self):
        s = int(time.time()-self.t0)
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, cfg):
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        self.path = os.path.join(cfg.LOG_DIR, cfg.LOG_FILE)
        self._f   = open(self.path, "w", newline="")
        self._w   = csv.writer(self._f)
        self._w.writerow(["time","ear","mar","pitch","yaw","roll",
                           "perclos","eye_sec","head_sec","yawns","score","alarm"])
        self._tick = 0

    def write(self, fps, *vals):
        self._tick += 1
        if self._tick % max(1, int(fps/2)) == 0:
            self._w.writerow([datetime.now().isoformat(timespec="milliseconds"),
                               *[f"{v:.4f}" if isinstance(v,float) else v for v in vals]])
            self._f.flush()

    def close(self): self._f.close()


# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════
# BGR colours
_C = dict(
    bg=(15,15,20), line=(45,45,55), white=(230,230,230), dim=(110,110,120),
    ok=(60,210,80), caut=(40,190,255), warn=(30,130,255), danger=(30,30,230),
    eye_ok=(60,220,80), eye_cl=(30,30,220), mouth=(40,190,255),
)
def _lc(lv): return {"L3":_C["danger"],"L2":_C["warn"],"L1":_C["caut"]}.get(lv,_C["ok"])

def _bar(f,x,y,bw,bh,frac,col):
    frac=max(0.0,min(frac,1.0))
    cv2.rectangle(f,(x,y),(x+bw,y+bh),(35,35,45),-1)
    if frac>0: cv2.rectangle(f,(x,y),(x+int(bw*frac),y+bh),col,-1)
    cv2.rectangle(f,(x,y),(x+bw,y+bh),_C["line"],1)

def _t(f,txt,x,y,sc=0.44,col=None,th=1):
    cv2.putText(f,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,sc,col or _C["white"],th,cv2.LINE_AA)

def draw_calib(frame, prog):
    h,w=frame.shape[:2]
    x1,x2=int(w*.15),int(w*.85); y1,y2=int(h*.46),int(h*.54)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(30,30,30),-1)
    cv2.rectangle(frame,(x1,y1),(int(x1+prog*(x2-x1)),y2),_C["ok"],-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),_C["line"],2)
    _t(frame,f"Calibrating ... {prog*100:.0f}%  |  Keep eyes OPEN",x1+8,y1-12,0.58,_C["white"],2)

def draw_no_face(frame):
    h,w=frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,h),(0,0,0),5)
    _t(frame,"NO FACE DETECTED",int(w*.26),int(h*.50),1.0,_C["warn"],2)
    _t(frame,"Monitoring paused — move into frame",int(w*.26),int(h*.57),0.52,_C["dim"])

def draw_hud(frame, st: State, calib: Calibrator,
             ear_v,mar_v,pitch,yaw,roll,fps,alarm,night):
    h,w=frame.shape[:2]; pw=Cfg.PANEL_W
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(pw,h),_C["bg"],-1)
    cv2.addWeighted(ov,0.73,frame,0.27,0,frame)
    cv2.line(frame,(pw,0),(pw,h),_C["line"],1)
    ac=_lc(alarm); y=28
    rel=pitch-calib.neu_pitch

    # title
    _t(frame,"DRIVER MONITOR",10,y,0.60,ac,2); y+=25
    _t(frame,f"Session {st.dur_str}   FPS {fps:.0f}",10,y,0.39,_C["dim"]); y+=17
    cv2.line(frame,(8,y),(pw-8,y),_C["line"],1); y+=10

    # score
    sc_col=(_C["ok"] if st.score<30 else _C["caut"] if st.score<55
            else _C["warn"] if st.score<75 else _C["danger"])
    _t(frame,f"Drowsiness Score: {st.score:.0f}/100",10,y,0.43,_C["white"]); y+=14
    _bar(frame,10,y,pw-20,10,st.score/100,sc_col); y+=22
    cv2.line(frame,(8,y),(pw-8,y),_C["line"],1); y+=10

    # EAR
    ec=_C["eye_ok"] if ear_v>=calib.thr else _C["eye_cl"]
    _t(frame,f"EAR {ear_v:.3f}  thr {calib.thr:.3f}",10,y,0.42,ec); y+=13
    _bar(frame,10,y,pw-20,8,ear_v/0.45,ec); y+=14
    if st.eye_sec>0:
        ef=min(st.eye_sec/Cfg.CRITICAL_EYE_SEC,1.0)
        ec2=(_C["danger"] if st.eye_sec>=Cfg.CRITICAL_EYE_SEC
             else _C["warn"] if st.eye_sec>=Cfg.DROWSY_EYE_SEC else _C["caut"])
        _t(frame,f"  Eyes closed: {st.eye_sec:.1f}s",10,y,0.42,ec2); y+=13
        _bar(frame,10,y,pw-20,6,ef,ec2); y+=14
    else: y+=4
    cv2.line(frame,(8,y),(pw-8,y),_C["line"],1); y+=10

    # MAR / yawn
    mc=_C["caut"] if mar_v>Cfg.MAR_THRESH else _C["white"]
    _t(frame,f"MAR {mar_v:.3f}  thr {Cfg.MAR_THRESH:.2f}",10,y,0.42,mc); y+=13
    _bar(frame,10,y,pw-20,8,mar_v/0.9,mc); y+=14
    yc=(_C["caut"] if st.recent_yawns>=Cfg.YAWN_COUNT_ALARM else _C["white"])
    _t(frame,f"  Yawns (60s): {st.recent_yawns}/{Cfg.YAWN_COUNT_ALARM}  Total: {st.tot_yawns}",
       10,y,0.40,yc); y+=15
    cv2.line(frame,(8,y),(pw-8,y),_C["line"],1); y+=10

    # PERCLOS
    pc=_C["warn"] if st.perclos>=Cfg.PERCLOS_ALARM else _C["white"]
    _t(frame,f"PERCLOS {st.perclos*100:.1f}%  alarm≥{Cfg.PERCLOS_ALARM*100:.0f}%",10,y,0.42,pc); y+=13
    _bar(frame,10,y,pw-20,8,st.perclos/Cfg.PERCLOS_ALARM,pc); y+=18
    cv2.line(frame,(8,y),(pw-8,y),_C["line"],1); y+=10

    # head pose
    pp=_C["danger"] if rel<-Cfg.HEAD_DOWN_DEG else _C["ok"]
    yp=_C["warn"]   if abs(yaw)>Cfg.HEAD_SIDE_DEG else _C["ok"]
    _t(frame,f"Pitch {pitch:+.1f}  rel {rel:+.1f}",10,y,0.42,pp); y+=15
    _t(frame,f"Yaw   {yaw:+.1f}",10,y,0.42,yp); y+=15
    _t(frame,f"Roll  {roll:+.1f}",10,y,0.42,_C["white"]); y+=14
    if st.head_sec>0:
        _t(frame,f"  Head down: {st.head_sec:.1f}s",10,y,0.42,_C["warn"]); y+=13
        _bar(frame,10,y,pw-20,6,st.head_sec/Cfg.HEAD_DOWN_SEC,_C["warn"]); y+=13
    cv2.line(frame,(8,y),(pw-8,y),_C["line"],1); y+=10

    # alerts
    _t(frame,f"L1:{st.alerts['L1']}  L2:{st.alerts['L2']}  L3:{st.alerts['L3']}",
       10,y,0.42,_C["dim"]); y+=15
    _t(frame,f"Blinks:{st.blinks}  Long:{st.long_close}",10,y,0.40,_C["dim"]); y+=15
    nm=_C["caut"] if night else _C["dim"]
    _t(frame,f"Night mode: {'ON [N]' if night else 'OFF [N]'}",10,y,0.38,nm)

    # bottom banner
    if alarm:
        msgs={"L1":"YAWN x3 — Please take a break",
              "L2":"WARNING — Drowsiness detected!",
              "L3":"CRITICAL — PULL OVER NOW!"}
        by=h-54
        cv2.rectangle(frame,(0,by),(w,h),ac,-1)
        _t(frame,msgs.get(alarm,""),pw+12,h-18,0.78,_C["white"],2)

def draw_lm(frame,el,er,mo,ear_v,thr):
    col=_C["eye_ok"] if ear_v>=thr else _C["eye_cl"]
    for pts in (el,er):
        cv2.drawContours(frame,[cv2.convexHull(pts.astype(np.int32))],-1,col,1)
    cv2.drawContours(frame,[cv2.convexHull(mo.astype(np.int32))],-1,_C["mouth"],1)


# ══════════════════════════════════════════════════════════════════════════════
# NIGHT-MODE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(frame, night):
    if not night: return frame
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b  = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l),a,b]), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DMS APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class DMS:
    def __init__(self):
        _ensure_model()
        self.cfg    = Cfg()
        self.calib  = Calibrator(self.cfg)
        self.state  = State(self.cfg)
        self.audio  = Audio()
        self.logger = Logger(self.cfg)

        self._fps_buf  = deque(maxlen=Cfg.FPS_SMOOTH)
        self._prev_t   = time.time()
        self._fms      = 0          # frame timestamp ms for MediaPipe VIDEO mode
        self._alarm    = ""
        self._prev_alm = ""
        self._dtick    = 0

        # portfolio modules
        self.recorder  = (IncidentRecorder() if _EXT and self.cfg.INCIDENT_ON else None)
        self.dashboard = (DashboardServer(self.cfg.DASH_PORT) if _EXT else None)
        if self.dashboard: self.dashboard.start()

        # MediaPipe face landmarker
        opts = mp_vision.FaceLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode = mp_vision.RunningMode.VIDEO,
            num_faces    = 1,
            min_face_detection_confidence=0.65,
            min_face_presence_confidence =0.65,
            min_tracking_confidence      =0.65,
        )
        self._lmk = mp_vision.FaceLandmarker.create_from_options(opts)

    # ── single frame ──────────────────────────────────────────────────────────
    def _process(self, frame, fps):
        h,w  = frame.shape[:2]
        now  = time.time()
        enh  = preprocess(frame, self.state.night)
        rgb  = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
        self._fms += max(1, int(1000/max(fps,1)))
        res  = self._lmk.detect_for_video(
                   mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                   self._fms)

        if not res.face_landmarks:
            draw_no_face(enh); return enh

        lms  = res.face_landmarks[0]
        el   = _px(lms, LM.L_EYE, w, h)
        er   = _px(lms, LM.R_EYE, w, h)
        mo   = _px(lms, LM.MOUTH, w, h)
        ear_v= (_ear(el)+_ear(er))/2.0
        mar_v= _mar(mo)
        pitch,yaw,roll = _pose(lms,w,h)

        # calibration phase
        if not self.calib.done:
            self.calib.feed(ear_v, pitch)
            draw_calib(enh, self.calib.progress)
            return enh

        thr = self.calib.thr
        closed = ear_v < thr

        # state ticks
        e_alm = self.state.tick_eyes(closed, now)
        m_alm = self.state.tick_mouth(mar_v > self.cfg.MAR_THRESH, now)
        h_alm = self.state.tick_head(pitch, self.calib.neu_pitch, now)
        p_alm = self.state.tick_perclos(closed, now)
        score = self.state.update_score(yaw)

        # alarm priority
        prev  = self._alarm
        if e_alm=="L3":
            self._alarm="L3"
        elif e_alm=="L2" or h_alm=="L2" or p_alm=="L2":
            if prev=="L3": self.audio.stop_l3()
            self._alarm="L2"
        elif m_alm=="L1":
            self._alarm="L1"
        else:
            if prev=="L3": self.audio.stop_l3()
            self._alarm=""

        # audio
        if   self._alarm=="L3": self.audio.play("L3")
        elif self._alarm=="L2": self.audio.play("L2")
        elif self._alarm=="L1": self.audio.play("L1")

        # alert counts (edge)
        if self._alarm and self._alarm!=prev:
            self.state.alerts[self._alarm] += 1

        # draw
        draw_lm(enh, el, er, mo, ear_v, thr)
        draw_hud(enh, self.state, self.calib,
                 ear_v, mar_v, pitch, yaw, roll,
                 fps, self._alarm, self.state.night)

        # log
        self.logger.write(fps, ear_v,mar_v,pitch,yaw,roll,
                          self.state.perclos, self.state.eye_sec,
                          self.state.head_sec, self.state.recent_yawns,
                          score, self._alarm or "OK")

        # incident recorder
        if self.recorder:
            self.recorder.feed(enh, fps)
            if self._alarm in ("L2","L3") and self._prev_alm not in ("L2","L3"):
                self.recorder.trigger(self._alarm,
                    {"L2":"Eye/Head/PERCLOS warning","L3":"Eyes closed 5s"}.get(self._alarm,""),
                    fps)
        self._prev_alm = self._alarm

        # dashboard push (throttled to DASH_HZ)
        if self.dashboard:
            self._dtick += 1
            if self._dtick % max(1, int(fps/self.cfg.DASH_HZ)) == 0:
                self.dashboard.push({
                    "score":round(score,1), "ear":round(ear_v,4),
                    "mar":round(mar_v,4),   "perclos":round(self.state.perclos,4),
                    "eye_sec":round(self.state.eye_sec,2),
                    "head_sec":round(self.state.head_sec,2),
                    "rel_pitch":round(pitch-self.calib.neu_pitch,1),
                    "yaw":round(yaw,1), "blinks":self.state.blinks,
                    "yawns":self.state.recent_yawns,
                    "tot_yawns":self.state.tot_yawns,
                    "long_close":self.state.long_close,
                    "l1":self.state.alerts["L1"], "l2":self.state.alerts["L2"],
                    "l3":self.state.alerts["L3"], "alarm":self._alarm,
                    "fps":round(fps,1), "session":self.state.dur_str,
                    "night":self.state.night,
                })
        return enh

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAM_H)

        print("="*64)
        print("  DMS v4  |  Q=quit  R=reset  S=screenshot  N=night mode")
        print(f"  Log   → {self.logger.path}")
        if self.dashboard:
            print(f"  Dash  → http://localhost:{self.cfg.DASH_PORT}")
        if self.recorder:
            print(f"  Clips → incidents/")
        print("="*64)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam read failed."); break

            now = time.time()
            self._fps_buf.append(1.0/max(now-self._prev_t,1e-9))
            self._prev_t = now
            fps = float(np.mean(self._fps_buf))

            out = self._process(cv2.flip(frame,1), fps)
            cv2.imshow(self.cfg.WIN_TITLE, out)

            k = cv2.waitKey(1) & 0xFF
            if   k==ord('q'): break
            elif k==ord('r'):
                self.state=State(self.cfg); self.calib=Calibrator(self.cfg)
                self._alarm=self._prev_alm=""; print("[INFO] Session reset.")
            elif k==ord('s'):
                fn=f"snap_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(fn,out); print(f"[INFO] Screenshot → {fn}")
            elif k==ord('n'):
                self.state.night=not self.state.night
                print(f"[INFO] Night mode {'ON' if self.state.night else 'OFF'}")

        # ── cleanup ───────────────────────────────────────────────────────────
        cap.release(); cv2.destroyAllWindows()
        self.audio.stop_l3(); self.audio.quit()
        self.logger.close(); self._lmk.close()
        if self.recorder:  self.recorder.close()
        if self.dashboard: self.dashboard.stop()

        s = self.state
        print(f"\n[DONE] Log → {self.logger.path}")
        print("\n─── Session Summary ──────────────────────────────────────────")
        print(f"  Duration          : {s.dur_str}")
        print(f"  Blinks            : {s.blinks}")
        print(f"  Total yawns       : {s.tot_yawns}")
        print(f"  Long eye closures : {s.long_close}")
        print(f"  L1 yawn alerts    : {s.alerts['L1']}")
        print(f"  L2 warnings       : {s.alerts['L2']}")
        print(f"  L3 critical       : {s.alerts['L3']}")
        print("─────────────────────────────────────────────────────────────\n")

        # generate HTML report
        if _EXT and self.cfg.REPORT_ON:
            try:
                generate_report(self.logger.path, {
                    "duration":s.dur_str, "blinks":s.blinks,
                    "yawns":s.tot_yawns, "long_closures":s.long_close,
                    "l1":s.alerts["L1"], "l2":s.alerts["L2"],
                    "l3":s.alerts["L3"],
                    "ear_thr":f"{self.calib.thr:.3f}",
                    "neutral_pitch":f"{self.calib.neu_pitch:.1f}",
                }, self.recorder.get_log() if self.recorder else [])
            except Exception as e:
                print(f"[WARN] Report failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DMS().run()
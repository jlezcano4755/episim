import threading, time, csv, io, os, sqlite3
from datetime import datetime, timedelta
from typing import Dict, Callable, List, Set
from flask import Flask, request, abort, jsonify, Response

# --------------------- Parámetros epidemiológicos ---------------------------
POPULATION      = 4_500_000
R0              = 5.0
INC_DAYS        = 4
INF_DAYS        = 7
FATALITY        = 0.70

# --------------------- Parámetros de simulación -----------------------------
SIM_DAYS_PER_MIN   = 9
SIM_HOURS_PER_STEP = 1
SECS_PER_STEP      = 60 / (SIM_DAYS_PER_MIN * 24)

# Parámetros derivados
BETA0  = R0 / INF_DAYS
SIGMA  = 1 / INC_DAYS
GAMMA0 = (1 - FATALITY) / INF_DAYS
MU0    = FATALITY / INF_DAYS

# Economía
PER_CAPITA_DAILY = 10_000_000 / POPULATION    # ≈ 2.22 USD
INITIAL_BUDGET   = 2_000_000_000

# Sistema sanitario
H_CAP            = 10_000                     # camas UCI iniciales

# --------------------- Estado global --------------------------------------
class State:
    __slots__ = ("S","E","I","R","D","t_hours",
                 "beta","gamma","mu",
                 "budget","income_mult","upkeep_daily",
                 "done_actions","lock",
                 "mutation_counter","mutated",
                 "game_over","game_over_reason")
    def __init__(self):
        self.S = POPULATION - 10_000
        self.E = 10_000
        self.I = 0
        self.R = 0
        self.D = 0
        self.t_hours = 0

        self.beta = BETA0
        self.gamma = GAMMA0
        self.mu = MU0

        self.budget = INITIAL_BUDGET
        self.income_mult = 1.0
        self.upkeep_daily = 0

        self.done_actions: Set[str] = set()
        self.lock = threading.Lock()

        self.mutation_counter = 0
        self.mutated = False

        self.game_over = False
        self.game_over_reason = ""

STATE = State()

# --------------------- SQLite trace ---------------------------------------
DB_PATH = f"round_{int(time.time())}.db"
_db_lock = threading.Lock()

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""CREATE TABLE IF NOT EXISTS state (
            t_hours INTEGER PRIMARY KEY,
            S REAL,E REAL,I REAL,R REAL,D REAL,
            budget REAL
        )""")
        conn.commit()
_init_db()

def _save_state():
    with _db_lock, sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR REPLACE INTO state VALUES (?,?,?,?,?,?,?)",
                     (STATE.t_hours, STATE.S, STATE.E, STATE.I,
                      STATE.R, STATE.D, STATE.budget))
        conn.commit()

# --------------------- Tech‑tree & acciones --------------------------------
class Action:
    def __init__(self,name,tech,cost,upkeep,beta_mult,gamma_mult,mu_mult,income_mult,vaccinate,prereq,effect):
        self.name=name;self.tech=tech;self.cost=cost
        self.upkeep=upkeep
        self.beta_mult=beta_mult
        self.gamma_mult=gamma_mult
        self.mu_mult=mu_mult
        self.income_mult=income_mult
        self.vaccinate=vaccinate
        self.prereq=prereq
        self.effect=effect

ACTIONS: Dict[str,Action] = {}
TECH_TREES = {
    "Salud Pública":[],
    "Capacidad Médica":[],
    "Farmacéutica":[],
    "Datos & Vigilancia":[]
}

def register_action(name,*,tech,cost,prereq=None,
                    upkeep=0,beta_mult=1.0,gamma_mult=1.0,mu_mult=1.0,
                    income_mult=1.0,vaccinate=0):
    prereq = prereq or []
    def deco(fn:Callable[[],None]):
        ACTIONS[name]=Action(name,tech,cost,upkeep,beta_mult,gamma_mult,mu_mult,
                             income_mult,vaccinate,prereq,fn)
        TECH_TREES[tech].append(name)
        return fn
    return deco

# -------- Definición de las 12 acciones (con upkeep & economía) ------------
@register_action("uso_mascarillas", tech="Salud Pública",
                 cost=50_000_000, upkeep=1_000_000, beta_mult=0.85, income_mult=0.97)
def _mask(): pass

@register_action("cierres_regionales", tech="Salud Pública",
                 cost=200_000_000, upkeep=0, beta_mult=0.60, income_mult=0.80,
                 prereq=["uso_mascarillas"])
def _regional(): pass

@register_action("cierre_nacional", tech="Salud Pública",
                 cost=500_000_000, upkeep=0, beta_mult=0.40, income_mult=0.60,
                 prereq=["cierres_regionales"])
def _national(): pass

@register_action("hospitales_campo", tech="Capacidad Médica",
                 cost=150_000_000, upkeep=1_500_000, mu_mult=0.9)
def _field(): pass

@register_action("expansion_ucis", tech="Capacidad Médica",
                 cost=300_000_000, upkeep=2_000_000, mu_mult=0.8,
                 prereq=["hospitales_campo"])
def _icus(): pass

@register_action("ventiladores", tech="Capacidad Médica",
                 cost=250_000_000, upkeep=2_000_000, mu_mult=0.7,
                 prereq=["expansion_ucis"])
def _vents(): pass

@register_action("investigar_antivirales", tech="Farmacéutica",
                 cost=400_000_000, upkeep=4_000_000, gamma_mult=1.2, mu_mult=0.9)
def _antiviral(): pass

@register_action("investigar_vacuna", tech="Farmacéutica",
                 cost=600_000_000, upkeep=6_000_000,
                 prereq=["investigar_antivirales"])
def _vaxR(): pass

@register_action("vacunacion_masiva", tech="Farmacéutica",
                 cost=800_000_000, upkeep=0, vaccinate=2_000_000,
                 prereq=["investigar_vacuna"])
def _massvax(): pass

@register_action("pruebas_infeccion", tech="Datos & Vigilancia",
                 cost=120_000_000, upkeep=2_000_000, beta_mult=0.90)
def _testing(): pass

@register_action("trazabilidad_contactos", tech="Datos & Vigilancia",
                 cost=180_000_000, upkeep=3_000_000, beta_mult=0.85,
                 prereq=["pruebas_infeccion"])
def _tracing(): pass

@register_action("comunicacion_riesgos", tech="Datos & Vigilancia",
                 cost=70_000_000, upkeep=500_000, beta_mult=0.90,
                 prereq=["pruebas_infeccion"])
def _risk(): pass

# --------------------- Scheduling -----------------------------------------
_sched: List[tuple] = []  # (execute_at_hour, action_name)
_sched_lock = threading.Lock()

def _apply_action(act:Action):
    with STATE.lock:
        # economics
        STATE.budget -= act.cost
        STATE.upkeep_daily += act.upkeep
        STATE.income_mult  *= act.income_mult
        STATE.beta   *= act.beta_mult
        STATE.gamma  *= act.gamma_mult
        STATE.mu     *= act.mu_mult
        if act.vaccinate:
            moved=min(STATE.S, act.vaccinate)
            STATE.S -= moved
            STATE.R += moved
        STATE.done_actions.add(act.name)

def schedule_action(name:str, delay:int=0):
    if name not in ACTIONS:
        raise ValueError("Acción desconocida")
    act=ACTIONS[name]
    missing=[p for p in act.prereq if p not in STATE.done_actions]
    if missing:
        raise ValueError(f"Prerequisitos no cumplidos: {missing}")
    with STATE.lock:
        if STATE.budget < act.cost:
            raise ValueError("Presupuesto insuficiente")
    with _sched_lock:
        _sched.append((STATE.t_hours+delay, name))
        _sched.sort(key=lambda x:x[0])

def _run_sched():
    with _sched_lock:
        while _sched and _sched[0][0] <= STATE.t_hours:
            _, n = _sched.pop(0)
            _apply_action(ACTIONS[n])

# --------------------- Motor de simulación ---------------------------------
def _income_daily():
    return (STATE.S + STATE.R) * PER_CAPITA_DAILY * STATE.income_mult

def _check_mutation():
    if STATE.I > 1_000_000:
        STATE.mutation_counter += 1
    else:
        STATE.mutation_counter = 0
    if STATE.mutation_counter >= 10 and not STATE.mutated:
        STATE.beta  *= 1.20
        STATE.mu    *= 1.20
        STATE.mutated = True

def _step():
    # evitar integrar si terminó
    if STATE.game_over:
        return
    with STATE.lock:
        # variables locales
        S,E,I,R,D = STATE.S, STATE.E, STATE.I, STATE.R, STATE.D
        N = S+E+I+R
        # hospital overload
        mu_effective = STATE.mu
        if I > H_CAP:
            mu_effective = STATE.mu * (I / H_CAP)
        # rates per hour
        beta_h  = STATE.beta  / 24
        sigma_h = SIGMA       / 24
        gamma_h = STATE.gamma / 24
        mu_h    = mu_effective / 24
        # Euler
        dS = -beta_h*S*I/N
        dE =  beta_h*S*I/N - sigma_h*E
        dI =  sigma_h*E - (gamma_h+mu_h)*I
        dR =  gamma_h*I
        dD =  mu_h*I
        STATE.S += dS; STATE.E += dE; STATE.I += dI; STATE.R += dR; STATE.D += dD
        STATE.t_hours += SIM_HOURS_PER_STEP
        # economía
        if STATE.t_hours % 24 == 0:
            STATE.budget += _income_daily() - STATE.upkeep_daily
        # mutación
        if STATE.t_hours % 24 == 0:
            _check_mutation()
        # condiciones de game over
        survivors = STATE.S + STATE.R
        if STATE.budget < 0:
            STATE.game_over = True
            STATE.game_over_reason = "Bancarrota"
        if survivors < POPULATION*0.01:
            STATE.game_over = True
            STATE.game_over_reason = "Colapso demográfico"
    _save_state()

def _loop():
    while True:
        t0 = time.time()
        _run_sched()
        _step()
        time.sleep(max(0.0, SECS_PER_STEP - (time.time()-t0)))

threading.Thread(target=_loop, daemon=True).start()

# --------------------- Flask API ------------------------------------------
app = Flask(__name__)

@app.route("/state")
def state_csv():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM state ORDER BY t_hours").fetchall()
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["t_hours","S","E","I","R","D","budget"])
        w.writerows(rows)
    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":"attachment; filename=state.csv"}
    )

@app.route("/tech")
def tech():
    view={}
    for tech, acts in TECH_TREES.items():
        items=[]
        for n in acts:
            a=ACTIONS[n]
            status=("done" if n in STATE.done_actions else
                    "queued" if any(x[1]==n for x in _sched) else
                    "available" if all(p in STATE.done_actions for p in a.prereq) and a.cost<=STATE.budget
                    else "locked")
            items.append(dict(
                name=n,
                cost=a.cost,
                upkeep=a.upkeep,
                prereq=a.prereq,
                status=status
            ))
        view[tech]=items
    return jsonify(view)

@app.route("/action", methods=["POST"])
def post_action():
    data = request.get_json(force=True, silent=True) or {}
    name  = data.get("name")
    delay = int(data.get("delay_hours",0))
    try:
        schedule_action(name, delay)
    except ValueError as e:
        abort(400, str(e))
    return {"status":"scheduled","action":name,"exec_in_hours":delay}

@app.route("/status")
def status():
    with STATE.lock:
        return jsonify(dict(
            t_hours = STATE.t_hours,
            game_over = STATE.game_over,
            reason = STATE.game_over_reason if STATE.game_over else "",
            budget = STATE.budget,
            survivors = STATE.S + STATE.R,
            infected = STATE.I,
            mutated = STATE.mutated
        ))

@app.route("/")
def root():
    return "Simulación epidémica."

if __name__ == "__main__":
    print(f"Servidor v4 listo en http://0.0.0.0:8000  |  DB: {DB_PATH}")
    app.run(host="0.0.0.0", port=8000, debug=False)

from __future__ import annotations
import csv
import io
import json
import os
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

from flask import Flask, Response, abort, jsonify, request


HOURS_PER_DAY = 24
SURVIVAL_THRESHOLD = 0.5


def load_config(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

class Tech:
    def __init__(self, data: Dict[str, Any]):
        self.name = data['name']
        self.cost = data.get('cost', 0)
        self.upkeep = data.get('upkeep', 0)
        self.beta_mult = data.get('beta_mult', 1.0)
        self.income_mult = data.get('income_mult', 1.0)
        self.mu_mult = data.get('mu_mult', 1.0)
        self.gamma_mult = data.get('gamma_mult', 1.0)
        self.vaccinate_total = data.get('vaccinate_total', 0)
        self.vaccinate_days = data.get('vaccinate_days', 0)
        self.prereq = data.get('prereq', [])
        self.cooldown_days = data.get('cooldown_days', 0)
        self.active = False
        self.cooldown = 0

    def step(self) -> None:
        self.cooldown = max(0, self.cooldown - 1)

    def effect(self) -> Dict[str, float]:
        if not self.active:
            return {}
        return {
            'beta_mult': self.beta_mult,
            'income_mult': self.income_mult,
            'mu_mult': self.mu_mult,
            'gamma_mult': self.gamma_mult,
        }

class Evt(NamedTuple):
    t: int
    k: str
    p: Dict[str, Any]

class Sim:
    def __init__(self, config: Dict[str, Any]):
        pop = config['population']
        self.S = float(pop - config['initial_exposed'] - config['initial_infected'])
        self.E = float(config['initial_exposed'])
        self.I = float(config['initial_infected'])
        self.R = 0.0
        self.D = 0.0

        epi = config['epidemiology']
        self.sg = 1 / epi['incubation_days']
        self.gm0 = 1 / epi['infectious_days']
        self.mu = epi['fatality'] / epi['infectious_days']
        self.b0 = epi['R0'] * (self.gm0 + self.mu)

        eco = config['economy']
        self.inc = eco['per_capita_daily']
        self.budget = float(eco['initial_budget'])

        self.ICU = config.get('healthcare', {}).get('icu_capacity', 0)

        sim_cfg = config.get('simulation', {})
        self.dh = sim_cfg.get('sim_hours_per_step', 1)
        self.time = 0

        stoch = config.get('stochastic', {})
        self.beta_noise_sd = stoch.get('beta_noise_sd', 0.03)
        self.mutation_prob = stoch.get('daily_mutation_prob', 0.02)
        self.mutation_beta_range = stoch.get('mutation_beta_range', [1.05, 1.3])
        self.mutation_mu_range = stoch.get('mutation_mu_range', [0.9, 1.2])
        self.superspreader_prob = stoch.get('superspreader_prob', 0.03)
        self.superspreader_size_range = stoch.get('superspreader_size_range', [100, 3000])

        self.tech = {t['name']: Tech(t) for t in config['actions']}
        self.queue: List[Tuple[int, str, bool]] = []
        self.events: List[Evt] = []

        self.db_file = f"run_{int(time.time())}.db"
        self.lock = threading.Lock()

        self._init_db()
        self._save_state()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_file) as conn:
            conn.execute('PRAGMA journal_mode = WAL;')
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS state (
                    t INTEGER PRIMARY KEY, S REAL, E REAL, I REAL, R REAL, D REAL, b REAL
                )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS events (
                    t INTEGER, k TEXT, p TEXT
                )
                '''
            )

    def _save_state(self) -> None:
        with self.lock, sqlite3.connect(self.db_file) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO state VALUES (?, ?, ?, ?, ?, ?, ?)',
                (self.time, self.S, self.E, self.I, self.R, self.D, self.budget)
            )
            if self.events:
                conn.executemany(
                    'INSERT INTO events VALUES (?, ?, ?)',
                    [(e.t, e.k, json.dumps(e.p)) for e in self.events]
                )
                self.events.clear()
            conn.commit()

    def schedule_action(self, name: str, delay: int = 0, activate: bool = True) -> None:
        if name not in self.tech:
            raise ValueError('Invalid action')
        self.queue.append((self.time + delay, name, activate))
        self.queue.sort()

    def _apply_scheduled_actions(self) -> None:
        while self.queue and self.queue[0][0] <= self.time:
            _, name, activate = self.queue.pop(0)
            tech = self.tech[name]
            # prerequisites
            if activate and not all(self.tech[req].active for req in tech.prereq):
                continue
            # cooldown
            if tech.cooldown:
                continue
            if activate and not tech.active:
                self.budget -= tech.cost
            tech.active = activate
            tech.cooldown = tech.cooldown_days

    def _introduce_random_events(self, dt: float) -> None:
        if random.random() < self.mutation_prob * dt:
            bf = random.uniform(*self.mutation_beta_range)
            mf = random.uniform(*self.mutation_mu_range)
            self.b0 *= bf
            self.mu *= mf
            self.events.append(Evt(self.time, 'mutation', {'bf': bf, 'mf': mf}))

        if random.random() < self.superspreader_prob * dt:
            add_E = random.randint(*self.superspreader_size_range)
            self.E += add_E
            self.events.append(Evt(self.time, 'ss', {'add_E': add_E}))

    def step(self) -> None:
        self._apply_scheduled_actions()

        beta_mult = income_mult = mu_mult = gamma_mult = 1.0
        total_upkeep = total_vacc = 0.0

        for tech in self.tech.values():
            tech.step()
            if tech.active:
                eff = tech.effect()
                beta_mult *= eff['beta_mult']
                income_mult *= eff['income_mult']
                mu_mult *= eff['mu_mult']
                gamma_mult *= eff['gamma_mult']
                total_upkeep += tech.upkeep
                if tech.vaccinate_total and tech.vaccinate_days:
                    total_vacc += tech.vaccinate_total / tech.vaccinate_days

        beta = self.b0 * beta_mult
        gamma = self.gm0 * gamma_mult
        mu = self.mu * mu_mult

        if self.ICU:
            mu *= max(1.0, self.I / self.ICU)

        dt = self.dh / HOURS_PER_DAY
        self._introduce_random_events(dt)
        beta *= max(0.0, 1 + random.gauss(0, self.beta_noise_sd))

        N = self.S + self.E + self.I + self.R
        self.S += (-beta * self.S * self.I / N - total_vacc) * dt
        self.E += (beta * self.S * self.I / N - self.sg * self.E) * dt
        self.I += (self.sg * self.E - (gamma + mu) * self.I) * dt
        self.R += (gamma * self.I + total_vacc) * dt
        self.D += mu * self.I * dt
        self.budget += (self.S + self.R) * self.inc * income_mult * dt - total_upkeep * dt

        self.time += self.dh
        self._save_state()

    def rt(self) -> float:
        """Compute the effective reproduction number."""
        beta_mult = gamma_mult = mu_mult = 1.0
        for tech in self.tech.values():
            if tech.active:
                eff = tech.effect()
                beta_mult *= eff['beta_mult']
                gamma_mult *= eff['gamma_mult']
                mu_mult *= eff['mu_mult']

        beta = self.b0 * beta_mult
        gamma = self.gm0 * gamma_mult
        mu = self.mu * mu_mult
        N = self.S + self.E + self.I + self.R
        return beta * self.S / N / (gamma + mu)

    def survival_rate(self, initial: float) -> float:
        return (self.S + self.R) / initial

    def export_csv(self) -> str:
        with sqlite3.connect(self.db_file) as conn:
            rows = conn.execute('SELECT * FROM state ORDER BY t').fetchall()
        buf = io.StringIO()
        csv.writer(buf).writerows([['t', 'S', 'E', 'I', 'R', 'D', 'b'], *rows])
        return buf.getvalue()

    def tech_state(self) -> Dict[str, Any]:
        queued_on = {name for _, name, active in self.queue if active}
        state: Dict[str, Any] = {}
        for name, tech in self.tech.items():
            if tech.active:
                status = 'done'
            elif name in queued_on:
                status = 'pending'
            elif all(self.tech[req].active for req in tech.prereq):
                status = 'available'
            else:
                status = 'locked'

            state[name] = {
                'cost': tech.cost,
                'upkeep': tech.upkeep,
                'prereq': tech.prereq,
                'status': status,
                'active': tech.active,
                'cooldown_days': tech.cooldown,
            }
        return state

    def get_events(self, since: int | None = None) -> List[Evt]:
        if since is None:
            return list(self.events)
        with sqlite3.connect(self.db_file) as conn:
            rows = conn.execute('SELECT t, k, p FROM events WHERE t >= ?', (since,)).fetchall()
        return [(t, k, json.loads(p)) for t, k, p in rows]

# ---------------------------------------------------------
# Initialization & Simulation Loop
# ---------------------------------------------------------

cfg_path = os.getenv('EPI_CFG_PATH', 'config.json')
CONFIG = load_config(cfg_path)
sim = Sim(CONFIG)
initial_count = sim.S + sim.E + sim.I + sim.R

game_over = False
game_reason: str | None = None

days_per_real_min = CONFIG.get('simulation', {}).get('days_per_real_min', 0.5)
INTERVAL = 60 / (days_per_real_min * HOURS_PER_DAY / sim.dh)

app = Flask(__name__)
lock = threading.Lock()

def simulation_loop() -> None:
    global game_over, game_reason
    while True:
        start_time = time.time()
        with lock:
            sim.step()
            if not game_over:
                if sim.budget < 0:
                    game_over = True
                    game_reason = 'bankrupt'
                elif sim.survival_rate(initial_count) < SURVIVAL_THRESHOLD:
                    game_over = True
                    game_reason = 'collapse'
        if game_over:
            break
        elapsed = time.time() - start_time
        time.sleep(max(0, INTERVAL - elapsed))

threading.Thread(target=simulation_loop, daemon=True).start()

# ---------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------

@app.route('/')
def root() -> str:
    return 'sim'

@app.route('/ping')
def ping() -> str:
    return 'pong'

@app.route('/state')
def get_state_csv():
    csv_data = sim.export_csv()
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=state.csv'}
    )

@app.route('/tech')
def get_tech():
    return jsonify(sim.tech_state())

@app.route('/events')
def get_events():
    since = request.args.get('since', type=int)
    with lock:
        return jsonify(sim.get_events(since))

@app.route('/action', methods=['POST'])
def post_action():
    if game_over:
        return abort(400, f'Game over: {game_reason}')
    data = request.get_json(force=True, silent=True) or {}
    name = data.get('name')
    delay = int(data.get('delay_hours', 0))
    activate = bool(data.get('active', True))

    if not name:
        return abort(400, 'missing name')

    try:
        with lock:
            sim.schedule_action(name, delay, activate)
    except ValueError as e:
        return abort(400, str(e))

    return {'status': 'scheduled', 'action': name, 'exec_in_hours': delay}

@app.route('/status')
def get_status():
    with lock:
        return jsonify({
            't_hours': sim.time,
            'budget': sim.budget,
            'survivors': sim.S + sim.R,
            'infected': sim.I,
            'dead': sim.D,
            'Rt': sim.rt(),
            'percent_survivors': sim.survival_rate(initial_count),
            'game_over': game_over,
            'reason': game_reason,
        })

if __name__ == '__main__':
    print(f'Server http://0.0.0.0:8000  DB:{sim.db_file}')
    app.run(host='0.0.0.0', port=8000, debug=False)

import argparse
import csv
import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

G = 6.67430e-11   # gravitaciona konstanta
DT = 1e3          # vremenski korak (s)
NUM_STEPS = 500   # broj iteracija

# -----------------------------
# Model
# -----------------------------
@dataclass
class Body:
    mass: float
    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)

    @staticmethod
    def from_vals(mass: float, pos_xy: Tuple[float, float], vel_xy: Tuple[float, float]):
        return Body(
            mass=mass,
            position=np.array(pos_xy, dtype=np.float64),
            velocity=np.array(vel_xy, dtype=np.float64),
        )

def init_solar_three() -> List[Body]:
    # Sunce, Zemlja, Mars (pojednostavljeno)
    return [
        Body.from_vals(1.989e30, (0.0, 0.0), (0.0, 0.0)),            # Sun
        Body.from_vals(5.972e24, (1.5e11, 0.0), (0.0, 29_800.0)),    # Earth
        Body.from_vals(6.39e23,  (2.2e11, 0.0), (0.0, 24_100.0)),    # Mars
    ]

# -----------------------------
# I/O: zapis stanja u CSV
# -----------------------------
def write_header(csv_path: str):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "body_id", "x", "y", "vx", "vy"])

def append_state(csv_path: str, step: int, bodies: List[Body]):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for i, b in enumerate(bodies):
            w.writerow([step, i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]])

# -----------------------------
# Fizicki proracuni
# -----------------------------
def compute_forces_sequential(bodies: List[Body]) -> np.ndarray:
    """Vrati sile (N x 2) za sva tela — sekvencijalno."""
    n = len(bodies)
    forces = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        fi = np.zeros(2, dtype=np.float64)
        mi = bodies[i].mass
        pi = bodies[i].position
        for j in range(n):
            if i == j:
                continue
            pj = bodies[j].position
            mj = bodies[j].mass
            r = pj - pi
            dist = np.linalg.norm(r) + 1e-12  # izbegni deljenje nulom
            fmag = G * mi * mj / (dist * dist)
            fi += fmag * (r / dist)
        forces[i] = fi
    return forces

# ---- Paralelni deo ----
def _forces_chunk(args) -> Tuple[int, int, np.ndarray]:
    """Radnik: racuna sile za tela u intervalu [start, end)."""
    start, end, masses, positions = args
    n = positions.shape[0]
    out = np.zeros((end - start, 2), dtype=np.float64)
    for local_idx, i in enumerate(range(start, end)):
        mi = masses[i]
        pi = positions[i]
        fi = np.zeros(2, dtype=np.float64)
        for j in range(n):
            if i == j:
                continue
            pj = positions[j]
            mj = masses[j]
            r = pj - pi
            dist = np.linalg.norm(r) + 1e-12
            fmag = G * mi * mj / (dist * dist)
            fi += fmag * (r / dist)
        out[local_idx] = fi
    return start, end, out

def compute_forces_parallel_with_pool(bodies, pool, nprocs: int) -> np.ndarray:
    n = len(bodies)
    masses = np.array([b.mass for b in bodies], dtype=np.float64)
    positions = np.stack([b.position for b in bodies]).astype(np.float64)

    # podela indeksa na segmente
    indices = np.linspace(0, n, num=min(nprocs, n) + 1, dtype=int)
    tasks = []
    for k in range(len(indices) - 1):
        start, end = indices[k], indices[k + 1]
        tasks.append((start, end, masses, positions))

    forces = np.zeros((n, 2), dtype=np.float64)
    # koristimo vec postojeci pool
    for start, end, chunk_forces in pool.map(_forces_chunk, tasks):
        forces[start:end, :] = chunk_forces
    return forces


# -----------------------------
# Integracija (Euler)
# -----------------------------
def step_euler(bodies: List[Body], forces: np.ndarray, dt: float):
    for i, b in enumerate(bodies):
        a = forces[i] / b.mass
        b.velocity += a * dt
        b.position += b.velocity * dt

# -----------------------------
# Glavna simulacija
# -----------------------------
def simulate(bodies, csv_path: str, mode: str, nprocs: int):
    write_header(csv_path)
    append_state(csv_path, 0, bodies)

    if mode == "par":
        # Pool samo jednom
        with mp.Pool(processes=min(nprocs, len(bodies))) as pool:
            for t in range(1, NUM_STEPS + 1):
                forces = compute_forces_parallel_with_pool(bodies, pool, nprocs=nprocs)
                step_euler(bodies, forces, DT)
                append_state(csv_path, t, bodies)
                if t % 50 == 0:
                    print(f"[PAR] step {t}/{NUM_STEPS}")
    else:
        # sekvencijalno
        for t in range(1, NUM_STEPS + 1):
            forces = compute_forces_sequential(bodies)
            step_euler(bodies, forces, DT)
            append_state(csv_path, t, bodies)
            if t % 50 == 0:
                print(f"[SEQ] step {t}/{NUM_STEPS}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="N-body (sekvencijalno/paralelno) sa zapisom stanja po iteracijama (CSV).")
    p.add_argument("--mode", choices=["seq", "par"], default="seq",
                   help="Način rada: seq (sekvencijalno) ili par (multiprocessing).")
    p.add_argument("--nprocs", type=int, default=max(1, mp.cpu_count() - 1),
                   help="Broj procesa za parallel (ignorise se u seq).")
    p.add_argument("--steps", type=int, default=NUM_STEPS, help="Broj iteracija.")
    p.add_argument("--dt", type=float, default=DT, help="Vremenski korak.")
    p.add_argument("--out", type=str, default=None, help="Putanja do izlaznog CSV fajla.")
    return p.parse_args()

def main():
    args = parse_args()

    global NUM_STEPS, DT
    NUM_STEPS = int(args.steps)
    DT = float(args.dt)

    bodies = init_solar_three()

    out_csv = args.out or (f"states_{args.mode}.csv")

    print(f"[INFO] Mode={args.mode} steps={NUM_STEPS} dt={DT} out={out_csv}")
    if args.mode == "par":
        print(f"[INFO] nprocs={args.nprocs}")

    simulate(bodies, out_csv, mode=args.mode, nprocs=args.nprocs)
    print(f"[OK] Simulacija završena. Stanja su upisana u: {out_csv}")

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()


# Full Stack Developer — Take-Home Exercise

**Project:** GA-Based PID Controller Tuning Platform  
**Time we expect:** about 4–6 hours  
**Please send it back within:** 48 hours of receiving it  
**Stack:** FastAPI · React + Tailwind CSS · Docker Compose

---

## Why this exercise

We’re building an automatic controller-tuning platform for dynamical systems. The Genetic-Algorithm optimizer and closed-loop simulator already live in `src/`. Control engineers currently use a Streamlit prototype (`app.py`) to try ideas.

We’d like to turn that into a small web app that the rest of the team can use. This exercise is a chance for us to work through that problem together: you get a realistic slice of the work, and we get to see how you approach backend APIs, React UIs, and Docker packaging.

You don’t need any control-theory or GA background. The engine is a black box — just call it the way the Streamlit app does.

---

## What you’ll receive

```
ga-tuner-pid/
|-- src/                          # shared engine (please treat as read-only)
|   |-- ga_optimizer.py
|   |-- simulator.py
|   |-- callbacks.py
|   `-- logger.py
|
|-- case_studies/
|   `-- json/                     # ready-to-use test systems
|       |-- BallBeam.json
|       |-- DCMotor.json
|       `-- InvPendulum.json
|
|-- app.py                        # reference UI (read-only)
`-- README.md
```

**Quick orientation**

1. Run the Streamlit app so you can see the intended flow:
   ```bash
   pip install streamlit numpy plotly pygad
   streamlit run app.py
   ```
2. In your FastAPI code, import the engine the same way:
   ```python
   from src.simulator import SystemSimulator
   from src.ga_optimizer import GAOptimizer
   ```
3. Use the JSON files in `case_studies/json/` for end-to-end testing — each one contains both the dynamics code and all configuration metadata.

---

## What we’d like you to build

A thin productization of the Streamlit prototype:

1. **FastAPI backend** that exposes the GA engine over a small REST API  
2. **React + Tailwind frontend** that covers the main user flow  
3. **Docker Compose** so both services start with a single command

Aim for a clean vertical slice that works end-to-end. Polish is welcome, but a working core path matters more than covering every edge case.

### Backend

Create a `backend/` directory. Please use `src` directly — no need to re-implement the simulator or the genetic algorithm.

Suggested endpoints (feel free to adjust names or group them if something cleaner fits your design):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/systems` | Accept either a JSON case-study body or a `.py` upload + metadata; load dynamics, keep the system in memory, return a `system_id` + summary |
| `GET`  | `/api/v1/systems/{system_id}` | Return system metadata (name, dimensions, limits, suggested targets, etc.) |
| `POST` | `/api/v1/systems/{system_id}/optimize` | Launch a GA run with the supplied ranges, weights, targets and GA hyper-parameters; return best gains, achieved metrics, progress history and stop reason |
| `POST` | `/api/v1/systems/{system_id}/simulate` | (Optional but useful) Run one closed-loop trajectory with fixed gains for plotting |

**Example contracts** (illustrative — refine the shapes as long as the frontend can consume them):

`POST /api/v1/systems` (JSON case-study mode)  
Request body: the full content of e.g. `BallBeam.json`.  
Response:
```json
{
  "system_id": "uuid-string",
  "name": "Ball Beam",
  "num_states": 2,
  "num_inputs": 1
}
```

`POST /api/v1/systems/{system_id}/optimize`  
Request body:
```json
{
  "param_ranges": {
    "Kp": [1.0, 50.0],
    "Ki": [0.01, 5.0],
    "Kd": [0.01, 10.0]
  },
  "weights": {
    "mse": 1.0,
    "settling_time": 0.5,
    "overshoot": 0.5,
    "control_effort": 0.1
  },
  "fixed_targets": {
    "mse": 0.05,
    "settling_time": 1.0,
    "overshoot": 1.0,
    "control_effort": 0.1
  },
  "ga_config": {
    "population_size": 20,
    "generations": 30,
    "seed": 42
  }
}
```

Response:
```json
{
  "success": true,
  "controller_parameters": {"Kp": 12.3, "Ki": 0.4, "Kd": 3.1},
  "achieved_metrics": {
    "mse": 0.03,
    "settling_time": 0.85,
    "overshoot": 0.7,
    "control_effort": 0.08
  },
  "progress": { "... generation-wise history ..." },
  "stop_reason": "score_100",
  "num_evaluations": 412
}
```

**Backend notes**
- Pydantic models for validation are appreciated.
- In-memory storage is fine — no database needed.
- GA runs can take several seconds; a synchronous endpoint is acceptable for this exercise (you may add a simple progress mechanism if you have time).
- Enable CORS for the React dev server (`localhost:5173` or `localhost:3000`).
- Proper status codes (`404` when the system id is unknown, etc.) help a lot.

### Frontend

Create a `frontend/` directory. The core flow we care about is:

1. Load a system (either pick a built-in case study or upload a dynamics definition)  
2. Configure PID search ranges, target metrics & weights, and basic GA settings  
3. Run the optimization and display the results  
4. (Optional) Run a few test simulations with the best gains and plot the trajectories

**Minimum useful UI**
- Case-study selector **or** file/JSON upload; after loading, show system name / state count / target  
- Inputs for Kp/Ki/Kd search ranges  
- Inputs for the four target metrics and their weights  
- GA controls (population size, generations, seed)  
- “Run Optimization” button with loading feedback  
- Results panel: best gains, achieved metrics vs targets, stop reason  
- Simple charts of GA progress (cost / gains vs generation)  
- Optional: mean ± std trajectory plots for a handful of test simulations

Sensible defaults taken from the selected case study are enough. You don’t need to surface every Streamlit knob unless you have time left.

**Frontend notes**
- Tailwind for styling  
- React hooks for state is fine — no need for an external state library  
- Loading and error feedback (disabled button, spinner, short message) make the experience feel finished  
- Put the API base URL in an environment variable (e.g. `VITE_API_URL`)

### Docker

A `docker-compose.yml` at the repo root that brings both services up with:

```bash
docker-compose up --build
```

- Frontend should be able to reach the backend without hard-coded container IPs  
- Backend needs to import `src` (volume mount or `PYTHONPATH`) and have `pygad` + `numpy` installed  

Expected layout after your work:

```
ga-tuner-pid/
|-- src/                  # provided
|-- case_studies/         # provided
|-- app.py                # provided
|-- backend/              # your FastAPI app + Dockerfile
|-- frontend/             # your React app + Dockerfile
|-- docker-compose.yml
`-- README.md             # short notes on how to run it
```

---

## How we’ll try it

We’ll use the files in `case_studies/json/`:

| File               | States | Rough description                  |
|--------------------|--------|------------------------------------|
| `BallBeam.json`    | 2      | Ball on a beam; input = beam angle |
| `DCMotor.json`     | 2      | Motor speed & current; input = voltage |
| `InvPendulum.json` | 2      | Inverted pendulum; input = torque  |

Happy path: load `BallBeam.json` → keep the default ranges/targets → run optimization → see best gains, metrics and a progress plot.

---

## What we’re looking for

| Area        | We care about |
|-------------|---------------|
| Code quality| Clear structure, minimal duplication, especially no re-implementation of `src` |
| API         | Sensible REST shape, validation, understandable errors |
| React       | Component boundaries that make sense, async handling, loading/error states |
| Docker      | Compose file that actually starts both services cleanly |
| README      | A few sentences on how to run it and any assumptions you made |

If something is ambiguous, pick a reasonable approach, note it in the README, and keep going. We’re more interested in judgment and a working result than in perfect adherence to every detail.

---

## Optional extras (only if you have time left)

- Streaming progress updates (SSE / WebSocket) while the GA is running  
- A “reset to case-study defaults” button  
- Basic safety check that uploaded `.py` files don’t contain obvious dangerous imports  
- Unit tests around system load + optimize  

None of these are required for a solid submission.

---

## How to send it back

Push the work to a GitHub repo (public or private with access for us). Please keep the provided `src/`, `case_studies/`, and `app.py` unchanged, and add your `backend/`, `frontend/`, `docker-compose.yml`, and an updated `README.md`.

We’ll review by running:

```bash
docker-compose up --build
```

and walking through the load → optimize path with `BallBeam.json`.

---

Thanks for taking the time. Looking forward to seeing how you’d approach this with us.

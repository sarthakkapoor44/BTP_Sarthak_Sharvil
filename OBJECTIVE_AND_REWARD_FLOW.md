# Objective Value Flow & Reward Prediction Model

## Part 1: Where Objective Comes From

### 1.1 Solver Execution → Objective

```
solve_slot(t, A_prev)
├─ Choose solver (e.g., beam_search)
├─ Execute: result = solver.solve_slot(t, A_prev)
│  └─ Returns SlotSolution with:
│        objective_value = computed cost at slot t
│        placement = new A_t dict
│
└─ This objective_value is the RAW REWARD
```

### 1.2 What Is This Objective?

The objective is the **cost** minimization for slot t:

```python
# From solver implementations
objective = (
    sum(migration costs)              # Cost to move datasets
    + sum(hop violations)              # Penalty for exceeding hop budget
    + sum(uncovered demand penalties)  # Penalty for unmet demand
)

# LOWER is BETTER
# Example values:
#   -34.87 (bad: many violations)
#    0.77  (good: minimal cost)
#   40.17  (excellent: high coverage, low migrations)
```

---

## Part 2: Objective → Reward Transformation Pipeline

### Step 1: Fetch Raw Objective from Solver Result

**File: `adaptive_ensemble_solver.py`, lines 280-294**

```python
def solve_slot(self, t: int, A_prev: Dict[Tuple[int, int], int]) -> SlotSolution:
    slot_start = time.time()

    # ... (solver selection happens here)

    solver = self.candidate_solvers[selected_name]
    one_start = time.time()
    selected = solver.solve_slot(t, copy.deepcopy(A_prev))  # ← EXECUTES SOLVER
    elapsed = time.time() - one_start

    # Extract objective from result
    self._update_after_decision(
        selected_name=selected_name,
        context=context,
        selected_obj=float(selected.objective_value),  # ← RAW OBJECTIVE (e.g., 0.77)
        A_prev=A_prev,
        t=t,
    )
```

**Example:**

```
Solver output: SlotSolution(
    objective_value = 0.77,
    placement = {(dataset_0, server_3): 1, ...},
    status = "Feasible"
)

Raw objective extracted: 0.77
```

### Step 2: Normalize Objective to [0, 1]

**File: `adaptive_ensemble_solver.py`, lines 163-170**

```python
def _normalize_reward(self, raw_reward: float) -> float:
    # (1) Track min and max across all slots and solvers
    self._reward_min = min(self._reward_min, raw_reward)
    self._reward_max = max(self._reward_max, raw_reward)

    # (2) Compute span
    span = self._reward_max - self._reward_min
    if span <= 1e-12:
        return 0.5  # Avoid division by zero

    # (3) Scale to [0, 1]
    val = (raw_reward - self._reward_min) / span
    return float(max(0.0, min(1.0, val)))
```

**Example Flow:**

```
Slot 1:  raw_obj = -34.87 → norm = 0.0 (since it's min observed)
Slot 2:  raw_obj =   0.77 → norm = 0.5 (middle value)
Slot 3:  raw_obj =  40.17 → norm = 1.0 (since it's max observed)

_reward_min = -34.87
_reward_max = 40.17
span = 40.17 - (-34.87) = 75.04

Normalized rewards now all in [0, 1]:
  slot 1: (−34.87 − (−34.87)) / 75.04 = 0.0
  slot 2: (0.77 − (−34.87)) / 75.04 = 0.47
  slot 3: (40.17 − (−34.87)) / 75.04 = 1.0
```

### Step 3: Store Normalized Reward

**File: `adaptive_ensemble_solver.py`, lines 227-231**

```python
def _update_one_arm(self, name: str, x: np.ndarray, raw_reward: float) -> None:
    norm_reward = self._normalize_reward(raw_reward)  # ← NORMALIZED [0,1]

    self.history[name].append(norm_reward)            # ← STORE FOR AVERAGING
    self.pull_counts[name] += 1
```

**Storage:**

```
self.history = {
    'beam_search': [0.0, 0.47, 1.0, 0.85, 0.92, ...],
    'greedy':      [0.3, 0.55, 0.88, 0.75, 0.81, ...],
    'gnn_ppo':     [0.1, 0.42, 0.70, 0.65, 0.72, ...]
}
```

---

## Part 3: Prediction Model for Reward

### The Model: Ridge-Regularized Online Linear Regression

**Mathematical Form:**

```
ŷ_a(x) = θ_a^T · x

where:
  ŷ_a(x) = predicted normalized reward for arm 'a' given context x
  θ_a = learned weight vector for arm 'a'
  x = context feature vector (6D)
```

### 3.1 Model Initialization

**File: `adaptive_ensemble_solver.py`, lines 77-81**

```python
self.context_dim = 6
self.lin_A: Dict[str, np.ndarray] = {}
self.lin_b: Dict[str, np.ndarray] = {}
for name in self.candidate_names:
    self.lin_A[name] = self.lin_ridge * np.eye(self.context_dim, dtype=float)
    #                   ↑ Ridge regularization (lambda=1.0)
    #                   ↑ Initialize as diagonal matrix
    self.lin_b[name] = np.zeros(self.context_dim, dtype=float)
```

**For each arm:**

```
lin_A[beam_search] = [ 1.0   0    0    0    0    0  ]      (6×6 identity × ridge)
                      [  0  1.0   0    0    0    0  ]
                      [  0   0  1.0   0    0    0  ]
                      [  0   0   0  1.0   0    0  ]
                      [  0   0   0   0  1.0   0  ]
                      [  0   0   0   0   0  1.0 ]

lin_b[beam_search] = [ 0, 0, 0, 0, 0, 0 ]              (6×1 zero vector)
```

### 3.2 Making Predictions

**File: `adaptive_ensemble_solver.py`, lines 155-158**

```python
def _predict_context_reward(self, name: str, x: np.ndarray) -> float:
    A_inv = np.linalg.inv(self.lin_A[name])        # Compute (A)^{-1}
    theta = A_inv @ self.lin_b[name]               # Solve: θ = (A)^{-1} · b
    return float(theta @ x)                         # Compute: prediction = θ · x
```

**Step-by-step for slot t=100:**

```python
# Context at slot 100
x = [1.0, 0.65, 0.8, 0.2, 0.15, 0.0686]
#    bias entropy active uncovered migration t_norm

# For beam_search arm:
lin_A_beam = [...updated accumulation matrix...]   # 6×6
lin_b_beam = [...updated target vector...]         # 6×1

A_inv = inv(lin_A_beam)                           # Invert to get (A)^{-1}
theta = A_inv @ lin_b_beam                        # Solve least squares: θ
#     = [0.45, 0.32, -0.18, 0.55, 0.12, -0.08]   # Learned weights

pred = theta @ x
     = 0.45(1.0) + 0.32(0.65) + (-0.18)(0.8) + 0.55(0.2) + 0.12(0.15) + (-0.08)(0.0686)
     = 0.45 + 0.208 - 0.144 + 0.11 + 0.018 - 0.005
     = 0.631                                       # PREDICTED normalized reward
```

**Interpretation:**

```
Predicted reward for beam_search at slot 100: 0.631 [on scale 0-1]
```

### 3.3 Updating the Model (Learning)

**File: `adaptive_ensemble_solver.py`, lines 227-234**

```python
def _update_one_arm(self, name: str, x: np.ndarray, raw_reward: float) -> None:
    norm_reward = self._normalize_reward(raw_reward)

    # Update A matrix: A ← A + x·x^T
    self.lin_A[name] += np.outer(x, x)
    #                    ↑ Outer product: (6×1) ⊗ (1×6) = 6×6 matrix

    # Update b vector: b ← b + reward·x
    self.lin_b[name] += norm_reward * x
    #                    ↑ Scale context by observed reward
```

**Example after one slot:**

```
Before update:
  lin_A = 1.0 * I₆ (identity)
  lin_b = [0, 0, 0, 0, 0, 0]

After observing (x=[1.0, 0.65, 0.8, 0.2, 0.15, 0.0686], reward=0.47):

  x ⊗ x = [1.0²,     1.0×0.65,   1.0×0.8,    ...]     [1.0,   0.65,  0.8,  0.2,  0.15, 0.069]
          [0.65×1.0, 0.65²,      0.65×0.8,   ...] =   [0.65,  0.42,  0.52, 0.13, 0.098, 0.045]
          [...]                                         [...]

  lin_A = [1.0,   0,    0,    0,    0,    0   ] + [1.0,   0.65, 0.8,  0.2,  0.15, 0.069]
          [0,     1.0,  0,    0,    0,    0   ]   [0.65,  0.42, 0.52, 0.13, 0.098, 0.045]
          [...etc...]

  lin_b = [0, 0, 0, 0, 0, 0]^T + 0.47 × [1.0, 0.65, 0.8, 0.2, 0.15, 0.069]^T
        = [0.47, 0.305, 0.376, 0.094, 0.0705, 0.0324]^T
```

---

## Part 4: Decision to Action Flow

### The Full Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ SLOT t: Decision to Execution to Update                          │
└──────────────────────────────────────────────────────────────────┘

1. EXTRACT CONTEXT
   ├─ Active datasets: [dataset_0, dataset_2, dataset_5]
   ├─ Demand entropy from attachment points
   ├─ Uncovered pairs: 45 out of 200
   ├─ Migration pressure: 2 datasets have zero replicas
   └─ x = [1.0, 0.65, 0.8, 0.2, 0.15, 0.0686]

2. PREDICT REWARD FOR EACH ARM
   ├─ beam_search:  pred = θ_beam · x = 0.631
   ├─ greedy:       pred = θ_greedy · x = 0.58
   └─ gnn_ppo:      pred = θ_gnn · x = 0.52

3. SELECT SOLVER (max over predictions)
   └─ chosen = beam_search (highest predicted 0.631)

4. EXECUTE SOLVER
   ├─ Call beam_search.solve_slot(t, A_prev)
   ├─ Returns: SlotSolution(objective_value=0.77)
   └─ Extract: actual_raw_reward = 0.77

5. NORMALIZE OBJECTIVE
   ├─ Normalize: 0.77 → 0.47 (on [0,1] scale)
   └─ norm_reward = 0.47

6. UPDATE MODEL FOR beam_search
   ├─ lin_A += [1.0, 0.65, 0.8, ...]^T ⊗ [1.0, 0.65, 0.8, ...]
   ├─ lin_b += 0.47 × [1.0, 0.65, 0.8, ...]
   └─ Next time we see similar context, θ_beam will predict higher!

7. STORE IN HISTORY
   ├─ history['beam_search'].append(0.47)
   ├─ pull_counts['beam_search'] += 1
   └─ Rolling window: [0.5, 0.72, 0.47]

8. OUTPUT
   └─ SlotSolution with placement A_t, objective 0.77
```

---

## Part 5: Decision Method Integration

### How UCB Uses the Model

**File: `adaptive_ensemble_solver.py`, lines 204-217**

```python
scores: Dict[str, float] = {}
for name in self.candidate_names:
    # STEP 1: Get prediction from context model
    pred = self._predict_context_reward(name, x)  # ← Uses learned θ

    # STEP 2: For UCB family (not Thompson/EXP3)
    if self.bandit_method == "sw_ucb":
        hist = self.history[name][-self.window:]
        n = len(hist)
        mean = float(sum(hist) / max(1, n)) if n > 0 else 0.0
        bonus = self.exploration_c * math.sqrt(math.log(self.total_decisions + 1.0) / max(1.0, n))

        # STEP 3: Combine empirical mean + exploration bonus + context prediction
        scores[name] = mean + bonus + self.context_weight * pred
        #              ↑     ↑      ↑                       ↑
        #              |     |      |                       └─ Predicted reward (0-1)
        #              |     |      └─ Exploration term
        #              |     └─ Recent empirical mean
        #              └─ Solver score for this slot

return max(self.candidate_names, key=lambda nm: scores[nm])
```

**Example Scoring:**

```
Slot 100:
  - beam_search:  mean=0.61, bonus=0.08, pred=0.631
    score = 0.61 + 0.08 + 0.2×0.631 = 0.754

  - greedy:       mean=0.55, bonus=0.09, pred=0.58
    score = 0.55 + 0.09 + 0.2×0.58 = 0.706

  - gnn_ppo:      mean=0.52, bonus=0.10, pred=0.52
    score = 0.52 + 0.10 + 0.2×0.52 = 0.684

DECISION: Pick beam_search (0.754 > 0.706 > 0.684)
```

---

## Part 6: Online Ridge Regression Solving Form

### The Linear Least Squares Problem

At each update, we're solving:

$$\min_{\theta} \|\mathbf{A} \theta - \mathbf{b}\|^2 + \lambda \|\theta\|^2$$

Where:

- $\mathbf{A}$: accumulated context outer products (6×6)
- $\mathbf{b}$: accumulated reward-weighted contexts (6×1)
- $\lambda$: ridge parameter (1.0)

**Solution:**
$$\theta = (\mathbf{A} + \lambda I)^{-1} \mathbf{b}$$

But we initialize:
$$\mathbf{A} = \lambda I$$
$$\mathbf{b} = \mathbf{0}$$

So after first update:
$$\mathbf{A} = \lambda I + \mathbf{x}_1 \mathbf{x}_1^T$$
$$\mathbf{b} = r_1 \mathbf{x}_1$$

**Interpretation:** We're building a linear model: given context $x$, predict reward $y$.

- $\mathbf{A}$ accumulates information about context (Gram matrix)
- $\mathbf{b}$ accumulates alignment between context and reward

---

## Summary Table

| Component             | Where                       | What                               | Example                            |
| --------------------- | --------------------------- | ---------------------------------- | ---------------------------------- |
| **Raw Objective**     | Solver result               | Actual cost (neg = bad)            | 0.77                               |
| **Normalized Reward** | `_normalize_reward()`       | Objective scaled to [0,1]          | 0.47                               |
| **Context**           | `_context_features()`       | 6D state features                  | [1.0, 0.65, 0.8, 0.2, 0.15, 0.069] |
| **Prediction Model**  | `_predict_context_reward()` | Linear regression: $\theta^T x$    | 0.631                              |
| **Learned Weights**   | `lin_A, lin_b`              | Ridge regression parameters        | θ_beam = [0.45, 0.32, -0.18, ...]  |
| **Bandit Score**      | `_choose_solver_no_peek()`  | score = mean + bonus + weight×pred | 0.754                              |
| **Decision**          | argmax(score)               | Choose highest-scoring solver      | beam_search                        |

---

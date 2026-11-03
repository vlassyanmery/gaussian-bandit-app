# Gaussian Thompson Sampling Bandit

Extension of the carousel bandit app from
https://github.com/hovhannisyan91/DS223_Group_Project (`app/front/`).
The original Beta-Bernoulli carousel pages are kept unchanged; a new **Gaussian Bandit** page was added.

**Links**
- GitHub: https://github.com/vlassyanmery/gaussian-bandit-app
- Deployed app: <https://gaussian-bandit-app-kchuqzm63xry7yzprju7ki.streamlit.app>

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run streamlit run main.py
```

Files added to the starter code:
- `gaussian_bandit.py`: Gaussian Thompson sampling algorithm (pure Python/NumPy)
- `pages/Gaussian_Bandit.py`: the new Streamlit page
- `main.py`: one extra line registering the new page
- `pyproject.toml`: `gaussian_bandit` added to `py-modules`

## 1. Original algorithm (Beta-Bernoulli Thompson sampling)

The starter app is a carousel of several alternatives ("bandits"). Each round it shows
`visible_count` out of `total_alternatives`, and the user either clicks one displayed
alternative or clicks nothing.

**Alternative selection** (`sample_carousel` in `bandit_utils.py`). Every alternative has a
Beta(alpha, beta) posterior over its click probability, starting at Beta(1, 1) (uniform prior).
For each alternative the app draws one sample `theta = random.betavariate(alpha, beta)`, sorts
the draws in descending order, and displays the top `visible_count` alternatives (stored in
`carousel["current_display_ids"]`). This is Thompson sampling: uncertain or promising
alternatives often produce large draws, so exploration and exploitation happen automatically.

**Reward recording** (`update_after_feedback`, called from `pages/Interaction.py`). Each
displayed alternative has a "Click" button, and there is a "No click / Next round" button.
For every displayed alternative `impressions += 1`; the clicked one also gets `clicks += 1`.
A record `{timestamp, round, displayed, clicked, reward}` is appended to `carousel["history"]`,
with `reward = 1` if something was clicked and `0` otherwise. Then `round_number += 1` and a
new set is sampled. "Resample without update" only redraws and records nothing.

**Beta posterior update.** Beta is conjugate to the Bernoulli click model. For each displayed
alternative: if clicked, `alpha += 1`; if displayed but not clicked, `beta += 1`. Alternatives
that were not displayed are not changed. After s clicks and f non-clicks the posterior is
Beta(1 + s, 1 + f), with mean alpha / (alpha + beta) (`posterior_mean` in `get_stats_df`).
The champion (`get_champion_alt_id`) is the alternative with the largest posterior mean
(ties broken by clicks, then impressions, then lowest id).

**How session state connects the pages.** Streamlit re-runs a page script from the top on every
interaction, and normal variables are lost between runs, but `st.session_state` persists.
`main.py` calls `init_state()` and then `st.navigation([...]).run()`; each page also calls
`init_state()`, which creates, only once:
- `st.session_state.carousels`: a dict `carousel_id -> {name, alternatives (DataFrame with
  alpha, beta, impressions, clicks), history, round_number, current_display_ids, ...}`
- `st.session_state.active_carousel_id`: the currently selected carousel.

All pages read and write this same dict through helpers in `bandit_utils.py`:
- *Create Carousels* calls `create_carousel(...)`, which adds an entry and makes it active.
- *Interaction* calls `update_after_feedback(...)`, which changes the alternatives table and history.
- *Analytics* only reads (`get_stats_df`, `carousels_table`, `beta_pdf_points`) and plots
  posterior means, clicks, Beta curves, and cumulative reward from `history`.

`select_carousel_widget()` writes the chosen id back to `active_carousel_id`, so the selection
follows the user across pages. Data lives only in the browser session (the `TODO ... via API`
comments show that database persistence was planned but is not implemented), so a refresh
resets everything. The original uses the unseeded `random.betavariate`, so it is not reproducible.

## 2. Gaussian model

- **Reward:** R | A = i ~ Normal(μᵢ, σ²) with three alternatives. σ is known and shared;
  each μᵢ is unknown to the learner. Rewards are continuous and can be negative.
- **Prior:** μᵢ ~ Normal(0, 25) for every i, so prior precision = 1/25 = 0.04.
- **Reward precision:** τ = 1/σ².
- **Update** after reward r for the selected alternative i (Normal-Normal conjugacy):
  - new precision = old precision + τ
  - new mean = (old precision · old mean + τ · r) / new precision
  - posterior variance = 1 / precision
  
  Precision adds up with every observation, and the new mean is a precision-weighted average
  of the old belief and the new reward.
- **Each round** (`run_gaussian_bandit` in `gaussian_bandit.py`):
  1. Sample a possible mean from each current posterior Normal.
  2. Select the alternative with the largest sample (learned posteriors only).
  3. Generate one reward from Normal(true mean of the selected alternative, σ²). The true
     means are used only here, by the simulator.
  4. Update only the selected alternative's posterior.
- **Final recommendation:** the alternative with the largest posterior mean.
- **Reproducibility:** one `numpy.random.default_rng(seed)` generator drives both the posterior
  sampling and the reward simulation.
- **Validation:** σ must be positive and the number of rounds must be a positive integer
  (checked in `validate_inputs` and shown as an error on the page).

The page has inputs for three true means, σ, rounds, and seed, and shows each alternative's
true mean, posterior estimated mean, posterior std, and selection count, plus the total
observed reward, the final recommendation, cumulative selection curves, and posterior curves.

## 3. Experiments

Settings: true means [1, 2, 3], prior Normal(0, 25) for each mean, **seed = 42** for all runs.

| Experiment | σ | Rounds | Posterior means (Alt 1, 2, 3) | Selections (Alt 1, 2, 3) | Total reward |

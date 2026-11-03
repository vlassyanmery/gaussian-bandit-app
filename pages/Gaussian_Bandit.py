import numpy as np
import pandas as pd
import streamlit as st

from bandit_utils import init_state
from gaussian_bandit import PRIOR_MEAN, PRIOR_VAR, run_gaussian_bandit, validate_inputs

init_state()

st.title("Gaussian Bandit")
st.caption(
    "Thompson sampling for continuous rewards: R | A=i ~ Normal(μᵢ, σ²) with known σ. "
    f"Every unknown mean μᵢ has the prior Normal({PRIOR_MEAN:g}, {PRIOR_VAR:g})."
)

names = ["Alternative 1", "Alternative 2", "Alternative 3"]

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    mu1 = c1.number_input("True mean, Alternative 1", value=1.0, step=0.5)
    mu2 = c2.number_input("True mean, Alternative 2", value=2.0, step=0.5)
    mu3 = c3.number_input("True mean, Alternative 3", value=3.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    sigma = c4.number_input("Known σ (noise std)", value=1.0, step=0.1, format="%.3f")
    rounds = c5.number_input("Number of rounds", value=100, step=1, format="%d")
    seed = c6.number_input("Random seed", value=42, step=1, format="%d")

    run = st.button("Run", type="primary")

if run:
    try:
        validate_inputs(sigma, rounds)
        st.session_state["gaussian_result"] = run_gaussian_bandit(
            [mu1, mu2, mu3], sigma, int(rounds), int(seed)
        )
    except ValueError as e:
        st.session_state.pop("gaussian_result", None)
        st.error(str(e))

res = st.session_state.get("gaussian_result")
if res is not None:
    st.subheader("Results")

    df = pd.DataFrame(
        {
            "Alternative": names,
            "True mean": res.true_means,
            "Posterior estimated mean": res.post_means,
            "Posterior std": [v ** 0.5 for v in res.post_vars],
            "Selections": res.counts,
        }
    )
    st.dataframe(df, width="stretch", hide_index=True)

    best = res.recommendation
    m1, m2, m3 = st.columns(3)
    m1.metric("Total observed reward", f"{res.total_reward:.2f}")
    m2.metric("Final recommendation", names[best])
    m3.metric("Selections total", sum(res.counts))
    st.caption(
        f"Recommendation = largest posterior mean ({res.post_means[best]:.3f}). "
        f"Run settings: seed={res.seed}, σ={res.sigma:g}, rounds={res.n_rounds}."
    )

    st.markdown("**Cumulative selections per alternative**")
    choices = pd.Series(res.choices)
    cum = pd.DataFrame({n: (choices == i).cumsum() for i, n in enumerate(names)})
    cum.index = range(1, len(cum) + 1)
    st.line_chart(cum)

    st.markdown("**Posterior distribution of each mean**")
    sd_max = max(res.post_vars) ** 0.5
    lo = min(res.true_means + res.post_means) - 3 * sd_max - 1
    hi = max(res.true_means + res.post_means) + 3 * sd_max + 1
    x = np.linspace(lo, hi, 300)
    curves = []
    for n, m, v in zip(names, res.post_means, res.post_vars):
        dens = np.exp(-((x - m) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)
        curves.append(pd.DataFrame({"mu": x, "density": dens, "alternative": n}))
    st.vega_lite_chart(
        pd.concat(curves, ignore_index=True),
        {
            "mark": {"type": "line", "strokeWidth": 3},
            "encoding": {
                "x": {"field": "mu", "type": "quantitative", "title": "μ"},
                "y": {"field": "density", "type": "quantitative", "title": "Density"},
                "color": {"field": "alternative", "type": "nominal"},
            },
        },
        width="stretch",
    )

"""Shared code of ir_response_photon and ir_response_neutral_hadron (Fig. ir): mean |dF_W| of W jets for one added
photon (or neutral hadron) against its pT, 1 MeV - 10 GeV, for the original model and the models repaired by
fine-tuning, the SAE edit and the selected WSP edit; lower panel: the difference to the response to one added charged
hadron of the same pT, in the same model; dotted line: first percentile of the pT of that particle type in JetClass.
5000 W jets with two draws each; bands: standard error of the mean.

Input: results/figures/small/ir_response/<name>.npz / .json (scripts/analysis/soft_particle_scan.py with
--pt-min 0.001 --pt-max 10 --families photon "neutral hadron" "charged hadron"; scripts/evaluation/softer_particles.py
prints the numbers quoted in Sec. 4)."""
import json

import numpy as np

import style as S

MODELS = (("original_model", "original", "#222222"),
          ("finetuning_attention_only_both_neutral_types_seed0", "fine-tuning", S.MAGENTA),
          ("sae_three_readers_b7-6_b4-0_c1-2", "SAE edit", S.ORANGE),
          ("agents_target_informed_run2_edit_large", "WSP", S.BLUE))
BINS = np.geomspace(1e-3, 10, 21)  # 5 bins per decade


def curves(a, fam: str):
    """Per pT bin of the added particle, W jets: mean |dF_W| and its standard error (bins with > 20 entries)."""
    pt, dF = a[f"pt_{fam}_W"], a[f"dF_{fam}_W"].astype(np.float64)
    idx = np.digitize(pt, BINS) - 1
    x = np.sqrt(BINS[1:] * BINS[:-1])
    val, err = np.full(len(x), np.nan), np.full(len(x), np.nan)
    y = np.abs(dF)
    for i in range(len(x)):
        m = idx == i
        if m.sum() > 20:
            val[i], err[i] = y[m].mean(), y[m].std() / np.sqrt(m.sum())
    return x, val, err


def load(results):
    d = S.fig_dir(results) / "ir_response"
    return [(dict(np.load(d / f"{n}.npz")), json.loads((d / f"{n}.json").read_text()), lab, c)
            for n, lab, c in MODELS if (d / f"{n}.npz").exists()]


def draw(plt, results, neutral: str, name: str, out) -> None:
    from matplotlib.lines import Line2D
    models = load(results)
    h = S.H2 + 0.95
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(0.49 * S.TEXTWIDTH, h), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.06})
    hs, labs = [], []
    for a, res, lab, c in models:
        xs, vn, en = curves(a, neutral)
        _, vc, ec = curves(a, "charged hadron")
        ax.fill_between(xs, vn - en, vn + en, color=c, alpha=0.2, lw=0)
        hs += ax.plot(xs, vn, color=c, lw=1.3)
        labs.append(lab)
        r, re = vn - vc, np.hypot(en, ec)
        rax.fill_between(xs, r - re, r + re, color=c, alpha=0.2, lw=0)
        rax.plot(xs, r, color=c, lw=1.3)
    q01 = models[0][1]["families"][neutral]["data"]["pt_q01"]
    for a_ in (ax, rax):
        a_.axvline(q01, color="k", lw=1.0, ls=":")
    hs.append(Line2D([], [], color="k", lw=1.0, ls=":"))
    labs.append("1st percentile")
    S.log_x(rax, 1e-3, 10)
    ax.set_ylim(0, 3.0)
    ax.set_ylabel(r"mean $|\Delta F_W|$")
    rax.axhline(0, color=S.GREY, lw=1.2)
    rax.set_ylabel(S.tx("$|\\Delta F_W^\\text{neutral}|$\n$-\\,|\\Delta F_W^\\text{charged}|$"), linespacing=1.1)
    rax.set_xlabel(rf"$p_T$ of the added {neutral} [GeV]")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    S.top_legend(ax, hs, labs, ncol=3, handlelength=1.3, columnspacing=0.7, handletextpad=0.4)
    S.finish(fig, h, top=S.em(0.56))
    S.keep_inside(fig, rax)
    fig.align_ylabels([ax, rax])
    S.save(fig, out, name)
    plt.close(fig)

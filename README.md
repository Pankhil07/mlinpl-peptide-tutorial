# Controllable Generative AI for Sequence Design

### Steering a peptide language model toward antimicrobial activity

**ML in PL 2025 — hands-on tutorial (25 min)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pankhil07/mlinpl-peptide-tutorial/blob/main/mlinpl-controllable-peptide-design.ipynb)

---

Generative models can write plausible peptides all day. That is not the hard
part. The hard part is writing peptides that are *plausible **and** do what you
want* — and this tutorial is about the cheapest honest way to close that gap.

It uses [Hyformer](https://github.com/szczurek-lab/hyformer), a joint
transformer with one shared backbone and two heads: a language-modelling head
that **generates** sequences and a prediction head that **scores** them. A task
token at position 0 selects which. The thing that proposes and the thing that
judges are the same network.

Target property: **MIC** (minimum inhibitory concentration) against
*Escherichia coli*, in log₂ µM. Lower = more potent.

## What it covers

| § | Topic |
|---|-------|
| 1 | Why you cannot enumerate sequence space |
| 2 | Unconditional generation — plausible ≠ useful |
| 3 | **Knob 1** — decoding parameters, and why they can't target a property |
| 4 | **Knob 2** — a property predictor sharing the generator's backbone |
| 5 | **Knob 3** — best-of-k selection, and the potency↔diversity trade |
| 6 | **Knob 4** — fine-tuning on the winners, so the *distribution* moves |
| 7 | **Knob 5** — oracle-guided search (TASAR), steering *during* generation |
| 8 | Evaluation and failure modes: Goodhart, OOD, absent uncertainty |

Knobs 1–4 all let the model finish a sequence before the oracle speaks. TASAR
puts the oracle *inside* generation: stochastic beam search without replacement,
with the tree's log-probabilities reweighted by each sequence's advantage, so
every oracle call informs the next.

The last section is the point of the tutorial, not an appendix. Every number
the notebook produces is a model prediction, and §7 is about how to stay honest
about that.

## Quick start

Click the Colab badge above — the first cell clones this repo and installs it,
the second pulls two public checkpoints (~610 MB) from HuggingFace. A free-tier
T4 is plenty.

To run locally instead:

```bash
git clone https://github.com/Pankhil07/mlinpl-peptide-tutorial.git
cd mlinpl-peptide-tutorial
conda env create -f env.yml && conda activate ohamrhyf
pip install -e . && pip install huggingface_hub
jupyter lab mlinpl-controllable-peptide-design.ipynb
```

CPU works too — drop `POOL_SIZE` in §2 to around 400.

## Checkpoints

Both are public on HuggingFace under BSD-3-Clause:

| | model | what it is |
|---|---|---|
| generator | [`SzczurekLab/hyformer_peptides_34M`](https://huggingface.co/SzczurekLab/hyformer_peptides_34M) | pretrained on 3.5M general-purpose and antimicrobial peptides |
| predictor | [`SzczurekLab/hyformer_peptides_34M_MIC`](https://huggingface.co/SzczurekLab/hyformer_peptides_34M_MIC) | the same backbone jointly fine-tuned on MIC against *E. coli* |

The notebook downloads them automatically.

## What's in this repo

The notebook, plus the minimum subset of the Hyformer package it needs to run
(model, amino-acid tokenizer, configs, the TASAR sampler and the peptide rule
screens). It is a teaching copy, trimmed for a
25-minute session — for the full library, the baselines, and the molecular
experiments, use the upstream repo:

> **https://github.com/szczurek-lab/hyformer**

## Citation

```bibtex
@article{izdebski2025hyformer,
  title   = {Synergistic Benefits of Joint Molecule Generation and Property Prediction},
  author  = {Izdebski, Adam and Olszewski, Jan and Gawade, Pankhil and
             Koras, Krzysztof and Korkmaz, Serra and Rauscher, Valentin and
             Tomczak, Jakub M. and Szczurek, Ewa},
  journal = {arXiv preprint arXiv:2504.16559},
  year    = {2025}
}
```

## Scope and limitations

Everything here is a **model prediction**. Nothing in this notebook has touched
a bacterium. The MIC head's best validation MSE is 0.78 — a typical error near
±0.9 log₂ units, close to two-fold in MIC — and it reports no uncertainty.

The output is a *ranked list of hypotheses* for wet-lab triage, which is a far
better use of an assay budget than random screening. It is not a discovery.
Potency is also only one axis: a real candidate needs low haemolysis and
cytotoxicity, protease stability, and manufacturability, none of which are
modelled here.

## License

See [LICENSE](LICENSE). The upstream Hyformer release and the HuggingFace
checkpoints are BSD-3-Clause; Hyformer is © 2025 szczurek-lab.

`hyformer/generators/` vendors the TASAR search. The stochastic beam search
derives from [unique-randomizer](https://github.com/google-research/unique-randomizer)
(Apache-2.0) via [graphxform](https://github.com/grimmlab/graphxform) (MIT);
attribution is preserved in each file's docstring.

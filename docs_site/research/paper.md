# Paper

## Available Versions

UCEF papers are available in four formats:

| Format | Language | Pages | File |
|--------|----------|-------|------|
| IEEE | English | ~4 | `paper/ieee/ucef-en.pdf` |
| IEEE | Chinese | ~5 | `paper/ieee/ucef-cn.pdf` |
| Chinese Journal | English | ~10 | `paper/chinese-journal/ucef-en.pdf` |
| Chinese Journal | Chinese | ~8 | `paper/chinese-journal/ucef-cn.pdf` |

## Compiling

Papers use TeX Live. English papers use pdflatex, Chinese papers use xelatex:

```bash
# IEEE English
cd paper/ieee && pdflatex ucef-en && pdflatex ucef-en

# IEEE Chinese
cd paper/ieee && xelatex ucef-cn && xelatex ucef-cn

# Chinese Journal English
cd paper/chinese-journal && pdflatex ucef-en && pdflatex ucef-en

# Chinese Journal Chinese
cd paper/chinese-journal && xelatex ucef-cn && xelatex ucef-cn
```

## Citation

```bibtex
@article{he2026ucef,
  title={Breaking the Context Barrier: A Universal Context Extension Framework for LLMs},
  author={何红林},
  journal={arXiv preprint},
  year={2026}
}
```

## Experiment Report

The experiment report (`experiments/experiment-report.tex`) summarizes all simulated results in Chinese LaTeX format.

```bash
cd experiments && xelatex experiment-report && xelatex experiment-report
```

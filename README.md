# AGORA_Merger_PapersIX
Analysis codes for the two AGORA Papers IX (Part 1 and Part 2). The studies analyze the effects of a major merger on star formation and stellar morphology of a Milky Way-mass Galaxy Progenitor and compare the results across nine state-of-the-art galaxy simulation codes.

Links to papers:
- Part 1 (Effects on star formation): https://arxiv.org/abs/2607.21709
- Part 2 (Effects on stellar morphology): https://arxiv.org/abs/2607.21710

## Details of each file

*setup.py*: This module provides the shared setup and helper routines for analyzing the AGORA multi-code simulation suite, including the code list, yt-field name for each code, plotting parameters for each code, halo merger-tree and snapshot loading, and merger infall-timing calculations. It also defines derived yt fields—such as AGORA-normalized metallicity, cooling time, and free-fall time, so that datasets from the different simulation codes can be loaded and compared on a common footing.

- Part 1:
	- *Gas_phase_plot_at_different_merger_stages.py*: Figure 5 + Figure 6 
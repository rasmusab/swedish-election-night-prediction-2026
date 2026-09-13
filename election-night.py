# %% [markdown]
# # Swedish election 2026
# Run these cells in the project's uv environment, or use:
# `uv run update-report.py --environment production`
# The export command executes a fresh notebook and creates HTML with code hidden.

# %%
import os
from pathlib import Path
from IPython.display import HTML, Image, display
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from val2026.election_report import (ROOT, run_cycle, update_lock, header_html, party_table,
                             seat_summary_html, plot_bloc_seats, uncertainty_html, plot_bloc_uncertainty,
                             plot_shares, plot_coverage, plot_history, methods_html)
from val2026.forecast_uncertainty import DEFAULT_DRAWS

# Set these two values when running the notebook interactively.
environment = os.environ.get('VAL2026_ENVIRONMENT', 'production')
collect_latest = os.environ.get('VAL2026_OFFLINE', '0') != '1'
project_root = Path(os.environ.get('VAL2026_PROJECT_ROOT', str(ROOT)))
simulation_draws = int(os.environ.get('VAL2026_UNCERTAINTY_DRAWS', str(DEFAULT_DRAWS)))

# %%
# Pull one official snapshot, validate it, fit the frozen model and retain state.
# update-report.py already holds the lock throughout collection AND HTML export.
if os.environ.get('VAL2026_WRAPPER_LOCK') == '1':
    state = run_cycle(project_root, environment, collect=collect_latest, simulation_draws=simulation_draws)
else:
    with update_lock(project_root):
        state = run_cycle(project_root, environment, collect=collect_latest, simulation_draws=simulation_draws)
display(HTML(header_html(state)))

# %%
display(HTML(seat_summary_html(state)))
for plot in (plot_bloc_seats,):
    figure = plot(state)
    if figure is not None:
        png = BytesIO()
        figure.savefig(png, format='png', dpi=140)
        display(Image(data=png.getvalue()))
        plt.close(figure)
display(HTML(party_table(state)))
display(HTML(uncertainty_html(state)))
for plot in (plot_bloc_uncertainty, plot_history, plot_shares, plot_coverage):
    figure = plot(state)
    if figure is not None:
        png = BytesIO()
        figure.savefig(png, format='png', dpi=140)
        display(Image(data=png.getvalue()))
        plt.close(figure)

# %%
display(HTML(methods_html(state)))

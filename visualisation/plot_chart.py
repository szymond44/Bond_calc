import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def plot_fanchart(stats_dict, time_horizon, tolerance_percent=5, title=None, segments=None):
    mean_path = stats_dict["paths"]["mean"]
    worst_path = stats_dict["paths"]["worst"]
    best_path = stats_dict["paths"]["best"]

    months = np.arange(time_horizon)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    ax.fill_between(
        months,
        worst_path,
        best_path,
        color='#4A90E2',
        alpha=0.2,
        label=f'Przedział {tolerance_percent}% - {100 - tolerance_percent}%'
    )

    ax.plot(months, mean_path, color='#1C3D5A', linewidth=2.5, label='Średnia ścieżka')
    ax.plot(months, worst_path, color='#D0021B', linestyle='--', linewidth=1.5, label=f'Dolne {tolerance_percent}% (Pesymistycznie)')
    ax.plot(months, best_path, color='#7ED321', linestyle='--', linewidth=1.5, label=f'Górne {tolerance_percent}% (Optymistycznie)')

    if segments and len(segments) > 1:
        for seg in segments[:-1]:
            boundary = seg["end"]
            if 0 < boundary < time_horizon:
                ax.axvline(x=boundary, color='#000000', linestyle=':', linewidth=1, alpha=0.6, zorder=1)

        for i, seg in enumerate(segments):
            width_frac = (seg["end"] - seg["start"]) / time_horizon
            if width_frac < 0.035:
                continue  
            mid = (seg["start"] + seg["end"]) / 2
            row_y = 0.03 if i % 2 == 0 else 0.10
            ax.text(
                mid, row_y, seg["name"],
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom',
                fontsize=8.5, color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75),
            )

    ax.set_title(title or 'Projekcja Kapitału w Czasie (Monte Carlo Fanchart)', fontsize=14, pad=15)
    ax.set_xlabel('Miesiące', fontsize=12)
    ax.set_ylabel('Wartość portfela (PLN)', fontsize=12)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', ' ')))

    ax.grid(True, linestyle=':', alpha=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.legend(loc='upper left')
    fig.tight_layout()

    return fig


def plot_cohorts(cohort_means, time_horizon, title=None):
    months = np.arange(time_horizon)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    
    if not cohort_means:
        return fig
        
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(cohort_means) - 1)) for i in range(len(cohort_means))]
    
    ax.stackplot(months, *cohort_means, colors=colors, alpha=0.85)
    
    ax.set_title(title or 'Rozbicie kapitału na poszczególne wpłaty (DCA)', fontsize=14, pad=15)
    ax.set_xlabel('Miesiące', fontsize=12)
    ax.set_ylabel('Wartość portfela (PLN)', fontsize=12)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', ' ')))
    ax.grid(True, linestyle=':', alpha=0.6)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    fig.tight_layout()
    return fig

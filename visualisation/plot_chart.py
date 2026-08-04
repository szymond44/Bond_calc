import matplotlib.pyplot as plt
import numpy as np

def plot_fanchart(stats_dict, time_horizon, tolerance_percent=5):

    mean_path = stats_dict["paths"]["mean"]
    worst_path = stats_dict["paths"]["worst"]
    best_path = stats_dict["paths"]["best"]

    months = np.arange(time_horizon)

    plt.figure(figsize=(12, 6))

    plt.fill_between(
        months, 
        worst_path, 
        best_path, 
        color='#4A90E2', 
        alpha=0.2, 
        label=f'Przedział {tolerance_percent}% - {100 - tolerance_percent}%'
    )

    plt.plot(months, mean_path, color='#1C3D5A', linewidth=2.5, label='Średnia ścieżka')
    plt.plot(months, worst_path, color='#D0021B', linestyle='--', linewidth=1.5, label=f'Dolne {tolerance_percent}% (Pesymistycznie)')
    plt.plot(months, best_path, color='#7ED321', linestyle='--', linewidth=1.5, label=f'Górne {tolerance_percent}% (Optymistycznie)')

    plt.title('Projekcja Kapitału w Czasie (Monte Carlo Fanchart)', fontsize=14, pad=15)
    plt.xlabel('Miesiące', fontsize=12)
    plt.ylabel('Wartość portfela (PLN)', fontsize=12)

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(',', ' ')))
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()
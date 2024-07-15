import numpy as np
import statsmodels.api as sm


def draw_cdf_line(
    plt, line_data, lowerbd, upperbd, line_clr=None, line_style=None, line_marker=None, line_label=None, num_samples=100
):
    ecdf = sm.distributions.ECDF(line_data)
    x = np.linspace(lowerbd, upperbd, num=num_samples)
    y = ecdf(x)
    plt.plot(x, y, c=line_clr, linestyle=line_style, marker=line_marker, label=line_label)

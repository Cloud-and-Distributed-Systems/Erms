from math import ceil, floor, sqrt
from matplotlib import pyplot as plt


data = [
    {
        "draws": [{"data": "DF", "color": "b", "type": "line"}],
        "title": "xxx",
    }
]


def draw_figure(data, file):
    plt.clf()
    edge = ceil(sqrt(len(data)))
    _, axs = plt.subplots(edge, edge, figsize=(edge * 6, edge * 6), squeeze=False)

    for index, fig in enumerate(data):
        subplot = axs[floor(index / edge), index % edge]
        subplot.set_title(fig["title"])
        for draw in fig["draws"]:
            draw_data = draw["data"]
            draw_color = draw["color"]
            draw_type = draw["type"]
            if draw_type == "line":
                subplot.plot(draw_data["x"], draw_data["y"], color=draw_color)
            elif draw_type == "scatter":
                subplot.scatter(draw_data["x"], draw_data["y"], color=draw_color)
            elif draw_type == "axvline":
                subplot.axvline(x=draw_data, linestyle="--", color=draw_color)

    plt.savefig(file)

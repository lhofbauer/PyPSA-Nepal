"""
Script to run PyPSA-Nepal and analyse results


Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import logging
import os

import pandas as pd

pd.set_option("future.infer_string", False)

import geopandas as gpd
import matplotlib.pyplot as plt
import pypsa

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# %% Load network and perform checks

n = pypsa.Network()

n.import_from_csv_folder("../data/network")

# Check for issues
# n.sanitize()
issues = n.consistency_check()

logger.info(issues)


# %% Run model (optimization)

# Choose year
year = 2050
periods = n.snapshots[n.snapshots.get_level_values("period").isin([year])]


# Sampling to reduce complexity for testing
res = 3
# Select weeks to reduce complexity for testing
s = 8
e = 35
#snap = periods[168 * s : 168 * e : res]
n.set_snapshots(periods[168 * s : 168 * e : res])
n.snapshot_weightings.loc[:, :] = res

### run optimization

# add load shedding generators
# FIXME: there is currently a very small amount of load shedding for 2020 (in
# one hour), this requires further exploration, likely solved in model is
# properly calibrated.

n.optimize.add_load_shedding()
#n.calculate_dependent_values()

n.sanitize()
n.optimize(#snapshots =  snap,
    multi_investment_periods=False,
    include_objective_constant=False,
    solver_name="highs",
    solver_options={
        "solver": "ipx",
        "run_crossover": "off",
        # "user_bound_scale": -4,
        "random_seed": 123,
    },
)


n.export_to_csv_folder("../results/res/")


# if fixing capacities and re-running dispatch after changes (e.g., to explore
# resilience to certain shocks)
# fix capacities, add load shedding, constrain power plant/etc.

# n.optimize.fix_optimal_capacities()
# n.optimize.add_load_shedding()
# n.generators_t.p_max_pu.loc[:,"Gandaki_0 hydro False agg pp"] = 0
# n.lines_t.s_max_pu.loc[:,"Lumbini_0-Sudurpaschim_0"]=0

# optimize economic dispatch, check results
# n.optimize(snapshots=n.snapshots[-600:])



# %% Analyse results

# statistics
n.statistics.supply()
n.statistics.energy_balance()
n.statistics.optimal_capacity.plot()
n.statistics.installed_capacity.plot()

# FIXME: these plots are placeholders and need to be improved/re-implemented

n.statistics.energy_balance.plot("area")

# generation area plot for example time period
p_by_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T.iloc[1000:1300]

fig, ax = plt.subplots(figsize=(22, 4))
p_by_carrier.plot(
    kind="area",
    ax=ax,
    linewidth=0,
    cmap="tab20b",
)
ax.legend(ncol=5, loc="upper left", frameon=False)
ax.set_ylabel("Generation (MW)")
ax.set_xlabel("Time")
plt.xticks(rotation=45)
plt.savefig("gen.png")

# simple generation map
s = (
    n.generators_t.p.loc[year, :]
    .sum()
    .groupby([n.generators.bus, n.generators.carrier])
    .sum()
    / 3000
)
fig = plt.figure(figsize=(7, 7))
ax = plt.axes()
n.plot(ax=ax, bus_sizes=s / 3000)
plt.savefig("gen_map.png")

# %%
# improved map
# FIXME: to crosscheck and improve where relevant
def enhanced_network_plot_with_boundaries(n, year, figsize=(12, 8)):
    """
    Enhanced version of n.plot() with optimized capacities, generation data,
    boundaries, and summed capacities for same province pairs
    """

    # load province boundaries
    try:
        provinces = gpd.read_file(
            "../raw_data/boundaries/geoBoundaries-NPL-ADM1-all/geoBoundaries-NPL-ADM1_simplified.shp"
        )
    except FileNotFoundError:
        print("Boundary files not found. Proceeding without boundaries.")
        provinces = None

    # get generation data for chosen year
    snapshots = n.snapshots[n.snapshots.get_level_values("period") == year]
    gen_data = pd.concat([n.generators_t.p.loc[snapshots, :].sum(),
                          n.storage_units_t.p.loc[snapshots, :].sum()])
    # aggregate by bus, convert to GWh
    gen_by_bus = gen_data.groupby(pd.concat([n.generators.bus,
                                             n.storage_units.bus])).sum() / 1000
    # aggregate by bus, convert to GWh
    cap_data = pd.concat([n.generators.p_nom_opt,
                          n.storage_units.p_nom_opt])
    # remove load shedding capacities
    cap_data = cap_data.loc[[c for c in cap_data.index if "load shedding" not in c]]
    cap_by_bus = cap_data.groupby(pd.concat([n.generators.bus,
                                             n.storage_units.bus])).sum() / 1000

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot province boundaries
    if provinces is not None:
        provinces.boundary.plot(ax=ax, color="black", linewidth=1.2, alpha=0.8)

    # sum capacities for lines between same bus pairs
    line_capacities = {}

    # process AC transmission lines
    for line_name, line_data in n.lines.iterrows():
        bus0, bus1 = line_data["bus0"], line_data["bus1"]
        # create a sorted tuple to handle bidirectional connections
        bus_pair = tuple(sorted([bus0, bus1]))

        if bus_pair not in line_capacities:
            line_capacities[bus_pair] = {"capacity": 0, "type": "AC"}
        line_capacities[bus_pair]["capacity"] += line_data["s_nom_opt"]

    # process links if they exist (internal connections only)
    if hasattr(n, "links") and not n.links.empty:
        for link_name, link_data in n.links.iterrows():
            bus0, bus1 = link_data["bus0"], link_data["bus1"]
            # create a sorted tuple to handle bidirectional connections
            bus_pair = tuple(sorted([bus0, bus1]))

            if bus_pair not in line_capacities:
                line_capacities[bus_pair] = {"capacity": 0, "type": "Link"}
            line_capacities[bus_pair]["capacity"] += link_data["p_nom_opt"]
            line_capacities[bus_pair]["type"] = "Link"

    # enhanced transmission lines with summed capacities
    max_capacity = (
        max(conn["capacity"] for conn in line_capacities.values())
        if line_capacities
        else 1
    )
    min_capacity = (
        min(conn["capacity"] for conn in line_capacities.values())
        if line_capacities
        else 0
    )

    for (bus0, bus1), connection_data in line_capacities.items():
        if bus0 in n.buses.index and bus1 in n.buses.index:
            x0, y0 = n.buses.loc[bus0, ["x", "y"]]
            x1, y1 = n.buses.loc[bus1, ["x", "y"]]

            total_capacity = connection_data["capacity"]
            conn_type = connection_data["type"]

            # scale line width based on total capacity
            if max_capacity > min_capacity and total_capacity > 0:
                normalized_cap = (total_capacity - min_capacity) / (
                    max_capacity - min_capacity
                )
                line_width = 1 + normalized_cap * 6
            elif total_capacity > 0:
                line_width = 3
            else:
                line_width = 1  # Thin line for zero capacity connections

            # different styling for Links vs AC lines
            if conn_type == "Link":
                line_color = "blue" if total_capacity > 0 else "lightgray"
                alpha = 0.9 if total_capacity > 0 else 0.6
                line_style = "-" if total_capacity > 0 else "--"
            else:
                line_color = "darkred"
                alpha = 0.9
                line_style = "-"

            # draw enhanced line
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=line_color,
                linewidth=line_width,
                alpha=alpha,
                linestyle=line_style,
                zorder=3,
            )

            # add capacity labels only for non-zero capacities
            # FIXME: currently not plotting transmission capacities – data
            # to be checked
            # if total_capacity > 0:
            #     mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            #     label_text = f'{total_capacity:.0f}MW'

            #     ax.text(mid_x, mid_y, label_text, fontsize=8,
            #             ha='center', va='center', weight='bold',
            #             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow",
            #                     alpha=0.9), zorder=4)

    # enhanced buses
    max_cap = cap_by_bus.max() if len(cap_by_bus) > 0 else 1
    min_cap = cap_by_bus.min() if len(cap_by_bus) > 0 else 0

    for bus_name, bus_data in n.buses.iterrows():
        x, y = bus_data["x"], bus_data["y"]
        cap_value = cap_by_bus.get(bus_name, 0)

        # Size based on generation
        if max_cap > min_cap:
            normalized_cap = (cap_value - min_cap) / (max_cap - min_cap)
            size = 200 + normalized_cap * 1800
        else:
            size = 600

        # styling for province buses
        bus_color = "lightcoral"
        edge_color = "black"
        province_name = bus_name.replace("_0", "")
        label_text = f"{province_name}\n{cap_value:.1f} GW"

        # Enhanced bus visualization
        ax.scatter(
            x,
            y,
            s=size,
            c=bus_color,
            alpha=0.9,
            edgecolors=edge_color,
            linewidth=2,
            zorder=5,
        )

        # Bus labels with generation
        ax.text(
            x,
            y + 0.05,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                alpha=0.95,
            ),
            zorder=6,
        )

    # enhanced formatting
    # ax.set_title(f'Overview map - y\n(optimized power system)',
    #             fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.grid(True, alpha=0.3, zorder=1)

    # set appropriate axis limits for Nepal
    ax.set_xlim(79.5, 88.5)
    ax.set_ylim(25.5, 31.0)

    # summary statistics
    total_ac_capacity = sum(conn["capacity"] for conn in line_capacities.values())

    # • AC Lines: {total_ac_capacity:.0f} MW
    #• Buses: {len(n.buses)}
    #• Total Connections: {len(line_capacities)}
    #• Generation: {gen_by_bus.sum():.0f} GWh
    
    summary_text = f"""Network Summary ({year})
• Generation capacity: {cap_by_bus.sum():.0f} GW"""

    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        zorder=7,
    )

    plt.tight_layout()
    plt.show()

    return fig, ax


fig, ax = enhanced_network_plot_with_boundaries(n, year)

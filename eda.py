# %%
import plotly.graph_objects as go
import pycountry
from pandas import NA, DataFrame, Index, options, read_csv, to_datetime
from plotly.express import bar, line, scatter_geo
from statsmodels.tsa.filters.hp_filter import hpfilter

# %%
options.display.max_rows = None
options.display.max_columns = None
options.display.max_colwidth = None
options.display.max_seq_items = None


# %%
data = read_csv(
    "data/region_08.csv",
)
data.head()


# %%
data.shape


# %%
data.columns


# %%
data.dtypes


# %%
def separate_variables(df: DataFrame) -> tuple[Index, Index]:
    """Identify numeric and categorical columns.

    Args:
        df (DataFrame): Input Dataframe.

    Returns:
        tuple[Index[str], Index[str]]: Returns an index of numeric and catagrical cols.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns

    return numeric_cols, categorical_cols


# %%
numeric_cols, categorical_cols = separate_variables(data)
print("Numeric:", len(numeric_cols))  # noqa: T201
print("Categorical:", len(categorical_cols))  # noqa: T201


# %% [markdown]
#    ## Cardinality Analysis


# %%
def analyze_categorical_cardinality(df: DataFrame, input_cols: Index) -> None:
    """Calculate cardinality for each categorical column.

    Args:
        df (DataFrame): Input dataframe.
        input_cols (Index[str]): List of columns.
    """
    # Calculate cardinality for each categorical column
    cardinality_dict = {}
    for column in input_cols:
        # Get number of unique values (excluding nulls)
        unique_count = df[column].nunique()
        # Get number of missing values
        missing_count = df[column].isnull().sum()
        # Calculate percentage of unique values
        unique_percentage = (unique_count / df[column].count()) * 100
        # Get value counts for most common categories
        # Calculate percentage of unique values
        missing_percentage = (missing_count / len(df)) * 100
        # Get value counts for most common categories
        top_values = df[column].value_counts().head(5)

        cardinality_dict[column] = {
            "unique_count": unique_count,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "unique_percentage": unique_percentage,
            "top_values": top_values,
        }

    # Create a summary dataframe
    summary_data = {
        # "Unique_Count": [d["unique_count"] for d in cardinality_dict.values()],
        # "Missing_Count": [d["missing_count"] for d in cardinality_dict.values()],
        "Missing_Percentage": [d["missing_percentage"] for d in cardinality_dict.values()],
        "Unique_Percentage": [d["unique_percentage"] for d in cardinality_dict.values()],
        # "Top": [d["top_values"] for d in cardinality_dict.values()],
    }

    summary_df = DataFrame(summary_data, index=input_cols)
    summary_df = summary_df.sort_values("Unique_Percentage", ascending=False)

    # Print detailed analysis
    print("=== Categorical Variables Cardinality Analysis ===\n")  # noqa: T201
    print(summary_df)  # noqa: T201


# %% [markdown]
#    ### Categorical Cols

# %%
analyze_categorical_cardinality(data, categorical_cols)


# %% [markdown]
#    ### Numeric Cols

# %%
analyze_categorical_cardinality(data, numeric_cols)


# %% [markdown]
#    ## Duplicate Analysis

# %%
any(data.duplicated(subset=["eventid"]))


# %% [markdown]
#  ## Fixing Missing Dates
#
#  - Back Fill if approx date is not valid. (As data is ordered by date.)
#
#  - Fill with approx date if valid.

# %%
data["event_date"] = to_datetime(
    data[["iyear", "imonth", "iday"]].rename(columns={"iyear": "year", "imonth": "month", "iday": "day"}),
    errors="coerce",
)
approx_dates1 = to_datetime(data["approxdate"], errors="coerce", format="%m/%d/%Y")
approx_dates2 = to_datetime(data["approxdate"], errors="coerce", format="%B %d, %Y")
approx_dates3 = to_datetime(data["approxdate"], errors="coerce", format="%Y-%m-%d %H:%M:%S")

data["event_date"] = data[approx_dates1.isnull() & approx_dates2.isnull() & approx_dates3.isnull()][
    "event_date"
].bfill()

data = data.drop(["iyear", "imonth", "iday"], axis=1)

data["event_date"] = data["event_date"].fillna(approx_dates1)
data["event_date"] = data["event_date"].fillna(approx_dates2)
data["event_date"] = data["event_date"].fillna(approx_dates3)


# %% [markdown]
#    ## Number of events by date.

# %%
# Sort by event date.
data = data.sort_values("event_date")

# Create a daily time series: count events per day
daily_counts = data.groupby("event_date").agg(count=("eventid", "count"))
daily_counts = daily_counts.asfreq("D", fill_value=0)  # Ensure continuous dates

# Aggregate to monthly counts.
monthly_counts = daily_counts["count"].resample("ME").sum()

# Aggregate to yearly counts.
yearly_counts = daily_counts["count"].resample("YE").sum()

# Detecting Trend Using a Hodrick-Prescott Filter
sw_cycle, sw_trend = hpfilter(yearly_counts, lamb=100)
# sw_trend.plot(figsize=(10, 5)).autoscale(axis="x", tight=True)

fig = go.Figure()

# Daily view: raw
fig.add_trace(
    go.Scatter(
        x=daily_counts.index,
        y=daily_counts["count"],
        mode="lines",
        name="Daily Count",
        visible=False,
    ),
)

# Monthly view: aggregated
fig.add_trace(
    go.Scatter(
        x=monthly_counts.index,
        y=monthly_counts,
        mode="lines",
        name="Monthly Count",
        visible=False,
    ),
)

# Yearly view: aggregated
fig.add_trace(
    go.Scatter(
        x=yearly_counts.index,
        y=yearly_counts,
        mode="lines",
        name="Yearly Count",
        visible=True,
    ),
)

# Yearly view: Trend
fig.add_trace(
    go.Scatter(
        x=sw_trend.index,
        y=sw_trend,
        mode="lines",
        name="Trend",
        visible=True,
        line={"color": "#c7947c"},
    ),
)

# --- 3. Add update buttons to toggle between views ---

fig.update_layout(
    updatemenus=[
        {
            "active": 0,
            "buttons": [
                {
                    "label": "Yearly",
                    "method": "update",
                    "args": [{"visible": [False, False, True, True]}, {"title": "Yearly Event Counts"}],
                },
                {
                    "label": "Monthly",
                    "method": "update",
                    "args": [{"visible": [False, True, False, False]}, {"title": "Monthly Event Counts"}],
                },
                {
                    "label": "Daily",
                    "method": "update",
                    "args": [{"visible": [True, False, False, False]}, {"title": "Daily Event Counts"}],
                },
            ],
            "x": 1.0,
            "xanchor": "right",
            "y": 1.1,
            "yanchor": "bottom",
        },
    ],
)

# Add range slider and selector on the x-axis
fig.update_layout(
    xaxis={
        "title": "Date",
        "rangeslider": {"visible": True},
        "rangeselector": {
            "buttons": [
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                {"step": "all"},
            ],
        },
        "type": "date",
    },
    yaxis={"title": "Event Count"},
    title="Event Counts Over Time",
)

fig.show()


# %% [markdown]
#    The nuber of events peaked in 1979.
#
#
#
#
#
#
#
#    No data or no attacks in the year 1993.
#
#
#
#
#
#
#
#   The trend shows a rise in recent years.

# %% [markdown]
#  ## Events by Region

# %%
data["region"].unique()


# %%
data["region_txt"].unique()


# %% [markdown]
#  This data is only for Western Europe.

# %% [markdown]
# ### Incidents each year (by couuntry)

# %%
data["year"] = data["event_date"].dt.year
yearly_counts = data.groupby(["year", "country", "country_txt"]).size().reset_index(name="count")
yearly_counts["total_py"] = yearly_counts.groupby("year")["count"].transform("sum")
yearly_counts["proportion"] = yearly_counts["count"] / yearly_counts["total_py"]
# yearly_counts.head(20)


# %%
yearly_counts.head()

# %%
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

codes = {country: countries.get(country, "Unknown code") for country in data["country_txt"].unique()}
codes["West Germany (FRG)"] = "DEU"
codes["Vatican City"] = "VAT"

yearly_counts["codes"] = yearly_counts["country_txt"].apply(lambda x: codes[x])


# %%
fig = scatter_geo(
    yearly_counts,
    locations="codes",
    hover_name="country_txt",
    size="proportion",
    animation_frame="year",
    projection="natural earth",
    title="Incidents by year",
)

# fig.update_geos(fitbounds="locations")

fig.update_geos(
    visible=True,
    resolution=110,
    scope="europe",
    showcountries=True,
)

fig.show()


# %% [markdown]
# ### Total incidents (by country)

# %%
total_counts = data.groupby("country_txt").size().reset_index(name="counts")
total_counts = total_counts.sort_values(by="counts")

# %%
fig = bar(
    total_counts,
    x="country_txt",
    y="counts",
    labels={"country_txt": "Countries", "counts": "Incidents"},
    title="Incident per country",
)
fig.show()

# %% [markdown]
# ### Incident Locations (by country)

# %%
years = sorted(data["year"].unique())
countries = sorted(data["country_txt"].unique())

initial_year = years[0]
df_initial = data[data["year"] == initial_year]

# Create the figure and add one trace per country (for the initial year)
fig = go.Figure()

for country in countries:
    df_country = df_initial[df_initial["country_txt"] == country]
    fig.add_trace(
        go.Scattergeo(
            lon=df_country["longitude"],
            lat=df_country["latitude"],
            text=df_country["event_date"],
            mode="markers",
            marker={"size": 8, "opacity": 0.7},
            name=country,
        ),
    )

# Build animation frames - one frame for each year
frames = []
for year in years:
    frame_traces = []
    df_year = data[data["year"] == year]
    for country in countries:
        df_country = df_year[df_year["country_txt"] == country]
        # Each trace in the frame corresponds to the country trace created above
        frame_traces.append(
            go.Scattergeo(
                lon=df_country["longitude"],
                lat=df_country["latitude"],
                text=df_country["event_date"],
                mode="markers",
                marker={"size": 8, "opacity": 0.7},
                name=country,
            ),
        )
    frames.append(go.Frame(data=frame_traces, name=str(year)))

fig.frames = frames

# Create a slider that will step through each year
slider_steps = []
for year in years:
    step = {
        "method": "animate",
        "args": [[str(year)], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
        "label": str(year),
    }
    slider_steps.append(step)

sliders = [
    {
        "active": 0,
        "currentvalue": {"prefix": "Year: "},
        "pad": {"t": 50},
        "steps": slider_steps,
    },
]

# Create a dropdown menu for filtering by country.
# When "All" is selected, all country traces are visible.
# When a specific country is selected, only that trace is shown.
updatemenus = [
    {
        "buttons": list(
            [
                {
                    "args": [{"visible": [True] * len(countries)}],
                    "label": "All",
                    "method": "update",
                },
            ]
            + [
                {
                    "args": [{"visible": [c == country for c in countries]}],
                    "label": country,
                    "method": "update",
                }
                for country in countries
            ],
        ),
        "direction": "down",
        "pad": {"r": 10, "t": 10},
        "showactive": True,
        "x": 0.1,
        "xanchor": "left",
        "y": 1.15,
        "yanchor": "top",
    },
]

# Update the layout with the geo map settings, slider, and dropdown menu
fig.update_layout(
    title="Terrorist Events in Western Europe",
    geo={
        "scope": "europe",
        "projection_type": "natural earth",
    },
    sliders=sliders,
    updatemenus=updatemenus,
)

# Display the figure
fig.show()


# %% [markdown]
# ## Incident Criterias

# %%
data["doubtterr"] = data["doubtterr"].replace(-9, NA)

# %%
crit_counts = data.groupby("year")[["crit1", "crit2", "crit3", "doubtterr"]].sum().reset_index()
crit_counts.head()

# %%
fig = go.Figure()

legend_labels = {
    "crit1": "Political, Economic,<br>Religious, or Social Goal",
    "crit2": "Intention to Coerce,Intimidate<br>or Publicize to Larger Audience",
    "crit3": "Outside International<br>Humanitarian Law",
    "doubtterr": "Doubt Terrorism Proper?",
}

for col in ["crit1", "crit2", "crit3", "doubtterr"]:
    fig.add_trace(go.Scatter(x=crit_counts["year"], y=crit_counts[col], mode="lines+markers", name=legend_labels[col]))

fig.update_layout(
    title="Different inclusion criterias over the years",
    xaxis_title="Year",
    yaxis_title="Count",
    xaxis={"showspikes": True},
    hovermode="x unified",
    legend={
        "yanchor": "top",
        "y": 0.99,
        "xanchor": "right",
        "x": 0.99,
        "bgcolor": "rgba(0,0,0,0)",
    },
)


fig.show()

# %% [markdown]
# #### Categorization of the incident other than terrorism

# %%
alt_counts = data[data["doubtterr"] == 1].groupby(["year", "alternative_txt"]).size().reset_index(name="count")

# %%
fig = line(
    alt_counts,
    x="year",
    y="count",
    color="alternative_txt",
    symbol="alternative_txt",
    labels={"alternative_txt": "Alternative Designation"},
)
fig.update_traces(hovertemplate="%{y}")
fig.update_layout(xaxis={"showspikes": True}, hovermode="x unified")

fig.show()

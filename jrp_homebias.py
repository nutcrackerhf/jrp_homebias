import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Home Bias vs FX Hedge Simulator", layout="wide")
st.title("Home Bias vs FX Hedging: Volatility and Tracking Error Tradeoffs")

assets = ['US', 'EU', 'Japan', 'EM']

st.sidebar.header("Capital Market Assumptions")

cov_matrix_for_bl = None  # placeholder, will be built after required inputs
port_vol = None
market_risk_premium = st.sidebar.number_input("Market Risk Premium (e.g., 6%)", value=0.06)
# compute expected returns later after defining cov_matrix







local_vols = np.array([
    st.sidebar.number_input(f"Local Vol {asset}", value=val)
    for asset, val in zip(assets, [0.14, 0.15, 0.13, 0.19])
])

fx_vols = np.array([
    st.sidebar.number_input(f"FX Vol {asset}", value=val)
    for asset, val in zip(assets, [0.00, 0.08, 0.10, 0.12])
])

us_weight = st.sidebar.number_input("US Benchmark Weight", min_value=0.0, max_value=1.0, value=0.54)
non_us_base = np.array([0.20, 0.15, 0.11])
non_us_base /= non_us_base.sum()
remaining_weight = 1.0 - us_weight
benchmark_weights = np.array([us_weight] + list(remaining_weight * non_us_base))

eq_corr = np.array([
    [1.0, 0.85, 0.65, 0.75],
    [0.85, 1.0, 0.70, 0.75],
    [0.65, 0.70, 1.0, 0.65],
    [0.75, 0.75, 0.65, 1.0]
])

def build_cov_matrix(fx_hedge_ratio):
    effective_vols = np.sqrt(local_vols**2 + ((1 - fx_hedge_ratio)**2) * fx_vols**2)
    return np.outer(effective_vols, effective_vols) * eq_corr

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

def true_tracking_error(weights, mean_returns, cov_matrix, bench_returns, bench_cov):
    diff_cov = cov_matrix + bench_cov - 2 * np.sqrt(cov_matrix * bench_cov)
    return np.sqrt(np.dot(weights, np.dot(diff_cov, weights)))



# Build benchmark covariance matrix now that inputs are defined
cov_matrix_for_bl = build_cov_matrix(np.zeros(4))
port_vol = np.sqrt(np.dot(benchmark_weights.T, np.dot(cov_matrix_for_bl, benchmark_weights)))
lambda_risk_aversion = market_risk_premium / (port_vol ** 2)
implied_returns = lambda_risk_aversion * np.dot(cov_matrix_for_bl, benchmark_weights)

st.subheader("Implied Expected Returns")
st.write(pd.DataFrame({"Asset": assets, "Implied Return": implied_returns.round(4)}))

st.sidebar.subheader("Adjust Implied Returns")
bl_adjustments = []
for i, asset in enumerate(assets):
    adj = st.sidebar.number_input(f"{asset} Return Adj (bps)", value=0, step=5)
    bl_adjustments.append(implied_returns[i] + adj / 10000)

expected_returns = np.array(bl_adjustments)

# Simulation
combined_data = []
us_base = benchmark_weights[0]
us_ow_range = np.linspace(0, 0.20, 21)
for ow in us_ow_range:
    weights = benchmark_weights.copy()
    weights[0] += ow
    weights[1:] -= ow * benchmark_weights[1:] / benchmark_weights[1:].sum()
    fx_hedge = np.zeros(4)
    fx_costs = np.array([0.002, 0.002, 0.002, 0.0])
    adj_returns = expected_returns - fx_costs * fx_hedge
    cov_matrix = build_cov_matrix(fx_hedge)
    ret, vol = portfolio_performance(weights, adj_returns, cov_matrix)
    active = weights - benchmark_weights
    te = np.sqrt(np.dot(active, np.dot(cov_matrix, active)))
    combined_data.append({
        'X Axis': round(weights[0] - us_base, 4),
        'Volatility': round(vol, 4),
        'Expected Return': round(ret, 4),
        'Tracking Error': round(te, 4),
        'Size': round(te, 4),
        'Strategy': 'US Allocation OW'
    })

hedge_range = np.linspace(0, 0.20, 21)
for h in hedge_range:
    fx_hedge = np.array([h, h, h, 0.0])
    fx_costs = np.array([0.002, 0.002, 0.002, 0.0])
    adj_returns = expected_returns - fx_costs * fx_hedge
    cov_matrix_hedged = build_cov_matrix(fx_hedge)
    weights = benchmark_weights.copy()
    ret, vol = portfolio_performance(weights, adj_returns, cov_matrix_hedged)
    cov_matrix_unhedged = build_cov_matrix(np.zeros(4))
    te = true_tracking_error(weights, adj_returns, cov_matrix_hedged, expected_returns, cov_matrix_unhedged)
    combined_data.append({
        'X Axis': round(h, 4),
        'Volatility': round(vol, 4),
        'Expected Return': round(ret, 4),
        'Tracking Error': round(te, 4),
        'Size': round(te, 4),
        'Strategy': 'FX Hedge Ratio'
    })

# Plot
combined_df = pd.DataFrame(combined_data)

# Individual Plots
us_df = combined_df[combined_df['Strategy'] == 'US Allocation OW']
us_fig = px.scatter_3d(us_df, x='X Axis', y='Volatility', z='Expected Return',
                       size='Size', color='Tracking Error',
                       title='US Allocation Overweight Tradeoff')
us_fig.update_layout(scene=dict(xaxis_title='Relative US Overweight'))
st.plotly_chart(us_fig, use_container_width=True)

fx_df = combined_df[combined_df['Strategy'] == 'FX Hedge Ratio']
fx_fig = px.scatter_3d(fx_df, x='X Axis', y='Volatility', z='Expected Return',
                       size='Size', color='Tracking Error',
                       title='FX Hedge Ratio Tradeoff')
fx_fig.update_layout(scene=dict(xaxis_title='FX Hedge Ratio'))
st.plotly_chart(fx_fig, use_container_width=True)


combined_df = pd.DataFrame(combined_data)
fig = px.scatter_3d(combined_df, x='X Axis', y='Volatility', z='Expected Return',
                    size='Size', color='Strategy',
                    title='Volatility vs Tracking Error Tradeoff')
fig.update_layout(scene=dict(
    xaxis_title='Relative US OW or FX Hedge Ratio',
    xaxis=dict(range=[0, 0.2])
))
st.plotly_chart(fig, use_container_width=True, height=800)

# Illustration: Impact of US OW Depends on Benchmark Composition
st.subheader("Illustrating US Overweight Effects at Different Benchmarks")

illustration_benchmarks = [us_weight, 0.2, 0.9]  # Use current user-selected US benchmark weight
illustration_ow = 0.03  # +3% overweight
illustration_fx = np.zeros(4)

illustration_table = []
for base_us in illustration_benchmarks:
    non_us_base = np.array([0.20, 0.15, 0.11])
    non_us_base /= non_us_base.sum()
    remaining_weight = 1.0 - base_us
    base_weights = np.array([base_us] + list(remaining_weight * non_us_base))
    overweight_weights = base_weights.copy()
    overweight_weights[0] += illustration_ow
    overweight_weights[1:] -= illustration_ow * base_weights[1:] / base_weights[1:].sum()

    cov_matrix = build_cov_matrix(illustration_fx)
    ret_base, vol_base = portfolio_performance(base_weights, expected_returns, cov_matrix)
    ret_ow, vol_ow = portfolio_performance(overweight_weights, expected_returns, cov_matrix)

    illustration_table.append({
        'Base US Weight': base_us,
        'Vol Base': round(vol_base, 4),
        'Vol OW +3%': round(vol_ow, 4),
        'Δ Volatility': round(vol_ow - vol_base, 4),
        'Return Base': round(ret_base, 4),
        'Return OW +3%': round(ret_ow, 4),
        'Δ Return': round(ret_ow - ret_base, 4),
        'Tracking Error Base': 0.0 if base_us != us_weight else np.sqrt(np.dot(base_weights - benchmark_weights, np.dot(cov_matrix, base_weights - benchmark_weights))),
        'Tracking Error OW+3%': np.sqrt(np.dot(overweight_weights - benchmark_weights, np.dot(cov_matrix, overweight_weights - benchmark_weights)))
    })

st.dataframe(pd.DataFrame(illustration_table))

# User-selected comparison rows
st.subheader("Scenario Comparison")
us_selected = st.selectbox("Select US Overweight (%)", us_df['X Axis'].unique(), index=3)
fx_selected = st.selectbox("Select FX Hedge Ratio (%)", fx_df['X Axis'].unique(), index=5)

base_row = combined_df[(combined_df['X Axis'] == 0.0) & (combined_df['Strategy'] == 'US Allocation OW')].iloc[0]
us_row = us_df[us_df['X Axis'] == us_selected].iloc[0]
fx_row = fx_df[fx_df['X Axis'] == fx_selected].iloc[0]

delta_table = pd.DataFrame([
    {
        'Strategy': 'US OW vs Base',
        'Δ Volatility': us_row['Volatility'] - base_row['Volatility'],
        'Δ Return': us_row['Expected Return'] - base_row['Expected Return'],
        'Δ Tracking Error': us_row['Tracking Error'] - base_row['Tracking Error'],
        'Δ Sharpe': (us_row['Expected Return'] / us_row['Volatility']) - (base_row['Expected Return'] / base_row['Volatility']) if base_row['Volatility'] != 0 else np.nan
    },
    {
        'Strategy': 'FX Hedge vs Base',
        'Δ Volatility': fx_row['Volatility'] - base_row['Volatility'],
        'Δ Return': fx_row['Expected Return'] - base_row['Expected Return'],
        'Δ Tracking Error': fx_row['Tracking Error'] - base_row['Tracking Error'],
        'Δ Sharpe': (fx_row['Expected Return'] / fx_row['Volatility']) - (base_row['Expected Return'] / base_row['Volatility']) if base_row['Volatility'] != 0 else np.nan
    }
])

st.dataframe(delta_table.round(6))

# Display dataframe and offer CSV download
st.subheader("Scenario Results Table")
st.dataframe(combined_df)
st.download_button(
    label="Download Results as CSV",
    data=combined_df.to_csv(index=False).encode('utf-8'),
    file_name='home_bias_fx_hedge_results.csv',
    mime='text/csv'
)

import os
import datapane as dp
import pandas as pd
import numpy as np
import requests
from io import StringIO
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
import datetime
import json
import math
from scipy.stats import linregress
import seaborn as sns
import warnings

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import Files
import data_format

from data_definitions import (
    tickers, market_data_start_date, moving_avg_metrics, fiat_money_data_top10,
    gold_silver_supply, gold_supply_breakdown, stock_tickers, today, yesterday,
    report_date, filter_data_columns, stats_start_date, valuation_data_metrics,
    valuation_metrics, volatility_windows, correlation_data,
    
)

# Fetch the data
data = data_format.get_data(tickers, market_data_start_date)
data = data_format.calculate_custom_on_chain_metrics(data)
data = data_format.calculate_moving_averages(data, moving_avg_metrics)
data = data_format.calculate_btc_price_to_surpass_fiat(data, fiat_money_data_top10)
data = data_format.calculate_metal_market_caps(data, gold_silver_supply)
data = data_format.calculate_gold_market_cap_breakdown(data, gold_supply_breakdown)
data = data_format.calculate_btc_price_for_stock_mkt_caps(data, stock_tickers)
data = data_format.calculate_stock_to_flow_metrics(data)

# Forward fill the data for all columns
data.ffill(inplace=True)

# Flatten the list of columns from the dictionary
columns_to_keep = [item for sublist in filter_data_columns.values() for item in sublist]

# Filter the dataframe
filter_data = data[columns_to_keep]

# Run Data Analysis On Report Data
report_data = data_format.run_data_analysis(filter_data, stats_start_date)

# Get Bitcoin Difficulty Blockchain Data
difficulty_report = data_format.check_difficulty_change()

# Calcualte Difficulty Period Changes
difficulty_period_changes = data_format.calculate_metrics_change(difficulty_report, report_data)

# Format Bitcoin Difficulty Blockchain Data Output To Pandas
difficulty_report = pd.DataFrame([difficulty_report])

# Calcualte 52 Week High Low Based On Report Timeframe
weekly_high_low = data_format.calculate_52_week_high_low(report_data, report_date)

# Calcualte Valuation Target Data
valuation_data = data_format.create_valuation_data(report_data, valuation_metrics, report_date)

# Calcualte Grwoth Rate Data
cagr_results = data_format.calculate_rolling_cagr_for_all_metrics(data)

# Calcuate Sharpe Ratio Data
sharpe_data = data[correlation_data]
sharpe_results = data_format.calculate_daily_sharpe_ratios(sharpe_data)

# Calcuate Correlations
correlation_data = data[correlation_data]
# Drop NA Values
correlation_data = correlation_data.dropna()
# Calculate Bitcoin Correlations
correlation_results = data_format.create_btc_correlation_tables(report_date, tickers, correlation_data)

# Import Report Table Functions
import report_tables

# Create the difficulty update table
difficulty_update_table = report_tables.create_difficulty_update_table(report_data, difficulty_report, report_date)

# Create the difficulty big numbers
difficulty_big_numbers = report_tables.create_difficulty_big_numbers(difficulty_update_table)

# Create the difficulty big bumbers block
difficulty_update_summary_dp = dp.Table(difficulty_update_table, name='Difficulty_Summary')

# Create the performance table
performance_table = report_tables.create_performance_table(report_data, difficulty_period_changes,report_date,weekly_high_low,cagr_results,sharpe_results,correlation_results)

# Create the styled performance table
styled_performance_table = report_tables.style_performance_table(performance_table)

# Create a DataPane table with the styled table
performance_table_dp = dp.Table(styled_performance_table, name='Performance_Table')

# Create the fundamentals table
fundamentals_table = report_tables.create_bitcoin_fundamentals_table(report_data, difficulty_period_changes, weekly_high_low,report_date,cagr_results)

# Create the styled fundamentals table
styled_fundamentals_table = report_tables.style_bitcoin_fundamentals_table(fundamentals_table)

# Create a DataPane table with the styled table
fundamentals_table_dp = dp.Table(styled_fundamentals_table, name='Fundamentals_Table')

# Create the valuation table
valuation_table = report_tables.create_bitcoin_valuation_table(report_data, difficulty_period_changes, weekly_high_low, valuation_data, report_date)

# Create the styled valuation table
styled_valuation_table = report_tables.style_bitcoin_valuation_table(valuation_table)

# Create a DataPane table with the styled table
valuation_table_dp = dp.Table(styled_valuation_table, name='Valuation_Table')

# Datapane Report Imports
from datapane_report import generate_report_layout

# Configure Datapane Report
report_layout = generate_report_layout(difficulty_big_numbers,performance_table_dp,fundamentals_table_dp,valuation_table_dp)

# DataPane Styling
custom_formatting = dp.Formatting(
    light_prose=False,
    accent_color="#000", 
    bg_color="#EEE",  # White background
    text_alignment=dp.TextAlignment.LEFT,
    font=dp.FontChoice.SANS,
    width=dp.Width.FULL
  )

# Create Difficulty Report
dp.save_report(report_layout, path='Difficulty_Report.html', formatting=custom_formatting)

# Create CSV Files
difficulty_update_table.to_csv('difficulty_table.csv', index=False)
styled_performance_table.data.to_csv('performance_table.csv', index=False)
styled_fundamentals_table.data.to_csv('fundamentals_table.csv', index=False)
styled_valuation_table.data.to_csv('valuation_table.csv', index=False)
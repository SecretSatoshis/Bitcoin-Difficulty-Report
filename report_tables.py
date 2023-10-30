import datapane as dp
import pandas as pd
import seaborn as sns


# Create Summary Table
def create_difficulty_update_table(report_data, difficulty_report, report_date, difficulty_period_changes):
  # Extract data from report_data for the specific date
  report_date_data = report_data.loc[report_date].name
  bitcoin_supply = report_data.loc[report_date, 'SplyCur']
  HashRate = report_data.loc[report_date, '7_day_ma_HashRate']
  PriceUSD = report_data.loc[report_date, 'PriceUSD']
  Marketcap = report_data.loc[report_date, 'CapMrktCurUSD']
  sats_per_dollar = 100000000 / report_data.loc[report_date, 'PriceUSD']
  difficulty_period_return = difficulty_period_changes.loc['PriceUSD']

  # Extract data from difficulty_report
  block_height = difficulty_report['last_difficulty_change'][0]['block_height']
  difficulty = difficulty_report['last_difficulty_change'][0]['difficulty']
  difficulty_change = difficulty_report['difficulty_change_percentage'][0]

  # Create a dictionary with the extracted values
  difficulty_update_data = {
      "Report Date": report_date_data,
      "Bitcoin Supply": bitcoin_supply,
      "7 Day Average Hashrate": HashRate,
      "Network Difficulty": difficulty,
      "Last Difficulty Adjustment Block Height": block_height,
      "Last Difficulty Change": difficulty_change,
      "Price USD": PriceUSD,
      "Marketcap": Marketcap,
      "Sats Per Dollar": sats_per_dollar,
      "Bitcoin Price Change Difficulty Period": difficulty_period_return
  }

  # Create and return the "Difficulty Update" DataFrame
  difficulty_update_df = pd.DataFrame([difficulty_update_data])
  return difficulty_update_df


#Style Table
def format_value(value, format_type):
  """Helper function to format a value based on the format_type"""
  if format_type == "percentage":
    return f"{value:.2f}%"
  elif format_type == "integer":
    return f"{int(value):,}"
  elif format_type == "float":
    return f"{value:,.0f}"
  elif format_type == "currency":
    return f"${value:,.0f}"
  elif format_type == "date":
    return value.strftime('%Y-%m-%d')
  else:
    return str(value)


# Create Big Numbers
def create_difficulty_big_numbers(difficulty_update_df):
  # Define a dictionary for formatting rules
  format_rules = {
      "Report Date": "date",
      "Bitcoin Supply": "integer",
      "Last Difficulty Adjustment Block Height": "float",
      "Network Difficulty": "float",
      "Last Difficulty Change": "percentage",
      "7 Day Average Hashrate": "float",
      "Price USD": "currency",
      "Marketcap": "currency",
      "Sats Per Dollar": "float",
  }

  # Create a series of BigNumbers for each metric in the table
  big_numbers = []

  for column, value in difficulty_update_df.iloc[0].items():
    if column == "Bitcoin Price Change Difficulty Period":
      continue  # Skip this entry
    formatted_value = format_value(value, format_rules.get(column, ""))
    if column == "Difficulty Change":
      # Assuming positive change is upward, negative is downward
      is_upward = value >= 0
      big_numbers.append(
          dp.BigNumber(
              heading=column,
              value=formatted_value,
              is_upward_change=is_upward,
          ))
    else:
      big_numbers.append(dp.BigNumber(
          heading=column,
          value=formatted_value,
      ))

  # Combine the BigNumbers into a Group with 3 columns and return
  return dp.Group(*big_numbers, columns=3)


# Create Performance Table
def create_performance_table(report_data, difficulty_period_changes,
                             report_date, weekly_high_low, cagr_results,
                             sharpe_results, correlation_results):
  # Define the structure for performance metrics
  performance_metrics_dict = {
      "Bitcoin": {
          "Asset":
          "Bitcoin",
          "Price":
          report_data.loc[report_date, 'PriceUSD'],
          "1 Day Return":
          report_data.loc[report_date, 'PriceUSD_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['PriceUSD'],
          "MTD Return":
          report_data.loc[report_date, 'PriceUSD_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'PriceUSD_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'PriceUSD_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'PriceUSD_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['PriceUSD']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD', 'PriceUSD'],
          "52 Week Low":
          weekly_high_low['PriceUSD']['52_week_low'],
          "52 Week High":
          weekly_high_low['PriceUSD']['52_week_high'],
      },
      "Nasdaq": {
          "Asset":
          "Nasdaq",
          "Price":
          report_data.loc[report_date, '^IXIC_close'],
          "1 Day Return":
          report_data.loc[report_date, '^IXIC_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['^IXIC_close'],
          "MTD Return":
          report_data.loc[report_date, '^IXIC_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, '^IXIC_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, '^IXIC_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, '^IXIC_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['^IXIC_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD',
                                                      '^IXIC_close'],
          "52 Week Low":
          weekly_high_low['^IXIC_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['^IXIC_close']['52_week_high'],
      },
      "S&P500": {
          "Asset":
          "S&P500",
          "Price":
          report_data.loc[report_date, '^GSPC_close'],
          "1 Day Return":
          report_data.loc[report_date, '^GSPC_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['^GSPC_close'],
          "MTD Return":
          report_data.loc[report_date, '^GSPC_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, '^GSPC_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, '^GSPC_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, '^GSPC_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['^GSPC_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD',
                                                      '^GSPC_close'],
          "52 Week Low":
          weekly_high_low['^GSPC_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['^GSPC_close']['52_week_high'],
      },
      "XLF": {
          "Asset":
          "XLF Financials ETF",
          "Price":
          report_data.loc[report_date, 'XLF_close'],
          "1 Day Return":
          report_data.loc[report_date, 'XLF_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['XLF_close'],
          "MTD Return":
          report_data.loc[report_date, 'XLF_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'XLF_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'XLF_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'XLF_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['XLF_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD', 'XLF_close'],
          "52 Week Low":
          weekly_high_low['XLF_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['XLF_close']['52_week_high'],
      },
      "XLE": {
          "Asset":
          "XLE Energy ETF",
          "Price":
          report_data.loc[report_date, 'XLE_close'],
          "1 Day Return":
          report_data.loc[report_date, 'XLE_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['XLE_close'],
          "MTD Return":
          report_data.loc[report_date, 'XLE_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'XLE_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'XLE_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'XLE_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['XLE_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD', 'XLE_close'],
          "52 Week Low":
          weekly_high_low['XLE_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['XLE_close']['52_week_high'],
      },
      "FANG+": {
          "Asset":
          "FANG+ ETF",
          "Price":
          report_data.loc[report_date, 'FANG.AX_close'],
          "1 Day Return":
          report_data.loc[report_date, 'FANG.AX_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['FANG.AX_close'],
          "MTD Return":
          report_data.loc[report_date, 'FANG.AX_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'FANG.AX_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'FANG.AX_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'FANG.AX_close_2_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['FANG.AX_close']['2_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD',
                                                      'FANG.AX_close'],
          "52 Week Low":
          weekly_high_low['FANG.AX_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['FANG.AX_close']['52_week_high'],
      },
      "BITQ": {
          "Asset":
          "BITQ Crypto Industry ETF",
          "Price":
          report_data.loc[report_date, 'BITQ_close'],
          "1 Day Return":
          report_data.loc[report_date, 'BITQ_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['BITQ_close'],
          "MTD Return":
          report_data.loc[report_date, 'BITQ_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'BITQ_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'BITQ_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'BITQ_close_2_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['BITQ_close']['2_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD',
                                                      'BITQ_close'],
          "52 Week Low":
          weekly_high_low['BITQ_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['BITQ_close']['52_week_high'],
      },
      "Gold Futures": {
          "Asset":
          "Gold",
          "Price":
          report_data.loc[report_date, 'GC=F_close'],
          "1 Day Return":
          report_data.loc[report_date, 'GC=F_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['GC=F_close'],
          "MTD Return":
          report_data.loc[report_date, 'GC=F_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'GC=F_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'GC=F_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'GC=F_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['GC=F_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD',
                                                      'GC=F_close'],
          "52 Week Low":
          weekly_high_low['GC=F_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['GC=F_close']['52_week_high'],
      },
      "US Dollar Futures": {
          "Asset":
          "US Dollar Index",
          "Price":
          report_data.loc[report_date, 'DX=F_close'],
          "1 Day Return":
          report_data.loc[report_date, 'DX=F_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['DX=F_close'],
          "MTD Return":
          report_data.loc[report_date, 'DX=F_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'DX=F_close_90_change'],
          "YTD Return":
          report_data.loc[report_date, 'DX=F_close_YTD_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'DX=F_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['DX=F_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD',
                                                      'DX=F_close'],
          "52 Week Low":
          weekly_high_low['DX=F_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['DX=F_close']['52_week_high'],
      },
      "TLT": {
          "Asset":
          "TLT Treasury Bond ETF",
          "Price":
          report_data.loc[report_date, 'TLT_close'],
          "1 Day Return":
          report_data.loc[report_date, 'TLT_close_1_change'],
          "Difficulty Period Return":
          difficulty_period_changes.loc['TLT_close'],
          "MTD Return":
          report_data.loc[report_date, 'TLT_close_MTD_change'],
          "90 Day Return":
          report_data.loc[report_date, 'TLT_close_90_change'],
          "4 Year CAGR":
          cagr_results.loc[report_date, 'TLT_close_4_Year_CAGR'],
          "4 Year Sharpe":
          sharpe_results['TLT_close']['4_year'].loc[report_date],
          "90 Day BTC Correlation":
          correlation_results['priceusd_90_days'].loc['PriceUSD', 'TLT_close'],
          "YTD Return":
          report_data.loc[report_date, 'TLT_close_YTD_change'],
          "52 Week Low":
          weekly_high_low['TLT_close']['52_week_low'],
          "52 Week High":
          weekly_high_low['TLT_close']['52_week_high'],
      }
  }

  # Convert the dictionary to a DataFrame
  performance_table_df = pd.DataFrame(list(performance_metrics_dict.values()))

  return performance_table_df


# Style Table
def style_performance_table(performance_table):
  import seaborn as sns

  format_dict = {
      'Asset': '{}',
      'Price': '{:,.2f}',
      "1 Day Return": '{:.2%}',
      'Difficulty Period Return': '{:.2f}%',
      'MTD Return': '{:.2%}',
      '90 Day Return': '{:.2%}',
      'YTD Return': '{:.2%}',
      '4 Year CAGR': '{:.2%}',
      '4 Year Sharpe': '{:,.2f}',
      '90 Day BTC Correlation': '{:,.2f}',
      '52 Week Low': '{:,.2f}',
      '52 Week High': '{:,.2f}'
  }

  diverging_cm = sns.diverging_palette(100, 133, as_cmap=True)
  diverging_cm = sns.diverging_palette(0, 0, s=0, l=85, as_cmap=True)
  bg_colormap = sns.light_palette("white", as_cmap=True)

  def color_values(val):
    color = 'green' if val > 0 else ('red' if val < 0 else 'black')
    return 'color: %s' % color

  gradient_columns = [
      '1 Day Return', 'Difficulty Period Return', 'MTD Return',
      '90 Day Return', 'YTD Return', '4 Year CAGR', '4 Year Sharpe',
      '90 Day BTC Correlation'
  ]

  styled_table = (performance_table.style.format(format_dict).applymap(
      color_values, subset=gradient_columns).hide_index().set_properties(**{'white-space': 'nowrap'}))

  return styled_table
  

# Create Fundamentals Table
def create_bitcoin_fundamentals_table(report_data, difficulty_period_changes,
                                      weekly_high_low, report_date, cagr_results):
  # Extract data from report_data for the specific date
  HashRate = report_data.loc[report_date, '7_day_ma_HashRate']
  TxCnt = report_data.loc[report_date, '7_day_ma_TxCnt']
  TxTfrValAdjUSD = report_data.loc[report_date, '7_day_ma_TxTfrValAdjUSD']
  TxTfrValMeanUSD = report_data.loc[report_date, '7_day_ma_TxTfrValMeanUSD']
  RevUSD = report_data.loc[report_date, 'RevUSD']
  AdrActCnt = report_data.loc[report_date, 'AdrActCnt']
  AdrBalUSD10Cnt = report_data.loc[report_date, 'AdrBalUSD10Cnt']
  FeeTotUSD = report_data.loc[report_date, 'FeeTotUSD']
  supply_pct_1_year_plus = report_data.loc[report_date,
                                           'supply_pct_1_year_plus']
  VelCur1yr = report_data.loc[report_date, 'VelCur1yr']

  HashRate_MTD = report_data.loc[report_date, 'HashRate_MTD_change']
  TxCnt_MTD = report_data.loc[report_date, 'TxCnt_MTD_change']
  TxTfrValAdjUSD_MTD = report_data.loc[report_date,
                                       'TxTfrValAdjUSD_MTD_change']
  TxTfrValMeanUSD_MTD = report_data.loc[report_date,
                                        '7_day_ma_TxTfrValMeanUSD_MTD_change']
  RevUSD_MTD = report_data.loc[report_date, 'RevUSD_MTD_change']
  AdrActCnt_MTD = report_data.loc[report_date, 'AdrActCnt_MTD_change']
  AdrBalUSD10Cnt_MTD = report_data.loc[report_date,
                                       'AdrBalUSD10Cnt_MTD_change']
  FeeTotUSD_MTD = report_data.loc[report_date, 'FeeTotUSD_MTD_change']
  supply_pct_1_year_plus_MTD = report_data.loc[
      report_date, 'supply_pct_1_year_plus_MTD_change']
  VelCur1yr_MTD = report_data.loc[report_date, 'VelCur1yr_MTD_change']

  HashRate_YTD = report_data.loc[report_date, 'HashRate_YTD_change']
  TxCnt_YTD = report_data.loc[report_date, 'TxCnt_YTD_change']
  TxTfrValAdjUSD_YTD = report_data.loc[report_date,
                                       'TxTfrValAdjUSD_YTD_change']
  TxTfrValMeanUSD_YTD = report_data.loc[report_date,
                                        '7_day_ma_TxTfrValMeanUSD_YTD_change']
  RevUSD_YTD = report_data.loc[report_date, 'RevUSD_YTD_change']
  AdrActCnt_YTD = report_data.loc[report_date, 'AdrActCnt_YTD_change']
  AdrBalUSD10Cnt_YTD = report_data.loc[report_date,
                                       'AdrBalUSD10Cnt_YTD_change']
  FeeTotUSD_YTD = report_data.loc[report_date, 'FeeTotUSD_YTD_change']
  supply_pct_1_year_plus_YTD = report_data.loc[
      report_date, 'supply_pct_1_year_plus_YTD_change']
  VelCur1yr_YTD = report_data.loc[report_date, 'VelCur1yr_YTD_change']

  HashRate_90 = report_data.loc[report_date, 'HashRate_90_change']
  TxCnt_90 = report_data.loc[report_date, 'TxCnt_90_change']
  TxTfrValAdjUSD_90 = report_data.loc[report_date, 'TxTfrValAdjUSD_90_change']
  TxTfrValMeanUSD_90 = report_data.loc[report_date,
                                       '7_day_ma_TxTfrValMeanUSD_90_change']
  RevUSD_90 = report_data.loc[report_date, 'RevUSD_90_change']
  AdrActCnt_90 = report_data.loc[report_date, 'AdrActCnt_90_change']
  AdrBalUSD10Cnt_90 = report_data.loc[report_date, 'AdrBalUSD10Cnt_90_change']
  FeeTotUSD_90 = report_data.loc[report_date, 'FeeTotUSD_90_change']
  supply_pct_1_year_plus_90 = report_data.loc[
      report_date, 'supply_pct_1_year_plus_90_change']
  VelCur1yr_90 = report_data.loc[report_date, 'VelCur1yr_90_change']

  HashRate_1 = report_data.loc[report_date, 'HashRate_1_change']
  TxCnt_1 = report_data.loc[report_date, 'TxCnt_1_change']
  TxTfrValAdjUSD_1 = report_data.loc[report_date, 'TxTfrValAdjUSD_1_change']
  TxTfrValMeanUSD_1 = report_data.loc[report_date,
                                      '7_day_ma_TxTfrValMeanUSD_1_change']
  RevUSD_1 = report_data.loc[report_date, 'RevUSD_1_change']
  AdrActCnt_1 = report_data.loc[report_date, 'AdrActCnt_1_change']
  AdrBalUSD10Cnt_1 = report_data.loc[report_date, 'AdrBalUSD10Cnt_1_change']
  FeeTotUSD_1 = report_data.loc[report_date, 'FeeTotUSD_1_change']
  supply_pct_1_year_plus_1 = report_data.loc[report_date,
                                             'supply_pct_1_year_plus_1_change']
  VelCur1yr_1 = report_data.loc[report_date, 'VelCur1yr_1_change']

  HashRate_CAGR = cagr_results.loc[report_date, 'HashRate_4_Year_CAGR']
  TxCnt_CAGR = cagr_results.loc[report_date, 'TxCnt_4_Year_CAGR']
  TxTfrValAdjUSD_CAGR = cagr_results.loc[report_date,
                                         'TxTfrValAdjUSD_4_Year_CAGR']
  TxTfrValMeanUSD_CAGR = cagr_results.loc[report_date,
                                          'TxTfrValMeanUSD_4_Year_CAGR']
  RevUSD_CAGR = cagr_results.loc[report_date, 'RevUSD_4_Year_CAGR']
  AdrActCnt_CAGR = cagr_results.loc[report_date, 'AdrActCnt_4_Year_CAGR']
  AdrBalUSD10Cnt_CAGR = cagr_results.loc[report_date,
                                         'AdrBalUSD10Cnt_4_Year_CAGR']
  FeeTotUSD_CAGR = cagr_results.loc[report_date, 'FeeTotUSD_4_Year_CAGR']
  supply_pct_1_year_plus_CAGR = cagr_results.loc[
      report_date, 'supply_pct_1_year_plus_4_Year_CAGR']
  VelCur1yr_CAGR = cagr_results.loc[report_date, 'VelCur1yr_4_Year_CAGR']

  # Fetch 52-week high and low for each metric
  HashRate_52_high = weekly_high_low['7_day_ma_HashRate']['52_week_high']
  TxCnt_52_high = weekly_high_low['7_day_ma_TxCnt']['52_week_high']
  TxTfrValAdjUSD_52_high = weekly_high_low['7_day_ma_TxTfrValAdjUSD'][
      '52_week_high']
  TxTfrValMeanUSD_52_high = weekly_high_low['7_day_ma_TxTfrValMeanUSD'][
      '52_week_high']
  RevUSD_52_high = weekly_high_low['RevUSD']['52_week_high']
  AdrActCnt_52_high = weekly_high_low['AdrActCnt']['52_week_high']
  AdrBalUSD10Cnt_52_high = weekly_high_low['AdrBalUSD10Cnt']['52_week_high']
  FeeTotUSD_52_high = weekly_high_low['FeeTotUSD']['52_week_high']
  supply_pct_1_year_plus_52_high = weekly_high_low['supply_pct_1_year_plus'][
      '52_week_high']
  VelCur1yr_52_high = weekly_high_low['VelCur1yr']['52_week_high']

  HashRate_52_low = weekly_high_low['7_day_ma_HashRate']['52_week_low']
  TxCnt_52_low = weekly_high_low['7_day_ma_TxCnt']['52_week_low']
  TxTfrValAdjUSD_52_low = weekly_high_low['7_day_ma_TxTfrValAdjUSD'][
      '52_week_low']
  TxTfrValMeanUSD_52_low = weekly_high_low['7_day_ma_TxTfrValMeanUSD'][
      '52_week_low']
  RevUSD_52_low = weekly_high_low['RevUSD']['52_week_low']
  AdrActCnt_52_low = weekly_high_low['AdrActCnt']['52_week_low']
  AdrBalUSD10Cnt_52_low = weekly_high_low['AdrBalUSD10Cnt']['52_week_low']
  FeeTotUSD_52_low = weekly_high_low['FeeTotUSD']['52_week_low']
  supply_pct_1_year_plus_52_low = weekly_high_low['supply_pct_1_year_plus'][
      '52_week_low']
  VelCur1yr_52_low = weekly_high_low['VelCur1yr']['52_week_low']

  HashRate_difficulty_change = difficulty_period_changes.loc['7_day_ma_HashRate']
  TxCnt_difficulty_change = difficulty_period_changes.loc['TxCnt']
  TxTfrValAdjUSD_difficulty_change = difficulty_period_changes.loc[
      'TxTfrValAdjUSD']
  TxTfrValMeanUSD_difficulty_change = difficulty_period_changes.loc[
      '7_day_ma_TxTfrValMeanUSD']
  RevUSD_difficulty_change = difficulty_period_changes.loc['RevUSD']
  AdrActCnt_difficulty_change = difficulty_period_changes.loc['AdrActCnt']
  AdrBalUSD10Cnt_difficulty_change = difficulty_period_changes.loc[
      'AdrBalUSD10Cnt']
  FeeTotUSD_difficulty_change = difficulty_period_changes.loc['FeeTotUSD']
  supply_pct_1_year_plus_difficulty_change = difficulty_period_changes.loc[
      'supply_pct_1_year_plus']
  VelCur1yr_difficulty_change = difficulty_period_changes.loc['VelCur1yr']

  # Create a dictionary with the extracted values
  bitcoin_fundamentals_data = {
      "Metrics Name": [
          'Hashrate', 'Transaction Count',
          'Transaction Volume', 'Avg Transaction Size',
          'Active Address Count', '+$10 USD Address', 'Miner Revenue',
          'Fees In USD', '1+ Year Supply %', '1 Year Velocity'
      ],
      "Value": [
          HashRate, TxCnt, TxTfrValAdjUSD, TxTfrValMeanUSD, AdrActCnt,
          AdrBalUSD10Cnt, RevUSD, FeeTotUSD, supply_pct_1_year_plus, VelCur1yr
      ],
      "1 Day Change": [
          HashRate_1, TxCnt_1, TxTfrValAdjUSD_1, TxTfrValMeanUSD_1,
          AdrActCnt_1, AdrBalUSD10Cnt_1, RevUSD_1, FeeTotUSD_1,
          supply_pct_1_year_plus_1, VelCur1yr_1
      ],
      "Difficulty Period Change": [
          HashRate_difficulty_change, TxCnt_difficulty_change,
          TxTfrValAdjUSD_difficulty_change, TxTfrValMeanUSD_difficulty_change,
          AdrActCnt_difficulty_change, AdrBalUSD10Cnt_difficulty_change,
          RevUSD_difficulty_change, FeeTotUSD_difficulty_change,
          supply_pct_1_year_plus_difficulty_change, VelCur1yr_difficulty_change
      ],
      "MTD Change": [
          HashRate_MTD, TxCnt_MTD, TxTfrValAdjUSD_MTD, TxTfrValMeanUSD_MTD,
          AdrActCnt_MTD, AdrBalUSD10Cnt_MTD, RevUSD_MTD, FeeTotUSD_MTD,
          supply_pct_1_year_plus_MTD, VelCur1yr_MTD
      ],
      "90 Day Change": [
          HashRate_90, TxCnt_90, TxTfrValAdjUSD_90, TxTfrValMeanUSD_90,
          AdrActCnt_90, AdrBalUSD10Cnt_90, RevUSD_90, FeeTotUSD_90,
          supply_pct_1_year_plus_90, VelCur1yr_90
      ],
      "YTD Change": [
          HashRate_YTD, TxCnt_YTD, TxTfrValAdjUSD_YTD, TxTfrValMeanUSD_YTD,
          AdrActCnt_YTD, AdrBalUSD10Cnt_YTD, RevUSD_YTD, FeeTotUSD_YTD,
          supply_pct_1_year_plus_YTD, VelCur1yr_YTD
      ],
      "4 Year CAGR": [
          HashRate_CAGR, TxCnt_CAGR, TxTfrValAdjUSD_CAGR, TxTfrValMeanUSD_CAGR,
          AdrActCnt_CAGR, AdrBalUSD10Cnt_CAGR, RevUSD_CAGR, FeeTotUSD_CAGR,
          supply_pct_1_year_plus_CAGR, VelCur1yr_CAGR
      ],
      "52 Week Low": [
          HashRate_52_low, TxCnt_52_low, TxTfrValAdjUSD_52_low,
          TxTfrValMeanUSD_52_low, AdrActCnt_52_low, AdrBalUSD10Cnt_52_low,
          RevUSD_52_low, FeeTotUSD_52_low, supply_pct_1_year_plus_52_low,
          VelCur1yr_52_low
      ],
      "52 Week High": [
          HashRate_52_high, TxCnt_52_high, TxTfrValAdjUSD_52_high,
          TxTfrValMeanUSD_52_high, AdrActCnt_52_high, AdrBalUSD10Cnt_52_high,
          RevUSD_52_high, FeeTotUSD_52_high, supply_pct_1_year_plus_52_high,
          VelCur1yr_52_high
      ],
  }

  # Create and return the "Bitcoin Fundamentals" DataFrame
  bitcoin_fundamentals_df = pd.DataFrame(bitcoin_fundamentals_data)

  return bitcoin_fundamentals_df


# Style Table
def style_bitcoin_fundamentals_table(fundamentals_table):
  format_rules = {
      'Hashrate': '{:,.0f}',
      'Transaction Count': '{:,.0f}',
      'Transaction Volume': '${:,.0f}',
      'Avg Transaction Size': '${:,.0f}',
      'Miner Revenue': '${:,.0f}',
      'Active Address Count': '{:,.0f}',
      '+$10 USD Address': '{:,.0f}',
      'Fees In USD': '${:,.0f}',
      '1+ Year Supply %': '{:.2f}%',
      '1 Year Velocity': '{:.2f}'
  }

  def custom_formatter(row, column_name):
    """Apply custom formatting based on the metric name for a specified column."""
    metric = row['Metrics Name']
    format_string = format_rules.get(metric, '{}')  # Default to '{}'
    return format_string.format(row[column_name])

  # Use the 'apply' function to format the 'Value', '52 Week Low', and '52 Week High' columns
  fundamentals_table['Value'] = fundamentals_table.apply(
      lambda row: custom_formatter(row, 'Value'), axis=1)
  fundamentals_table['52 Week Low'] = fundamentals_table.apply(
      lambda row: custom_formatter(row, '52 Week Low'), axis=1)
  fundamentals_table['52 Week High'] = fundamentals_table.apply(
      lambda row: custom_formatter(row, '52 Week High'), axis=1)

  format_dict_fundamentals = {
      'Metrics Name': '{}',
      "1 Day Change": '{:.2%}',
      'Difficulty Period Change': '{:.2f}%',
      'MTD Change': '{:.2%}',
      '90 Day Change': '{:.2%}',
      'YTD Change': '{:.2%}',
      '4 Year CAGR': '{:.2%}',
  }

  # Define a custom colormap that diverges from red to green
  diverging_cm = sns.diverging_palette(100, 133, as_cmap=True)
  diverging_cm = sns.diverging_palette(0, 0, s=0, l=85, as_cmap=True)
  bg_colormap = sns.light_palette("white", as_cmap=True)

  def color_values(val):
    """
        Takes a scalar and returns a string with
        the CSS property `'color: green'` for positive
        values, and `'color: red'` for negative values.
        """
    color = 'green' if val > 0 else ('red' if val < 0 else 'black')
    return 'color: %s' % color

  # Columns to apply the background gradient on
  gradient_columns = [
      '1 Day Change', 'Difficulty Period Change', 'MTD Change',
      '90 Day Change', 'YTD Change', '4 Year CAGR'
  ]

  # Apply the formatting and the background gradient only to the specified columns
  styled_table_colors = (
      fundamentals_table.style.format(format_dict_fundamentals).applymap(
          color_values, subset=gradient_columns).hide_index().set_properties(**{'white-space': 'nowrap'}))

  return styled_table_colors


# Create Valuation Table
def create_bitcoin_valuation_table(report_data, difficulty_period_changes,
                                   weekly_high_low, valuation_data, report_date):

  # Extract BTC Value
  btc_value = report_data.loc[report_date, 'PriceUSD']

  # Extraction for "NVTAdj"
  nvt_price_multiple = report_data.loc[report_date, 'nvt_price']
  nvt_difficulty_change = difficulty_period_changes.loc['nvt_price']
  nvt_buy_target = valuation_data['nvt_price_multiple_buy_target']
  nvt_sell_target = valuation_data['nvt_price_multiple_sell_target']
  nvt_pct_from_fair_value = (nvt_price_multiple - btc_value) / btc_value
  nvt_return_to_target = (nvt_sell_target - btc_value) / btc_value
  nvt_return_to_buy_target = (nvt_buy_target - btc_value) / btc_value

  # Extraction for "200_day_multiple"
  day_200_price = report_data.loc[report_date, '200_day_ma_priceUSD']
  day_200_difficulty_change = difficulty_period_changes.loc['200_day_multiple']
  day_200_buy_target = valuation_data['200_day_multiple_buy_target']
  day_200_sell_target = valuation_data['200_day_multiple_sell_target']
  day_200_pct_from_fair_value = (day_200_price - btc_value) / btc_value
  day_200_return_to_target = (day_200_sell_target - btc_value) / btc_value
  day_200_return_to_buy_target = (day_200_buy_target - btc_value) / btc_value

  # Extraction for "mvrv_ratio"
  mvrv_price = report_data.loc[report_date, 'realised_price']
  mvrv_difficulty_change = difficulty_period_changes.loc['realised_price']
  mvrv_buy_target = valuation_data['mvrv_ratio_buy_target']
  mvrv_sell_target = valuation_data['mvrv_ratio_sell_target']
  mvrv_pct_from_fair_value = (mvrv_price - btc_value) / btc_value
  mvrv_return_to_target = (mvrv_sell_target - btc_value) / btc_value
  mvrv_return_to_buy_target = (mvrv_buy_target - btc_value) / btc_value
  
  # Extraction for "thermocap_multiple"
  thermo_price = report_data.loc[report_date, 'thermocap_multiple_8']
  thermo_difficulty_change = difficulty_period_changes.loc[
      'thermocap_multiple_8']
  thermo_buy_target = valuation_data['thermocap_multiple_buy_target']
  thermo_sell_target = valuation_data['thermocap_multiple_sell_target']
  thermo_pct_from_fair_value = (thermo_price - btc_value) / btc_value
  thermo_return_to_target = (thermo_sell_target - btc_value) / btc_value
  thermo_return_to_buy_target = (thermo_buy_target - btc_value) / btc_value

  # Extraction for "stocktoflow"
  sf_price = report_data.loc[report_date, 'SF_Predicted_Price']
  sf_difficulty_change = difficulty_period_changes.loc['SF_Predicted_Price']
  sf_buy_target = valuation_data['SF_Multiple_buy_target']
  sf_sell_target = valuation_data['SF_Multiple_sell_target']
  sf_pct_from_fair_value = (sf_price - btc_value) / btc_value
  sf_return_to_target = (sf_sell_target - btc_value) / btc_value
  sf_return_to_buy_target = (sf_buy_target - btc_value) / btc_value

  # Extraction for "appl_marketcap"
  aapl_price = valuation_data['AAPL_MarketCap_bull_present_value']
  aapl_difficulty_change = difficulty_period_changes.loc['AAPL_mc_btc_price']
  aapl_buy_target = valuation_data['AAPL_MarketCap_base_present_value']
  aapl_sell_target = report_data.loc[report_date, 'AAPL_mc_btc_price']
  aapl_pct_from_fair_value = (aapl_price - btc_value) / btc_value
  aapl_return_to_target = (aapl_sell_target - btc_value) / btc_value
  aapl_return_to_buy_target = (aapl_buy_target - btc_value) / btc_value

  # Extraction for "gold_marketcap_billion_usd"
  gold_price = valuation_data['gold_marketcap_billion_usd_bull_present_value']
  gold_difficulty_change = difficulty_period_changes.loc[
      'gold_marketcap_billion_usd']
  gold_buy_target = valuation_data['gold_marketcap_billion_usd_base_present_value']
  gold_sell_target = report_data.loc[report_date,
                               'gold_marketcap_billion_usd'] / report_data.loc[
                                   report_date, 'SplyExpFut10yr']
  gold_pct_from_fair_value = (gold_price - btc_value) / btc_value
  gold_return_to_target = (gold_sell_target - btc_value) / btc_value
  gold_return_to_buy_target = (gold_buy_target - btc_value) / btc_value

  # Extraction for "silver_marketcap_billion_usd"
  silver_price = valuation_data['silver_marketcap_billion_usd_bull_present_value']
  silver_difficulty_change = difficulty_period_changes.loc[
      'silver_marketcap_billion_usd']
  silver_buy_target = valuation_data['silver_marketcap_billion_usd_base_present_value']
  silver_sell_target = report_data.loc[
      report_date,
      'silver_marketcap_billion_usd'] / report_data.loc[report_date,
                                                        'SplyExpFut10yr']
  silver_pct_from_fair_value = (silver_price - btc_value) / btc_value
  silver_return_to_target = (silver_sell_target - btc_value) / btc_value
  silver_return_to_buy_target = (silver_buy_target - btc_value) / btc_value

  # Extraction for "United_States_btc_price"
  us_btc_price = valuation_data['United_States_cap_bull_present_value']
  us_difficulty_change = difficulty_period_changes.loc[
      'United_States_btc_price']
  us_buy_target = valuation_data['United_States_cap_base_present_value']
  us_sell_target = report_data.loc[report_date, 'United_States_btc_price']
  us_pct_from_fair_value = (us_btc_price - btc_value) / btc_value
  us_return_to_target = (us_sell_target - btc_value) / btc_value
  us_return_to_buy_target = (us_buy_target - btc_value) / btc_value

  # Extraction for "United_Kingdom_btc_price"
  uk_btc_price = valuation_data['United_Kingdom_cap_bull_present_value']
  uk_difficulty_change = difficulty_period_changes.loc[
      'United_Kingdom_btc_price']
  uk_buy_target = valuation_data['United_Kingdom_cap_base_present_value']
  uk_sell_target = report_data.loc[report_date, 'United_Kingdom_btc_price']
  uk_pct_from_fair_value = (uk_btc_price - btc_value) / btc_value
  uk_return_to_target = (uk_sell_target - btc_value) / btc_value
  uk_return_to_buy_target = (uk_buy_target - btc_value) / btc_value

  # Update the dictionary with the extracted values
  bitcoin_valuation_data = {
      "Valuation Model": [
          '200 Day Moving Average', 'NVT Price', 'Realized Price',
          'ThermoCap Price', 'Stock To Flow Price', 'Silver Market Cap',
          'UK M0 Price', 'Apple Market Cap', 'US M0 Price', 'Gold Market Cap'
      ],
      "Model Price": [
          day_200_price, nvt_price_multiple, mvrv_price, thermo_price,
          sf_price, silver_price, uk_btc_price, aapl_price, us_btc_price,
          gold_price
      ],
      "Difficulty Period Change": [
          day_200_difficulty_change, nvt_difficulty_change,
          mvrv_difficulty_change, thermo_difficulty_change,
          sf_difficulty_change, silver_difficulty_change, uk_difficulty_change,
          aapl_difficulty_change, us_difficulty_change, gold_difficulty_change
      ],
      "BTC Price": [
          btc_value, btc_value, btc_value, btc_value, btc_value, btc_value,
          btc_value, btc_value, btc_value, btc_value
      ],
      "Buy Target": [
          day_200_buy_target, nvt_buy_target, mvrv_buy_target,
          thermo_buy_target, sf_buy_target, silver_buy_target, uk_buy_target,
          aapl_buy_target, us_buy_target, gold_buy_target
      ],
      "Sell Target": [
          day_200_sell_target, nvt_sell_target, mvrv_sell_target,
          thermo_sell_target, sf_sell_target, silver_sell_target,
          uk_sell_target, aapl_sell_target, us_sell_target, gold_sell_target
      ],
      "% To Buy Target": [
          day_200_return_to_buy_target, nvt_return_to_buy_target,
          mvrv_return_to_buy_target, thermo_return_to_buy_target, sf_return_to_buy_target,
          silver_return_to_buy_target, uk_return_to_buy_target, aapl_return_to_buy_target,
          us_return_to_buy_target, gold_return_to_buy_target
      ],
      "% To Model Price": [
          day_200_pct_from_fair_value, nvt_pct_from_fair_value,
          mvrv_pct_from_fair_value, thermo_pct_from_fair_value,
          sf_pct_from_fair_value, silver_pct_from_fair_value,
          uk_pct_from_fair_value, aapl_pct_from_fair_value,
          us_pct_from_fair_value, gold_pct_from_fair_value
      ],
      "% To Sell Target": [
          day_200_return_to_target, nvt_return_to_target,
          mvrv_return_to_target, thermo_return_to_target, sf_return_to_target,
          silver_return_to_target, uk_return_to_target, aapl_return_to_target,
          us_return_to_target, gold_return_to_target
      ]
  }

  # Create and return the "Bitcoin Valuation" DataFrame
  bitcoin_valuation_df = pd.DataFrame(bitcoin_valuation_data)

  return bitcoin_valuation_df


# Style Table
def style_bitcoin_valuation_table(bitcoin_valuation_table):
  format_dict_valuation = {
      'Valuation Model': '{}',
      'Model Price': '${:,.0f}',
      'Difficulty Period Change': '{:.2f}%',
      'BTC Price': '${:,.0f}',
      'Buy Target': '${:,.0f}',
      'Sell Target': '${:,.0f}',
      '% To Buy Target': '{:.2%}',
      '% To Model Price': '{:.2%}',
      '% To Sell Target': '{:.2%}'
  }

  # Define a custom colormap that diverges from red to green
  diverging_cm = sns.diverging_palette(100, 133, as_cmap=True)
  diverging_cm = sns.diverging_palette(0, 0, s=0, l=85, as_cmap=True)
  bg_colormap = sns.light_palette("white", as_cmap=True)

  def color_values(val):
    """
        Takes a scalar and returns a string with
        the CSS property `'color: green'` for positive
        values, and `'color: red'` for negative values.
        """
    color = 'green' if val > 0 else ('red' if val < 0 else 'black')
    return 'color: %s' % color

  # Columns to apply the background gradient on
  gradient_columns = [
      'Difficulty Period Change', '% To Model Price', '% To Sell Target', '% To Buy Target'
  ]

  # Apply the formatting and the background gradient only to the specified columns
  styled_table_colors = (
      bitcoin_valuation_table.style.format(format_dict_valuation).applymap(
          color_values, subset=gradient_columns).hide_index().set_properties(**{'white-space': 'nowrap'})  # Prevent content wrapping
                    .set_table_styles([
                         {'selector': 'th', 'props': [('white-space', 'nowrap')]}
                     ]))

  return styled_table_colors

import datetime
import pandas as pd

#TradFi Data
tickers = {
    'stocks': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-A',
        'BRK-B', 'TSM', 'V', 'JPM', 'PYPL', 'GS'
    ],
    'etfs': [
        'BITQ', 'CLOU', 'ARKK', 'XLK', 'QQQ', 'IUIT.L', 'VTI', 'TLT', 'LQD',
        'JNK', 'GLD', 'XLF', 'XLRE', 'SHY', 'XLE', 'FANG.AX'
    ],
    'indices': ['^GSPC', '^VIX', '^IXIC', '^TNX', '^TYX', '^FVX', '^IRX','^BCOM'],
    'commodities': ['GC=F', 'CL=F', 'SI=F'],
    'forex': [
        'DX=F', 'AUDUSD=X', 'CHFUSD=X', 'CNYUSD=X', 'EURUSD=X', 'GBPUSD=X',
        'HKDUSD=X', 'INRUSD=X', 'JPYUSD=X', 'RUBUSD=X'
    ]
}
# Start date for TradFi Data
market_data_start_date = '2010-01-01'

# Fiat Money Supply M0 Data
fiat_money_data_top10 = pd.DataFrame({
    'Country': [
        'United States', 'China', 'Eurozone', 'Japan', 'United Kingdom',
        'Switzerland', 'India', 'Australia', 'Russia', 'Hong Kong',
        'Global Fiat Supply'
    ],
    'US Dollar Trillion':
    [5.41, 4.81, 6.16, 4.50, 1.16, 0.71, 0.50, 0.36, 0.35, 0.25, 26.77]
})

# Gold And Silver Supply Data
gold_silver_supply = pd.DataFrame({
    "Metal": ["Gold", "Silver"],
    "Supply in Billion Troy Ounces": [5630000000, 28400000000]
})

# Gold Market Supply Breakdown
gold_supply_breakdown = pd.DataFrame({
    "Gold Supply Breakdown":
    ["Jewellery", "Private Investment", "Official Country Holdings", "Other"],
    "Percentage Of Market": [47.00, 22.00, 17.00, 14.00]
})

# Just stock tickers for marketcap calculation
stock_tickers = tickers['stocks']

# Get today's date
today = datetime.date.today()

# Get yesterday's date
yesterday = today - datetime.timedelta(days=1)

# Creat report data and convert to pandas.Timestamp
report_date = pd.Timestamp(yesterday)

# On-Chain Metrics to create moving averages of for data smoothing
moving_avg_metrics = [
    'HashRate',
    'AdrActCnt',
    'TxCnt',
    'TxTfrValAdjUSD',
    'TxTfrValMeanUSD',
    'FeeMeanUSD',
    'FeeMeanNtv',
    'IssContNtv',
    'RevUSD',
    'nvt_price',
    'nvt_price_adj',
]

# Define report data and fields
filter_data_columns = {
    'Report_Metrics': [
        'SplyCur', 'SplyExpFut10yr', '7_day_ma_IssContNtv',
        '365_day_ma_IssContNtv', 'TxCnt', '7_day_ma_TxCnt', '365_day_ma_TxCnt',
        'HashRate', '7_day_ma_HashRate', '365_day_ma_HashRate', 'PriceUSD',
        '50_day_ma_priceUSD', '200_day_ma_priceUSD', '200_day_multiple',
        '200_week_ma_priceUSD', 'TxTfrValAdjUSD', '7_day_ma_TxTfrValAdjUSD',
        '365_day_ma_TxTfrValAdjUSD', 'RevUSD', 'AdrActCnt', 'AdrBalUSD10Cnt',
        '30_day_ma_AdrActCnt', '365_day_ma_AdrActCnt',
        '7_day_ma_TxTfrValMeanUSD', 'FeeTotUSD', 'thermocap_multiple_4',
        'thermocap_multiple_8', 'thermocap_multiple_16',
        'thermocap_multiple_32', 'thermocap_multiple', 'nvt_price',
        'nvt_price_adj', 'nvt_price_multiple', '30_day_ma_nvt_price',
        '365_day_ma_nvt_price', 'NVTAdj', 'NVTAdj90', 'NVTAdjFF',
        'realised_price', 'VelCur1yr', 'supply_pct_1_year_plus', 'mvrv_ratio',
        'realizedcap_multiple_3', 'realizedcap_multiple_5', 'nupl',
        'AAPL_close', 'CapMrktCurUSD', '^IXIC_close', "RevHashRateUSD",
        '^GSPC_close', 'XLF_close', 'XLRE_close', 'GC=F_close', 'SI=F_close',
        'DX=F_close', 'SHY_close', 'TLT_close', '^TNX_close', '^TYX_close',
        '^FVX_close', '^IRX_close', 'XLE_close', 'BITQ_close', 'FANG.AX_close','^BCOM_close',
        'AAPL_mc_btc_price','AAPL_MarketCap', 'AUDUSD=X_close', 'CHFUSD=X_close',
        'CNYUSD=X_close', 'EURUSD=X_close', 'GBPUSD=X_close', 'HKDUSD=X_close',
        'INRUSD=X_close', 'JPYUSD=X_close', 'RUBUSD=X_close',
        'silver_marketcap_billion_usd', 'gold_marketcap_billion_usd',
        'United_Kingdom_btc_price','United_Kingdom_cap','United_States_btc_price','United_States_cap',
        'Global_Fiat_Supply_btc_price', 'SF_Predicted_Price', 'SF_Multiple'
    ]
}

# First Halving Date Start Stats Calculation
stats_start_date = '2012-11-28'

# Timeframes to calculate volatitlity for
volatility_windows = [30, 90, 180, 365]

# Data to run correlations on
correlation_data = [
    'PriceUSD', 'AAPL_close', 'MSFT_close', 'GOOGL_close', 'AMZN_close',
    'NVDA_close', 'TSLA_close', 'META_close', 'BRK-A_close', 'BRK-B_close',
    'TSM_close', 'V_close', 'JPM_close', 'PYPL_close', 'GS_close',
    'FANG.AX_close', 'BITQ_close', 'CLOU_close', 'ARKK_close', 'XLK_close',
    'QQQ_close', 'IUIT.L_close', 'VTI_close', 'TLT_close', 'LQD_close', '^BCOM_close',
    'JNK_close', 'GLD_close', 'XLF_close', 'XLRE_close', 'SHY_close',
    'XLE_close', '^GSPC_close', '^VIX_close', '^IXIC_close', '^TNX_close',
    '^TNX_close', '^TYX_close', '^FVX_close', '^IRX_close', 'GC=F_close',
    'CL=F_close', 'SI=F_close', 'DX=F_close', 'AUDUSD=X_close',
    'CHFUSD=X_close', 'CNYUSD=X_close', 'EURUSD=X_close', 'GBPUSD=X_close',
    'HKDUSD=X_close', 'INRUSD=X_close', 'JPYUSD=X_close', 'RUBUSD=X_close'
]

# Metrics to calculate valuation targets
valuation_data_metrics = {
    'valuation_metrics': [
        'NVTAdj', 'NVTAdj90', 'mvrv_ratio', 'thermocap_multiple',
        '200_day_multiple', 'nvt_price_multiple', 'nvt_price', 'nvt_price_adj',
        'NVTAdj_percentile', 'NVTAdj90_percentile', 'mvrv_ratio_percentile',
        'thermocap_multiple_percentile', '200_day_multiple_percentile',
        'nvt_price_multiple_percentile', 'NVTAdj_zscore', 'mvrv_ratio_zscore',
        'thermocap_multiple_zscore', '200_day_multiple_zscore',
        'NVTAdj90_zscore', 'nvt_price_multiple_zscore', 'PriceUSD',
        'SF_Predicted_Price', 'SF_Multiple', 'SF_Predicted_Price_percentile',
        'SF_Multiple_percentile', 'SF_Predicted_Price_zscore',
        'SF_Multiple_zscore'
    ]
}

# Buy & Sell Targets For Valuation Metrics | Calcualted in
valuation_metrics = {
    '200_day_multiple': {
        'buy_target': [0.6969],
        'sell_target': [2.1659],
    },
    'mvrv_ratio': {
        'buy_target': [0.884],
        'sell_target': [3.183],
    },
    'nvt_price_multiple': {
        'buy_target': [0.54],
        'sell_target': [2.08],
    },
    'thermocap_multiple': {
        'buy_target': [5],
        'sell_target': [25.422],
    },
    'SF_Multiple': {
        'buy_target': [0.29],
        'sell_target': [3.1],
    },
    'market_cap_metrics': {
        'silver_marketcap_billion_usd': {
            'probabilities': {'bull': 0.95, 'base': 0.80, 'bear': 0.50}
        },
        'United_Kingdom_cap': {
            'probabilities': {'bull': 0.65, 'base': 0.35, 'bear': 0.10}
        },
        'AAPL_MarketCap': {
            'probabilities': {'bull': 0.55, 'base': 0.25, 'bear': 0.10}
        },
        'United_States_cap': {
            'probabilities': {'bull': 0.35, 'base': 0.15, 'bear': 0.05}
        },
        'gold_marketcap_billion_usd': {
            'probabilities': {'bull': 0.20, 'base': 0.10, 'bear': 0.05}
        }
    }
}
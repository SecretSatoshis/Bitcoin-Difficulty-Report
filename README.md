# Bitcoin Difficulty Report

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Workflow Summary](#workflow-summary)
6. [Output Overview](#output-overview)
7. [Report Interface](#report-interface)
8. [Data Sources](#data-sources)
9. [License](#license)

## Introduction
The Bitcoin Difficulty Report is a Python project designed to analyze various metrics and data points related to Bitcoin and other financial assets. The project retrieves data from various sources, processes it, and generates a comprehensive report that includes a wide range of financial metrics, valuation metrics, and statistical analyses. It is a valuable tool for investors, financial analysts, and cryptocurrency enthusiasts looking to gain insights into the cryptocurrency market and traditional financial markets.

## Setup
1. Clone the repository to your local machine:
 ```
git clone https://github.com/SecretSatoshis/Bitcoin-Difficulty-Report.git
 ```
2. Navigate to the project directory.
 ```
cd Bitcoin-Difficulty-Report
 ```
3. Install the necessary packages using the following command:
 ```
pip install -r requirements.txt
 ```
4. You are now ready to run the script.
 ```
python main.py
 ```

The project code can also be accessed and run on [Replit](https://replit.com/@SecretSatoshis/Bitcoin-Difficulty-Report).

## Usage
To generate the Bitcoin Difficulty Report, run the `main.py` script. This script integrates various functions from other modules in the project to collect data, perform analyses, and generate the report.


## Project Structure
- `main.py`: The main script that orchestrates the data retrieval, analysis, and report generation.
- `data_format.py`: A module that contains functions for data retrieval, processing, and analysis. It includes functions for calculating various financial metrics, statistical analyses, and generating valuation data.
- `report_table.py` and `datapane_report.py`: Modules for formatting and generating the final report (details to be added based on the analysis of these modules).

## Workflow Summary
The project follows a structured workflow orchestrated through the `main.py` script, encompassing the following steps:

1. **Data Retrieval**: The script initiates the data retrieval process, leveraging functions defined in `data_format.py` to fetch data from various sources, including bitcoin metrics, traditional finance market data, and more.

2. **Data Processing and Analysis**: Following data retrieval, the script performs a series of data processing and analysis tasks. These include calculating various on-chain metrics, moving averages, market capitalizations, and more, utilizing functions from the `data_format.py` file.

3. **Report Generation**: After processing the data, the script proceeds to generate a comprehensive report. The report includes a detailed analysis of various metrics and valuation data, presenting insights into the bitcoin market and traditional financial markets. This step leverages functions from the `report_table.py` and `datapane_report.py` modules to format and create the report.

4. **Report Output**: Finally, the generated report is outputted, providing users with a detailed and insightful analysis of the bitcoin market.

## Output Overview
The script generates two main types of outputs:
1. **HTML Report**: A detailed report in HTML format, which includes various analyses and visual representations of the data.
- You can download the report from `Difficulty_Report.html` or vist the [Report Link](https://secretsatoshis.github.io/Bitcoin-Difficulty-Report/Difficulty_Report.html).
2. **CSV Files**: Data files in CSV format that store the report datasets.

The project is set up with a GitHub workflow that automatically updates the data and regenerates the report on a daily basis.

## Report Interface
The report leverages Datapane, a Python library, to create an interactive and visually appealing interface. The `datapane_report.py` module contains functions that use Datapane to generate tables and other visual elements that make up the report's interface, providing users with an intuitive and comprehensive overview of the analyzed data.

## Data Sources
The project utilizes data from various sources. The data sources include:

### Bitcoin Data
- **Coinmetrics**
  - [Twitter](https://twitter.com/coinmetrics)
  - [GitHub](https://github.com/coinmetrics/data/tree/master/csv)

### M0 Money Supply and Gold Silver Data
- **CryptoVoices**
  - [Twitter](https://twitter.com/crypto_voices?lang=en)
  - [Website](https://porkopolis.io/basemoney)

### Traditional Finance Data
- **Yahoo Finance**
  - [Website](https://finance.yahoo.com/)
- **Yahoo_fin Python Library**
  - [Documentation](https://theautomatic.net/yahoo_fin-documentation/)

### Bitcoin Blockchain API
- **Blockstream**
  - [Twitter](https://twitter.com/Blockstream)
  - [API Documentation](https://github.com/Blockstream/esplora/blob/master/API.md)

## License
Distributed under the GNU GENERAL PUBLIC LICENSE. See `LICENSE` for more information.

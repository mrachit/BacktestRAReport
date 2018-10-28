class ReportParameters:
    # Parameters required in SmartBetaStrategy Class

    COMPARISON_SUBPLOT1_POSITION = 211
    COMPARISON_SUBPLOT1_TITLE = "Performance history"
    COMPARISON_SUBPLOT1_X_LABEL = ""

    COMPARISON_SUBPLOT1_COLOR1 = 'tab:green'
    COMPARISON_SUBPLOT1_Y_LABEL_LINE1 = "Wealth Growth over Benchmark"
    COMPARISON_SUBPLOT1_COLOR2 = "#808080"  # grey color
    COMPARISON_SUBPLOT1_Y_LABEL_LINE2 = "Relative Valuation"

    COMPARISON_SUBPLOT2_POSITION = 212
    COMPARISON_SUBPLOT2_TITLE = "Model"
    COMPARISON_SUBPLOT2_TITLE_X_LABEL = "Relative Valuation"
    COMPARISON_SUBPLOT2_TITLE_Y_LABEL = \
        "Subsequent %d yr Excess return (ann.)"

    # scatter plot - Actual subsequent subsequent year returns
    # (default 5 year returns annualized)

    # 1 for opaque - between 0 & 1
    COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_TRANSPARENCY = 0.5

    COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_MARKER = "o"  # circles
    COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_COLOR = 'b'  # black
    COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_LABEL = \
        "actual subsequent %d year excess returns (annualized)"

    # scatter plot -  Predicted (wo shrinkage) subsequent year returns
    # (default 5 year returns annualized)

    COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_TRANSPARENCY = 0.8  # 1 for opaque - between 0 & 1
    COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_MARKER = "."  # want relative_analytics line
    COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_COLOR = '#A9A9A9'  # dark gray
    COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_LABEL = \
        "predicted %d year excess returns (no shrinkage)"
    COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_LINEWIDTH = 2
    # scatter plot - current Valuation
    COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_TRANSPARENCY = 1  # 1 for opaque - between 0 & 1
    COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_MARKER = "o"  # want relative_analytics line
    COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_COLOR = 'r'  # Red
    COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_LABEL = "Current Valuation"
    COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_LINEWIDTH = 5

    # --------------------------------------------------------------------------
    # scatter plot - median valuation vertical line

    # 1 for opaque - between 0 & 1
    COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_TRANSPARENCY = 0.3
    COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_MARKER = "."  # want relative_analytics line
    COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_COLOR = 'g'  # Green
    COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_LABEL = "Median Valuation"

    # scatter plot - legend
    COMPARISON_SUBPLOT2_LEGEND_LOC = 'upper right'
    COMPARISON_SUBPLOT2_LEGEND_FONTSIZE = 6
    COMPARISON_SUBPLOT2_LEGEND_FANCYBOX = True
    COMPARISON_SUBPLOT2_LEGEND_SHADOW = True

    # subplots_adjust
    COMPARISON_SUBPLOTS_ADJUST_BOTTOM = 0.15
    COMPARISON_SUBPLOTS_ADJUST_TOP = 0.85

    # chart title
    COMPARISON_CHART_SUP_TITLE = "%strategyname %strategylocation, %startdate - " \
                                 "%enddate  \n" \
                                 "Benchmark: %benchmarkname %benchmarklocation"
    COMPARISON_CHART_TITLE_FONTSIZE = 14
    COMPARISON_CHART_TITLE_Y_LOCATION = 0.9
    COMPARISON_CHART_TITLE_X_LOCATION = 0.1

    CONCENTRATION_EFF_STOCK_TITLE_TEXT = "Concentration: Effective number of Holdings \n " \
                                        "%strategyname %strategylocation, %startdate - " \
                                 "%enddate  \n" \

    CONCENTRATION_STOCK_WEIGHT_TITLE_TEXT= "Concentration: Weight of Top Holdings\n " \
                                        "%strategyname %strategylocation, %startdate - " \
                                 "%enddate  \n" \


    # chart footer
    REFERENCE_DATA_TEXT = \
        'Source: Reasearch Affiliates, LLC, using data from CRSP/ ' \
        'Compustat/ Worldscope/ Datastream/ Bloomberg/ Dr.Kenneth ' \
        'French Data Library.'
    REFERENCE_USE_FOOTER_TEXT=\
        "This report is for internal use only and shall not be distributed " \
        "outside of Research Affiliates, LLC without explicit written approval."

    COMPARISON_CHART_FOOTER_TXT = (
         "\nRelative Valuation Ratio is the geometric mean of the four " 
         "relative ratios, PF of strategy/ PF of Benchmark." 
         "\nThe Price to Fundamentals (PF) are: Price/Book Value, Price/5 yr" 
         " Sales, Price/5 year Earnings, Price/5 year Dividends. "
         "\nLagged Fundamentals are used to calculate the relative valuation.")

    COMPARISON_CHART_FOOTER_FONTSIZE = 8

    SECTORAL_ALLOCATION_CHART_SUP_TITLE = \
        "Historical Allocation: Sector\n" \
        "%strategyname %strategylocation, %startdate - %enddate  \n"

    COUNTRY_ALLOCATION_CHART_SUP_TITLE = \
        "Historical Allocation: Country\n" \
        "%strategyname %strategylocation, %startdate - %enddate  \n"

    REGION_ALLOCATION_CHART_SUP_TITLE = \
        "Historical Allocation: Region\n" \
        "%strategyname %strategylocation, %startdate - %enddate  \n"

    PAPER_REFERENCE_TEXT_FORECAST_EXCESS_RETURN = \
        '\nReference: Arnott, Rob, Noah Beck, and Vitali Kalesnik. 2017. ' \
        '“Forecasting Factor and Smart Beta Returns (Hint: History Is Worse' \
        ' than Useless)” Research Affiliates Publications (February).\n'

    REFERENCE_TEXT_OUTPUT_COMPARISON_TABLE = \
        PAPER_REFERENCE_TEXT_FORECAST_EXCESS_RETURN + \
        "\n*Revaluation Alpha is the component of excess return that comes " \
        "from " \
        "changing valuations and is defined as the trend in valuation ratio" \
        " over the sample in time. " \
        "\n*Structural Alpha  is calculated as Historical excess Return - " \
        "Revaluation alpha. " \
        "\n*Relative Valuation Ratio is the geometric mean of the four " \
        "relative ratios, PF of strategy/ PF of Benchmark." \
        "\n The Price to Fundamentals (PF) are: Price/Book Value, Price/5 yr" \
        " Sales, Price/5 year Earnings, Price/5 year Dividends. Lagged " \
        "Fundamentals are used to calculate the relative valuation. " \
        "\n*Expected Excess Return is calculated as ( structural alpha + " \
        "\u03B2^ * Z-Score). \u03B2^ is the slope coefficient " \
        " from the regression of subsequent 5 year returns on Z-Score. Note " \
        "that this is different on SBI, where the slope parameter is a " \
        "shrinkage of multiple estimations. \n"

    PAPER_REFERENCE_TEXT_TRADING_COST = \
        '\nReference: Chow, Tzee-Man, Yadwinder Garg, Feifei Li, ' \
        'and Alex Pickard. 2017. ' \
        '"Cost and Capacity: Comparing Smart Beta Strategies." Research ' \
        'Affiliates Publications (July).\n'

    REFERENCE_TEXT_TRADING_COST_TABLE = \
        PAPER_REFERENCE_TEXT_TRADING_COST + \
        '\nTurnover is averaged over the entire period. Other characteristics' \
        ' are based on the most recent rebalancing. Market impact costs ' \
        ' assume $10 Billion in AUM and are averaged over the most recent five' \
        ' years. Capacity is the estimated AUM at which the strategy is' \
        ' expected to have 50 bps of market impact cost.'

    PAPER_REFERENCE_TEXT_EXCESS_RETURN_DECOMPOSITION = \
        '\nReference: Brightman, Chris, Mark Clements, and Vitali Kalesnik. 2017. ' \
        '“A Smart Beta for Sustainable Growth.” Research Affiliates ' \
        'Publications (July).'

    REFERENCE_TEXT_EXCESS_RETURN_DECOMPOSITION = \
        PAPER_REFERENCE_TEXT_EXCESS_RETURN_DECOMPOSITION + \
        '\nAnalyses use contemporaneous fundamentals, therefore the most ' \
        'recent six months of calculations are unavailable.'


    # update with server location if required:
    # \\\\ra.local\\DFS\Groups\\Research\Product Research\\Product Research
    # Team Shared Doc\\


    # Naming columns for the PDF table
    COMPARISON_COLUMN_NAMES_LIST_ROW1 = [
        'Strategy',
        'Excess Return',
        'Struct. Alpha',
        'Reval. Alpha',
        'T.E.',
        'Valuation Ratios',
        '',
        '',
        '\u03B2^',
        'Expected Excess Returns',
        ''
    ]
    COMPARISON_COLUMN_NAMES_LIST_ROW2 = [
        '',
        '',
        '',
        '',
        '',
        'Current',
        'Median',
        'Current z(ln(.))',
        '',
        'at 5 yr',
        'Std. Error'
    ]

    ANNUALIZED_EXCESS_RETURN_TABLE_COLUMNS = [
        "Strategy",
        '',
        '1yr',
        '3yr',
        '5yr',
        '10yr',
        '10yr Vol',
        '10yr SR'
    ]

    TRADING_COST_COLUMN_NAMES_LIST_ROW1 = [
        "Strategy",
        "WAMC ($M)",
        "Eff. N",
        "Turnover",
        "Turnover Concen<Br\>tration",
        "Portfolio Volume ($M)",
        "Tilt",
        "Market Impact Cost at $10B AUM (bps)",
        "Capacity at 50 bps of cost ($B)",
        "#trade days"
    ]

    EXCESS_RETURN_DECOMPOSITION_COLUMN_NAMES_LIST = [
        "Strategy",
        "Log Excess Returns",
        "Excess Return From Dividends",
        "Excess Growth in Valuation",
        "Excess Growth in EPS"
    ]

    REFERENCE_TEXT_VALUATIONS_RETURNS_TABLE = \
        "\nRelative Valuation Ratio is the geometric mean of the four " \
        "relative ratios, PF of strategy/ PF of Benchmark." \
        "\nThe Price to Fundamentals (PF) are: Price/Book Value, Price/5 yr" \
        " Sales, Price/5 year Earnings, Price/5 year Dividends."\
        "\nLagged Fundamentals are used for the analyses."

    REFERENCE_TEXT_VALUATIONS_NOLAG_RETURNS_TABLE =  \
        "\nRelative Valuation Ratio is the geometric mean of the four " \
        "relative ratios, PF of strategy/ PF of Benchmark." \
        "\nThe Price to Fundamentals (PF) are: Price/Book Value, Price/5 yr" \
        " Sales, Price/5 year Earnings, Price/5 year Dividends." \
        "\nAnalyses use contemporaneous fundamentals, therefore the most " \
        "recent six months of calculations are unavailable."

    REFERENCE_TEXT_VALUATIONS_NOLAG_COMBINED_RETURNS_TABLE = \
        "\nRelative Valuation Ratio is the geometric mean of the four " \
        "relative ratios, PF of strategy/ PF of Benchmark." \
        "\nThe Price to Fundamentals (PF) are: Price/Book Value, Price/5 yr" \
        " Sales, Price/5 year Earnings, Price/5 year Dividends." \
        "\nAnalyses use contemporaneous fundamentals, therefore the most " \
        "recent six months of calculations are unavailable.Lagged Fundamentals" \
        " are used when contemporaneous fundamnetals are not available (after %nolagdate)."


    PERFORMANCE_TABLE_COLUMN_NAMES_LIST_ROW1 = [
        'Strategy',
        'Absolute Performance',
        '',
        '',
        'Relative Performance',
        '',
        '',
        'Cost at $10B AUM',
        'Performance (net of cost)',
        ''
    ]
    PERFORMANCE_TABLE_COLUMN_NAMES_LIST_ROW2 = [
        '',
        'Return',
        'Volatility',
        'Sharpe Ratio',
        'Excess Return',
        'Tracking Error',
        'Info.<Br\>  Ratio',
        '',
        'Sharpe Ratio',
        'Info.<Br\>  Ratio'
    ]

    FF3_FACTOR_LOADING_COLUMN_NAMES =[
        'Strategy',
        'Alpha (annualized)',
        'Alpha (<i>t</i>-stat)',
        'MKT_RF',
        'SMB',
        'HML',
        'adjusted <i>R</i>\u00B2'
    ]

    FF5_FACTOR_LOADING_COLUMN_NAMES =[
        'Strategy',
        'Alpha (annualized)',
        'Alpha (<i>t</i>-stat)',
        'MKT_RF',
        'SMB',
        'HML',
        'RMW',
        'CMA',
        'adjusted <i>R</i>\u00B2'
    ]

    CARHART4_FACTOR_LOADING_COLUMN_NAMES=[
        'Strategy',
        'Alpha (annualized)',
        'Alpha (<i>t</i>-stat)',
        'MKT_RF',
        'SMB',
        'HML',
        'WML',
        'adjusted <i>R</i>\u00B2'
    ]

    MOMENTS_DOWNSIDE_RISK_LIST_NAMES =[
        'Strategy',
        'Avg.',
        'Std. Dev.',
        'Skewness',
        'Kurtosis',
        'Upside Semi-Dev.',
        'Downside Semi-Dev.',
        'Beta',
        'Upside Beta',
        'Downside Beta'
    ]

    VAR_WINRATE_LIST_NAMES_ROW1 =[
        'Strategy',
        'Worst observation (ending in)',
        '',
        '',
        '',
        '95% VaR',
        '',
        '',
        'Win-Rate',
        '',
        ''
        ]

    VAR_WINRATE_LIST_NAMES_ROW2 =[
        '',
        '1m',
        '12m',
        '36m',
        '60m',
        '12m',
        '36m',
        '60m',
        '12m',
        '36m',
        '60m'
        ]

    RECENT_PORTFOLIO_LIST_NAMES =[
        'Strategy',
        'Price to 5Yr Earnings',
        'Price to Book',
        'Price to 5Yr Sales',
        'Dividend Yield',
        'Price to 5Yr CashFlows',
        'Composite Valuation Ratio'
    ]

    HISTORICAL_PORT_CHARACTERISCS_LIST_NAMES=[
        'Strategy',
        'P/E',
        'P/B',
        'P/S',
        'P/D',
        'Rel P/E',
        'Rel P/B',
        'Rel P/S',
        'Rel P/D',
        'Rel Valuation Ratio'
    ]

    REFERENCE_TEXT_FACTOR_LOADING = '\n*** 99.9% level of confidence for significance.' \
                                     '\n ** 99% level of confidence for significance.' \
                                     '\n  * 95% level of confidence for significance.' \
                                    '\nAlpha is annualized by multiplying by 12.'


    REFERENCE_TEXT_PERFORMANCE = ''


    COLOR_DICT = {
        'basic materials':'#0082c8',
        'telecomm':'#e6194b',
        'cyclical':'#808080',
        'non-cyclical':'#ffd8b1',
        'energy':'#800000',
        'financial':'#ffe119',
        'industrial':'#3cb44b',
        'technology':'#000080',
        'utilities':'#f58231',
        'health care':'#911eb4',
        'us': '#0082c8',
        'uk': '#e6194b',
        'united_states': '#0082c8',
        'united_kingdom': '#e6194b',
        'other_europe': '#808080',
        'other_dev': '#ffd8b1',
        'japan': '#800000',
        'germany': '#ffe119',
        'france': '#e6beff',
        'em': '#aaffc3',
        'australia': '#3cb44b',
        'canada':'#ffa07a',
        'hong_kong':'#f58231',
        'israel':'#808000',
        'new_zealand':'#000080',
        'singapore':'#911eb4',
        'austria':'#c0c0c0',
        'belgium':'#daa520',
        'denmark':'#2e8b57',
        'finland':'#6495ed',
        'greece':'#d2f53c',
        'ireland':'#fabebe',
        'italy':'#008080',
        'luxembourg':'#46f0f0',
        'netherlands':'#aa6e28',
        'norway':'#fffac8',
        'portugal':'#000000',
        'spain':'#aaffc3',
        'sweden':'#808080',
        'switzerland':'#ffd8b1',
        'brazil':'#ffe119',
        'chile':'#fabebe',
        'china':'#da70d6',
        'colombia':'#911eb4',
        'czech_republic':'#46f0f0',
        'egypt':'#f032e6',
        'hungary':'#d2f53c',
        'india':'#3cb44b',
        'indonesia':'#008080',
        'korea(south)':'#e6beff',
        'malaysia':'#aa6e28',
        'mexico':'#fffac8',
        'morocco':'#800000',
        'peru':'#aaffc3',
        'philippines':'#808000',
        'poland':'#ffd8b1',
        'taiwan':'#000080',
        'thailand':'#808080',
        'turkey':'#000000',
        ' ': '#ffffff'
    }


    STANDARD_19_COLOR_LIST=[
        '#e6194b', #Red
        '#3cb44b', #Green
        '#ffe119', #Yellow
        '#0082c8', #Blue
        '#f58231', #Orange
        '#911eb4', #Purple
        '#46f0f0', #Cyan
        '#f032e6', #Magenta
        '#d2f53c', #Lime
        '#fabebe', #Pink
        '#008080', #Teal
        '#e6beff', #Lavender
        '#aa6e28', #Brown
        '#fffac8', #Beige
        '#800000', #Maroon
        '#aaffc3', #Mint
        '#808000', #Olive
        '#ffd8b1', #Coral
        '#000080', #Navy
        '#808080', #Grey
        '#000000' #Black
    ]

    REFERENCE_TEXT_EXCESS_RET_1 = REFERENCE_TEXT_FACTOR_LOADING + '\n '

    REFERENCE_TEXT_VAR_WINRATE_WORST= \
        '\nWorst Retrns for 1m are non-annualized monthly returns while for' \
        ' 12m,36m,60m the worst returns are annualized.' \
        '\n95% VaR is calculated as 5th percentile of the returns.' \
        '\nWin-Rate is calculated as percentage of times the annualized 12m,' \
        '36m, 60m return of strategy exceeds the corresponding return from ' \
        'benchmark.'

    REFERENCE_TEXT_YEAR_BY_YEAR_EXCESS_RETURNS = \
        '\nAnnual Excess Returns are calculated using excess returns from ' \
        'January to December of the year  as return of Strategy - ' \
        'return of Benchmark' \
        '\n* The first and last rows may reflect excess returns for less than ' \
        '1 year if simulations do not start and end in January and December.'

    REFERENCE_TEXT_YEAR_BY_YEAR_TOTAL_RETURNS = \
        '\nReturns are calculated using monthly returns from January to December ' \
        'of the year. ' \
        '\n* The first and last rows may reflect returns for less than 1 year' \
        ' if simulations do not start and end in January and December.'

    REFERENCE_TEXT_MOMENTS_DOWNSIDE_RISK =\
    '\nAverage excess returns is caclulated as  12* mean of monthly excess returns which is calculated as ' \
    '12*(( 1+ return from Strategy)/ ( 1+ return from benchmanrk) -1 )).' \
    '\nStandard Deviation is calculated as sample standard deviation of monthly excess returns * sqrt(12)' \
    '\nUpside Semi-Deviation is calculated as sample standard deviation of ' \
    'monthly excess returns * sqrt(12) when excess return is positive' \
    '\nDownside Semi-Deviation is calculated as sample standard deviation of ' \
    'monthly excess returns * sqrt(12) when excess return is zero or negative' \
    '\nUpside Beta is calculated as benchmark beta when benchmark – RF is positive' \
    '\nDownside Beta is calculated as benchmark beta when benchmark – RF is zero or negative'




class StrategyAnalyticsParameters:

    PB_INPUT_COL_NAME = 'price_to_book'
    PE_INPUT_COL_NAME = 'price_to_5yrearnings'
    PS_INPUT_COL_NAME = 'price_to_5yrsales'
    DIV_YIELD_INPUT_COL_NAME = 'dividend_yield_5yr'
    PCF_COL_NAME ='price_to_5yrcashflow'

    PB_NOLAG_INPUT_COL_NAME = 'price_to_book_nolag'
    PE_NOLAG_INPUT_COL_NAME = 'price_to_5yrearnings_nolag'
    PS_NOLAG_INPUT_COL_NAME = 'price_to_5yrsales_nolag'
    DIV_YIELD_NOLAG_INPUT_COL_NAME = 'dividend_yield_5yr_nolag'

    MONTHLY_RETURN_INPUT_COL_NAME = 'total_return'

    KEEP_CHARACTERISTICS_COL_LIST = ['date',
                                     'price_to_book',
                                     'price_to_5yrsales',
                                     'price_to_5yrearnings',
                                     'dividend_yield_5yr',
                                     'price_to_5yrcashflow',
                                     'price_to_earnings_nolag',
                                     'price_to_book_nolag',
                                     'price_to_5yrsales_nolag',
                                     'price_to_5yrearnings_nolag',
                                     'dividend_yield_5yr_nolag',
                                     'price_to_5yrcashflow_nolag'

                                     ]

    CHARACTERISTICS_TAB_NAME = 'Characteristics'
    RETURNS_TAB_NAME = 'Returns'
    CONCENTRATIONS_TAB_NAME = 'Concentrations'

    CUMULATIVE_RETURN_COL_NAME = 'CumProdReturn'
    MONTHLY_RETURN_COL_NAME = 'MonthlyReturn'
    MONTHS_TO_DATE_COL_NAME = 'MonthsToDate'
    NUMBER_OF_SUBSEQUENT_YEARS = 5
    SUBSEQUENT_RETURN_COL_NAME = ('Subsequent' +
                                  str(NUMBER_OF_SUBSEQUENT_YEARS) +
                                  'YrReturn')

    HISTORICAL_NOM_ITD_COL_NAME = 'HistITDNominalReturn'
    HISTORICAL_YEAR_RETURN_COL_NAME = 'Historical' + '%year' + 'YrReturn'

    KEEP_RETURNS_COL_LIST = ['date', 'total_return', 'ret_ex_div']
    NO_OF_SUBSEQUENT_YEARS = 5

    FF3_col_names = {
        'date': 'Date',
        'hml': 'value_factor',
        'rf': 'RF',
        'mkt_rf': 'mkt_rf',
        'smb': 'size_factor_FF'
    }

    STANDARD_REGION_LIST = [
        'EM',
        'Other_Dev',
        'Japan',
        'Other_Europe',
        'France',
        'Germany',
        'UK',
        'US'
    ]


class RelativeStrategyAnalyticsParameters(StrategyAnalyticsParameters):

    BENCHMARK_COL_LIST = [
        StrategyAnalyticsParameters.MONTHLY_RETURN_COL_NAME,
        StrategyAnalyticsParameters.CUMULATIVE_RETURN_COL_NAME,
        'PB',
        'PS',
        'PD',
        'PE',
        'nolag_PE',
        'nolag_PB',
        'nolag_PS',
        'nolag_PD',
        StrategyAnalyticsParameters.HISTORICAL_NOM_ITD_COL_NAME,
        StrategyAnalyticsParameters.SUBSEQUENT_RETURN_COL_NAME,
        StrategyAnalyticsParameters.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
            '%year', '1'),
        StrategyAnalyticsParameters.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
            '%year', '3'),
        StrategyAnalyticsParameters.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
            '%year', '5'),
        StrategyAnalyticsParameters.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
            '%year', '10')
                          ]

    RELATIVE_PB_COL_NAME = 'RelPB'
    RELATIVE_PD_COL_NAME = 'RelPD'
    RELATIVE_PE_COL_NAME = 'RelPE'
    RELATIVE_PS_COL_NAME = 'RelPS'

    RELATIVE_PB_NOLAG_COL_NAME = 'nolag_RelPB'
    RELATIVE_PD_NOLAG_COL_NAME = 'nolag_RelPD'
    RELATIVE_PE_NOLAG_COL_NAME = 'nolag_RelPE'
    RELATIVE_PS_NOLAG_COL_NAME = 'nolag_RelPS'

    SUBSEQUENT_EXCESS_RET_COL_NAME = (
            'SubsequentExcess' +
            str(StrategyAnalyticsParameters.NUMBER_OF_SUBSEQUENT_YEARS) +
            'YrReturn')
    STRATEGY_MINUS_BENCHMARK_12M_RETURN_COL_NAME = 'Strategy_minus_benchmark_12M_return'

    HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME \
        = 'HistoricalExcess' + '%year' + 'YrReturn'
    HISTORICAL_EXCESS_ITD_COL_NAME = 'HistITDExGross'

    REL_MEASURE_NOLAG_COL_NAME = 'nolag_RelValAgg'
    LOG_REL_MEASURE_NOLAG_COL_NAME = 'nolag_Log_RelValAgg'
    Z_SCORE_NOLAG_COL_NAME = 'nolag_ZScore'

    REL_MEASURE_COL_NAME = 'RelValAgg'
    LOG_REL_MEASURE_COL_NAME = 'Log_RelValAgg'
    Z_SCORE_COL_NAME = 'ZScore'

    REL_MEASURE_CHANGE_COL_NAME = 'PercentChange' + '%year' + 'YrRelVal'
    REL_MEASURE_NOLAG_CHANGE_COL_NAME = 'nolag_PercentChange' + '%year' + 'YrRelVal'

    CUM_PROD_EXCESS_COL_NAME = 'CumProdExcessReturn'

    EXPECTED_EXCESS_RETURN_COL_NAME = 'Exp5yrExcessReturn'
    EXCESS_RETURN_COL_NAME='Excess_Ret_over_benchmark'
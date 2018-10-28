from ras.core.reporting.ReportParameters import ReportParameters
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_CENTER


class ReportDataClassCard:

    def __init__(self,
                 name: str,
                 index_name: str='',
                 title_text: str='',
                 output_name: str='',
                 index_section_name: str='',
                 index_page_num: int=0,
                 footer_text: str ='',
                 is_table: bool=False,
                 col_widths: list=None,
                 add_tbl_styles: list = None,
                 switch: bool=True,
                 values: list=None,
                 number_of_rows_to_repeat_next_page: int=1,
                 alternate_background_rows: int=1,
                 data_align=TA_RIGHT,
                 separate_startegy_plots: bool = False,
                 excel_output: bool = False,
                 excel_values: list = None
                 ):
        """

        Args:
            name:
            index_name:
            title_text:
            output_name:
            index_section_name:
            index_page_num:
            footer_text:
            is_table:
            col_widths:
            add_tbl_styles:
            switch:
            values:
            number_of_rows_to_repeat_next_page:
            alternate_background_rows:
            data_align:
            separate_startegy_plots:
        """
        self.name = name
        self.index_name = index_name
        self.output_name = output_name
        self.title_text = title_text
        self.index_section_name = index_section_name
        self.index_page_num = index_page_num
        self.footer_text = footer_text
        self.is_table = is_table
        self.col_widths = col_widths
        self.switch = switch,
        self.values = values
        self.add_tbl_styles = add_tbl_styles
        self.number_of_rows_to_repeat_next_page = \
            number_of_rows_to_repeat_next_page
        self.alternate_background_rows = alternate_background_rows
        self.data_align = data_align
        self.separate_startegy_plots = separate_startegy_plots
        self.excel_output = excel_output
        self.excel_values = excel_values

class RavisualDataParameters:
    ANALYSES_ORDER = [
        'hist_abs_rel_perf',
        'performance_at_yearly_horizon',
        'year_on_year_strategy_minus_benchmark_returns',
        'year_on_year_total_return',
        'market_impact_cost',
        'net_of_cost_SR_IR',
        'moments_downside_risks',
        'worst_period_var_winrate',
        'attributions_ff3',
        'attributions_carhart4',
        'attributions_ff5',
        'attributions_custom',
        'attributions_ff3_standardized',
        'attributions_carhart4_standardized',
        'attributions_ff5_standardized',
        'attributions_custom_standardized',
        'excess_return_decomposition',
        'excess_return_correlation',
        'tracking_error',
        'recent_allocation_sector',
        'recent_allocation_region',
        'recent_allocation_country',
        'historical_allocation_sector',
        'historical_allocation_region',
        'historical_allocation_country',
        'concentration_holdings',
        'concentration_weight',
        'recent_portfolio_characteristics',
        'hist_avg_portfolio_characteristics',
        'expected_excess_return_table',
        'expected_excess_return_charts'
    ]

    INDEX_SECTION_ORDER = [
        'Performance:',
        'Cost:',
        'Risks:',
        'Attributions:',
        'Characteristics of Excess Returns:',
        'Characteristics of Portfolios:',
        'Forward Looking Expectations:'
    ]

    def __init__(self):
        analyses_set = dict()
        analyses_set['hist_abs_rel_perf'] = ReportDataClassCard(
            name='hist_abs_rel_perf',
            is_table=True,
            index_name='Historical Absolute and Relative Performance',
            title_text='Historical Absolute and Relative Performance \n '
                       '%region, %start_date - %end_date \n'
                       'Benchmark: %benchmark',
            index_section_name='Performance:',
            output_name='//performance_table.pdf',
            footer_text=ReportParameters.REFERENCE_TEXT_PERFORMANCE,
            col_widths=[2.5 * inch] + [(7.5/9) * inch] * 9,
            add_tbl_styles=[
                    ('SPAN', (0, 0), (0, 1)),
                    ('SPAN', (1, 0), (3, 0)),
                    ('SPAN', (4, 0), (6, 0)),
                    ('SPAN', (7, 0), (7, 1)),
                    ('SPAN', (8, 0), (9, 0)),
                    ('BOX', (1, 0), (3, -1), 0.75, colors.black),
                    ('BOX', (0, 0), (9, 1), 0.75, colors.black),
                    ('BOX', (0, 0), (6, -1), 0.75, colors.black),
                    ('BOX', (8, 0), (9, -1), 0.75, colors.black),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.black)
                ],
            values=[ReportParameters.PERFORMANCE_TABLE_COLUMN_NAMES_LIST_ROW1,
                    ReportParameters.PERFORMANCE_TABLE_COLUMN_NAMES_LIST_ROW2],
            number_of_rows_to_repeat_next_page=2,
            excel_values=[
                ReportParameters.PERFORMANCE_TABLE_COLUMN_NAMES_LIST_ROW1,
                ReportParameters.PERFORMANCE_TABLE_COLUMN_NAMES_LIST_ROW2]
        )

        analyses_set['performance_at_yearly_horizon'] = ReportDataClassCard(
            name='performance_at_yearly_horizon',
            is_table=True,
            index_name='Performance at 1, 3, 5, 10 Year Horizons',
            title_text='Performance at 1, 3, 5, 10 Year Horizons\n'
                       '%region, %start_date - %end_date \n'
                       'Benchmark: %benchmark',
            index_section_name='Performance:',
            output_name='//annualized_excess_combinedreturn_table.pdf',
            footer_text=
            ReportParameters.REFERENCE_TEXT_VALUATIONS_NOLAG_COMBINED_RETURNS_TABLE,
            col_widths=[2.5 * inch] + [1.5 * inch] + [1 * inch] * 6,
            add_tbl_styles=[],
            values=[ReportParameters.ANNUALIZED_EXCESS_RETURN_TABLE_COLUMNS],
            alternate_background_rows=2,
            excel_values=[ReportParameters.ANNUALIZED_EXCESS_RETURN_TABLE_COLUMNS]
        )

        analyses_set['year_on_year_strategy_minus_benchmark_returns'] = \
            ReportDataClassCard(
                name='year_on_year_strategy_minus_benchmark_returns',
                is_table=True,
                index_section_name='Performance:',
                index_name='Year on Year Strategy minus Benchmark returns',
                title_text='Year on Year Strategy minus Benchmark returns \n'
                           '%region, %start_date - %end_date',
                output_name="//yearly_excess_ret.pdf",
                footer_text=
                ReportParameters.REFERENCE_TEXT_YEAR_BY_YEAR_EXCESS_RETURNS,
                add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)]
        )
        analyses_set['year_on_year_total_return'] = ReportDataClassCard(
            name='year_on_year_total_return',
            is_table=True,

            index_name='Year on Year Total Returns',
            title_text='Year on Year Total Returns \n'
                       '%region, %start_date - %end_date',
            output_name='//yearly_total_ret.pdf',
            index_section_name='Performance:',
            footer_text=ReportParameters.REFERENCE_TEXT_YEAR_BY_YEAR_TOTAL_RETURNS,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)]
        )

        analyses_set['market_impact_cost'] = ReportDataClassCard(
            name='market_impact_cost',
            is_table=True,
            index_name='Turnover, Capacity, Market Impact Cost and its '
                       'Decomposition',
            title_text='Turnover, Capacity, Market Impact Cost and its '
                       'Decomposition \n'
                       '%region, %start_date - %end_date',
            output_name='//trading_costs_table.pdf',
            index_section_name='Cost:',
            footer_text=ReportParameters.REFERENCE_TEXT_TRADING_COST_TABLE,
            add_tbl_styles=[
                             ('BOX', (0, 0), (0, -1), 0.75, colors.black),
                             ('BOX', (7, 0), (7, -1), 0.75, colors.black),
                             ('BOX', (8, 0), (8, -1), 0.75, colors.black),
                             ('BOX', (0, 0), (9, 1), 0.75, colors.black)
            ],
            col_widths=[2.5 * inch] + [0.75 * inch] * 6 + [1.125 * inch] * 2 +
                       [0.75 * inch],
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.TRADING_COST_COLUMN_NAMES_LIST_ROW1]
        )

        analyses_set['net_of_cost_SR_IR'] = ReportDataClassCard(
            name='net_of_cost_SR_IR',
            is_table=False,
            index_name='Net of Cost Sharpe Ratios and Information Ratios',
            title_text='Net of Cost Sharpe Ratios and Information Ratios\n'
                       '%region, %start_date - %end_date\n',
            output_name='//SR_IR.pdf',
            index_section_name='Cost:',
            footer_text='Cost calculations are made with respect to $10B AUM',
            add_tbl_styles=[
                             ('BOX', (0, 0), (0, -1), 0.75, colors.black),
                             ('BOX', (7, 0), (7, -1), 0.75, colors.black),
                             ('BOX', (8, 0), (8, -1), 0.75, colors.black),
                             ('BOX', (0, 0), (9, 1), 0.75, colors.black)
            ]
        )

        analyses_set['moments_downside_risks'] = ReportDataClassCard(
            name='moments_downside_risks',
            is_table=True,
            index_name='Excess Return Distribution: Moments & Down-side Risks',
            title_text='Excess Return Distribution: Moments & Down-side Risks \n'
                       '%region, %start_date - %end_date\n'
                       'Benchmark: %benchmark',
            output_name='//mom_downrisks.pdf',
            index_section_name='Risks:',
            footer_text=ReportParameters.REFERENCE_TEXT_MOMENTS_DOWNSIDE_RISK,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [0.7 * inch] + [0.85*inch]*8,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.MOMENTS_DOWNSIDE_RISK_LIST_NAMES],
            excel_values=[ReportParameters.MOMENTS_DOWNSIDE_RISK_LIST_NAMES]
        )

        analyses_set['worst_period_var_winrate'] = ReportDataClassCard(
            name='worst_period_var_winrate',
            is_table=True,
            index_name='Excess Return Distribution: Worst Period, VaR, Win-Rate',
            title_text='Excess Return Distribution: Worst Period, VaR, Win-Rate \n'
                       '%region, %start_date - %end_date\n'
                       'Benchmark: %benchmark',
            output_name='//var_winrate.pdf',
            index_section_name='Risks:',
            footer_text=ReportParameters.REFERENCE_TEXT_VAR_WINRATE_WORST,
            add_tbl_styles=[
                ('SPAN', (0, 0), (0, 1)),
                ('SPAN', (1, 0), (4, 0)),
                ('SPAN', (5, 0), (7, 0)),
                ('SPAN', (8, 0), (10, 0)),
                ('BOX', (1, 0), (4, -1), 0.75, colors.black),
                ('BOX', (5, 0), (7, -1), 0.75, colors.black),
                ('BOX', (8, 0), (10, -1), 0.75, colors.black),
                ('BOX', (0, 0), (10, 1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [.9 * inch] * 4 + [(3.9 / 6) * inch] * 6,
            number_of_rows_to_repeat_next_page=2,
            values=[ReportParameters.VAR_WINRATE_LIST_NAMES_ROW1,
                    ReportParameters.VAR_WINRATE_LIST_NAMES_ROW2],
            excel_values=[ReportParameters.VAR_WINRATE_LIST_NAMES_ROW1,
                          ReportParameters.VAR_WINRATE_LIST_NAMES_ROW2]
        )

        analyses_set['attributions_ff3'] = ReportDataClassCard(
            name='attributions_ff3',
            is_table=True,
            index_name='Factor Attributions: Fama French Three Factor Model',
            title_text='Factor Attributions: Fama French Three Factor Model \n'
                       '%region, %start_date - %end_date',
            output_name='//ff3.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [(7.5 / 6) * inch] * 6,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.FF3_FACTOR_LOADING_COLUMN_NAMES],
            excel_values=[ReportParameters.FF3_FACTOR_LOADING_COLUMN_NAMES],
            data_align=TA_CENTER
        )

        analyses_set['attributions_ff5'] = ReportDataClassCard(
            name='attributions_ff5',
            is_table=True,
            index_name='Factor Attributions: Fama French Five Factor Model',
            title_text='Factor Attributions: Fama French Five Factor Model \n'
                       '%region, %start_date - %end_date',
            output_name='//ff5.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [(7.5 / 8) * inch] * 8,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.FF5_FACTOR_LOADING_COLUMN_NAMES],
            excel_values=[ReportParameters.FF5_FACTOR_LOADING_COLUMN_NAMES],
            data_align=TA_CENTER
        )

        analyses_set['attributions_carhart4'] = ReportDataClassCard(
            name='attributions_carhart4',
            is_table=True,
            index_name='Factor Attributions: Carhart Four Factor Model',
            title_text='Factor Attributions: Carhart Four Factor Model \n'
                       '%region, %start_date - %end_date',
            output_name='//carhart4.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [(7.5 / 7) * inch] * 7,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.CARHART4_FACTOR_LOADING_COLUMN_NAMES],
            excel_values=[ReportParameters.CARHART4_FACTOR_LOADING_COLUMN_NAMES],
            data_align=TA_CENTER
        )

        analyses_set['attributions_custom'] = ReportDataClassCard(
            name='attributions_custom',
            is_table=True,
            index_name='Factor Attributions: Custom Model',
            title_text='Factor Attributions: Custom Model \n'
                       '%region, %start_date - %end_date',
            output_name='//factor_attributions_custom.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            values=[['Custom Model Still Under Construction']],
            excel_values=[['Custom Model Still Under Construction']],
            col_widths=[10.0 * inch]
        )

        analyses_set['attributions_ff3_standardized'] = ReportDataClassCard(
            name='attributions_ff3_standardized',
            is_table=True,
            index_name='Factor Attributions: Fama French Three Factor Model '
                       '(standardized)',
            title_text='Factor Attributions: Fama French Three Factor Model '
                       '(standardized) \n'
                       '%region, %start_date - %end_date',
            output_name='//ff3_std.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [(7.5 / 6) * inch] * 6,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.FF3_FACTOR_LOADING_COLUMN_NAMES],
            excel_values=[ReportParameters.FF3_FACTOR_LOADING_COLUMN_NAMES],
            data_align=TA_CENTER
        )

        analyses_set['attributions_ff5_standardized'] = ReportDataClassCard(
            name='attributions_ff5_standardized',
            is_table=True,
            index_name='Factor Attributions: Fama French Five Factor Model '
                       '(standardized)',
            title_text='Factor Attributions: Fama French Five Factor Model '
                       '(standardized)\n'
                       '%region, %start_date - %end_date',
            output_name='//ff5_std.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [(7.5 / 8) * inch] * 8,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.FF5_FACTOR_LOADING_COLUMN_NAMES],
            excel_values=[ReportParameters.FF5_FACTOR_LOADING_COLUMN_NAMES],
            data_align=TA_CENTER
        )

        analyses_set['attributions_carhart4_standardized'] = ReportDataClassCard(
            name='attributions_carhart4_standardized',
            is_table=True,
            index_name='Factor Attributions: Carhart Four Factor Model '
                       '(standardized)',
            title_text='Factor Attributions: Carhart Four Factor Model '
                       '(standardized) \n'
                       '%region, %start_date - %end_date',
            output_name='//carhart4_std.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            col_widths=[2.5 * inch] + [(7.5 / 7) * inch] * 7,
            number_of_rows_to_repeat_next_page=1,
            values=[ReportParameters.CARHART4_FACTOR_LOADING_COLUMN_NAMES],
            excel_values=[
                ReportParameters.CARHART4_FACTOR_LOADING_COLUMN_NAMES],
            data_align=TA_CENTER
        )

        analyses_set['attributions_custom_standardized'] = ReportDataClassCard(
            name='attributions_custom_standardized',
            is_table=True,
            index_name='Factor Attributions: Custom Model (standardized)',
            title_text='Factor Attributions: Custom Model (standardized) \n'
                       '%region, %start_date - %end_date',
            output_name='//factor_attributions_custom_std.pdf',
            index_section_name='Attributions:',
            footer_text=ReportParameters.REFERENCE_TEXT_FACTOR_LOADING,
            values=[['Custom Model Still Under Construction']],
            excel_values=[['Custom Model Still Under Construction']],
            col_widths=[10.0 * inch]
        )

        analyses_set['excess_return_decomposition'] = ReportDataClassCard(
            name='excess_return_decomposition',
            is_table=True,
            index_name='Log Excess Return Decomposition',
            title_text='Log Excess Return Decomposition\n'
                       '%region, %start_date - %nolagdate\n'
                       'Benchmark: %benchmark',
            output_name='//return_decomposition_table.pdf',
            index_section_name='Attributions:',
            col_widths=[2.5 * inch] + [(7.5 / 4) * inch] * 4,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            footer_text
            =ReportParameters.REFERENCE_TEXT_EXCESS_RETURN_DECOMPOSITION,
            values
            =[ReportParameters.EXCESS_RETURN_DECOMPOSITION_COLUMN_NAMES_LIST],
            excel_values
            =[ReportParameters.EXCESS_RETURN_DECOMPOSITION_COLUMN_NAMES_LIST]
        )

        analyses_set['excess_return_correlation'] = ReportDataClassCard(
            name='excess_return_correlation',
            is_table=True,
            index_name='Excess Return Correlation',
            title_text='Excess Return Correlation\n'
                       '%region, %start_date - %end_date\n'
                       'Benchmark: %benchmark',
            output_name='//excess_ret_correlation.pdf',
            index_section_name='Characteristics of Excess Returns:',
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            footer_text=ReportParameters.REFERENCE_TEXT_EXCESS_RETURN_DECOMPOSITION
        )

        analyses_set['tracking_error'] = ReportDataClassCard(
            name='tracking_error',
            is_table=True,
            index_name='Tracking Error',
            title_text='Tracking Error\n'
                       '%region, %start_date - %end_date',
            output_name='//tracking_error.pdf',
            index_section_name='Characteristics of Excess Returns:',
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            footer_text=ReportParameters.REFERENCE_TEXT_EXCESS_RETURN_DECOMPOSITION
        )

        analyses_set['recent_allocation_sector'] = ReportDataClassCard(
            name='recent_allocation_sector',
            is_table=False,
            index_name='Recent Allocation: Sector',
            title_text='Recent Allocation: Sector\n'
                       '%region, %start_date - %end_date\n',
            output_name='//recent_Sector.pdf',
            index_section_name='Characteristics of Portfolios:'
        )

        analyses_set['recent_allocation_region'] = ReportDataClassCard(
            name='recent_allocation_region',
            is_table=False,
            index_name='Recent Allocation: Region',
            title_text='Recent Allocation: Region\n'
                       '%region, %start_date - %end_date\n',
            output_name='//recent_region.pdf',
            index_section_name='Characteristics of Portfolios:'
        )

        analyses_set['recent_allocation_country'] = ReportDataClassCard(
            name='recent_allocation_country',
            is_table=False,
            index_name='Recent Allocation: Country',
            title_text='Recent Allocation: Country\n'
                       '%region, %start_date - %end_date\n',
            output_name='//recent_country.pdf',
            index_section_name='Characteristics of Portfolios:'
        )

        analyses_set['historical_allocation_sector'] = ReportDataClassCard(
            name='historical_allocation_sector',
            is_table=False,
            index_name='Historical Allocation: Sector',
            title_text='Historical Allocation: Sector\n'
                       '%strategyname %region, %start_date - %end_date\n',
            index_section_name='Characteristics of Portfolios:',
            output_name='\\sector_allocation_',
            separate_startegy_plots=True
        )

        analyses_set['historical_allocation_region'] = ReportDataClassCard(
            name='historical_allocation_region',
            is_table=False,
            index_name='Historical Allocation: Region',
            title_text='Historical Allocation: Region\n'
                       '%strategyname %region, %start_date - %end_date\n',
            index_section_name='Characteristics of Portfolios:',
            output_name='\\region_allocation_',
            separate_startegy_plots=True
        )

        analyses_set['historical_allocation_country'] = ReportDataClassCard(
            name='historical_allocation_country',
            is_table=False,
            index_name='Historical Allocation: Country',
            title_text='Historical Allocation: Country\n'
                       '%strategyname %region, %start_date - %end_date\n',
            index_section_name='Characteristics of Portfolios:',
            output_name='\\country_allocation_',
            separate_startegy_plots=True
        )

        analyses_set['concentration_holdings'] = ReportDataClassCard(
            name='concentration_holdings',
            is_table=False,
            index_name='Concentration: Effective number of Holdings',
            title_text='Concentration: Effective number of Holdings\n'
                       '%region, %start_date - %end_date\n',
            index_section_name='Characteristics of Portfolios:',
            output_name='//conc_effn.pdf'
        )


        analyses_set['concentration_weight'] = ReportDataClassCard(
            name='concentration_weight',
            is_table=False,
            index_name='Concentration: Weight of Top Holdings',
            title_text='Concentration: Weight of Top Holdings\n'
                       '%region, %start_date - %end_date\n',
            index_section_name='Characteristics of Portfolios:',
            output_name='//conc_weight.pdf'
        )
        analyses_set['recent_portfolio_characteristics'] = ReportDataClassCard(
            name='recent_portfolio_characteristics',
            is_table=True,
            index_name='Recent Portfolio Characteristics',
            title_text='Recent Portfolio Characteristics\n'
                       '%region, %end_date\n',
            footer_text='\nComposite Valuation Ratio is calculated as geometric '
                        'mean of Price to book, Price to 5Yr Sales, Price to 5Yr '
                        'Earnings, Price to 5Yr Dividends. '
                        'Lagged Fundamentals are used for the analyses.',
            output_name='//recent_port_chars.pdf',
            col_widths=[2.5 * inch] + [(7.5/6)* inch] * 6,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black),
                ('BOX', (0, 0), (-1, 1), 0.75, colors.black)],
            index_section_name='Characteristics of Portfolios:',
        )

        analyses_set['hist_avg_portfolio_characteristics'] = ReportDataClassCard(
            name='hist_avg_portfolio_characteristics',
            is_table=True,
            index_name='Historical Average Portfolio Characteristics',
            title_text='Historical Average Portfolio Characteristics\n'
                       '%region, %start_date - %end_date\n'
                       'Bechmark: %benchmark',
            footer_text='\nRelative Valuation Ratio is calculated as geometric '
                        'mean of Relative Price to book (Rel P/B), Relative Price '
                        'to 5Yr Sales (Rel P/S), Relative Price to 5Yr Earnings '
                        '(Rel P/E), Relative Price to 5Yr Dividends(Rel P/D. '
                        'Lagged Fundamentals are used for the analyses.',
            col_widths=[2.5 * inch] + [(7.5 / 9) * inch] * 9,
            add_tbl_styles=[('BOX', (0, 0), (0, -1), 0.75, colors.black)],
            index_section_name='Characteristics of Portfolios:',
            output_name='//hist_avg_port.pdf'
        )

        analyses_set['expected_excess_return_table'] = ReportDataClassCard(
            name='expected_excess_return_table',
            is_table=True,
            index_name='Excess Return,Structural and Revaluation Alpha, '
                       'Expected Excess Return',
            title_text='Excess Return,Structural and Revaluation Alpha, '
                       'Expected Excess Return\n'
                       '%region, %start_date - %end_date\n'
                       'Bechmark: %benchmark',
            values=[ReportParameters.COMPARISON_COLUMN_NAMES_LIST_ROW1,
                    ReportParameters.COMPARISON_COLUMN_NAMES_LIST_ROW2],
            excel_values=[ReportParameters.COMPARISON_COLUMN_NAMES_LIST_ROW1,
                    ReportParameters.COMPARISON_COLUMN_NAMES_LIST_ROW2],
            index_section_name='Forward Looking Expectations:',
            add_tbl_styles=[('SPAN', (0, 0), (0, 1)),
                            ('SPAN', (1, 0), (1, 1)),
                            ('SPAN', (2, 0), (2, 1)),
                            ('SPAN', (3, 0), (3, 1)),
                            ('SPAN', (4, 0), (4, 1)),
                            ('SPAN', (8, 0), (8, 1)),
                            ('SPAN', (5, 0), (7, 0)),
                            ('SPAN', (9, 0), (10, 0)),
                            ('BOX', (0, 0), (0, -1), 0.75, colors.black),
                            ('BOX', (5, 0), (7, -1), 0.75, colors.black),
                            ('BOX', (9, 0), (10, -1), 0.75, colors.black),
                            ('BOX', (0, 0), (-1, 1), 0.75, colors.black)
                            ],
            col_widths=[2.5 * inch] + [0.75 * inch] * 10,
            footer_text
            =ReportParameters.REFERENCE_TEXT_OUTPUT_COMPARISON_TABLE,
            number_of_rows_to_repeat_next_page=2,
            output_name='//excess_returns_table.pdf'
        )

        analyses_set['expected_excess_return_charts'] = ReportDataClassCard(
            name='expected_excess_return',
            is_table=False,
            index_name='Expected Excess Return Charts',
            title_text='Expected Excess Return\n'
                       '%strategyname %region, %start_date - %end_date\n'
                       'Bechmark: %benchmark %benchmark_region',
            index_section_name='Forward Looking Expectations:',
            output_name="//model_comparison_",
            separate_startegy_plots=True
        )
        self.analyses_set = analyses_set


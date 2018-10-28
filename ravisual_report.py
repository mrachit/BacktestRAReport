from ras.core.reporting.strategy_relative_analytics import StrategyRelativeAnalytics
from ras.core.reporting.strategy_analytics import StrategyAnalytics
from ras.core.reporting.Report import Report
from ras.core.reporting.ReportParameters import ReportParameters
from reportlab.lib.colors import red, green
from ras.core.reporting.strategy_analytics_parameters import StrategyAnalyticsParameters, \
    RelativeStrategyAnalyticsParameters as rsa_params
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.platypus import Paragraph,SimpleDocTemplate
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from ras.core.reporting.ravisual_data_parameters import RavisualDataParameters


class ProdResearchReport:
    @staticmethod
    def format_for_float(x,
                         excel_format: bool = False,
                         decimals: int = 2) -> str:
        """
        returns formatted  float till the the number of decimals
        Args:
            x: value to format
            excel_format: Bool if True format for excel False for pdf
            decimals: number of decimals in the formating

        Returns:

        """
        if type(x) == str or excel_format:
            return x
        else:
            format_specifier = '{:.' + str(decimals) + 'f}'
            return format_specifier.format(x)

    @staticmethod
    def format_for_percent(
            x,
            excel_format: bool = False,
            decimals: int = 2) -> str:
        """
        returns formatted % value for input decimals
        Args:
            x: value to format
            excel_format: Bool if True format for excel False for pdf
            decimals: number of decimals in the formating

        Returns:

        """
        if type(x) == str:
            return x
        elif excel_format:
            return '{:%}'.format(x)
        else:
            format_specifier = '{:.' + str(decimals) + '%}'
            return format_specifier.format(x)

    @staticmethod
    def create_concentration_pdf(
            strategy_list: list,
            output_filename: str,
            col_name_1: str,
            col_name_2: str,
            title_1: str,
            title_2: str,
            date_col: str = 'date',
            sup_title_text: str = "Concentration PDF",
            format_to_percent: bool = False,
            fig_size: tuple = (11, 8.5)
    ):
        """
        Creates the concentration chart pdf -- for both the holdings and weight

        Args:
            strategy_list: List of StrategyAnalytics objects
            output_filename: Output pdf name
            col_name_1: Column name for analysis 1 (y left chart)
            col_name_2: Column name for analysis 2 (y right chart)
            date_col: Date column name  which will be x axis
            title_1: Title of chart 1
            title_2: Title chart 2
            sup_title_text: Super Title text
            format_to_percent: Bool to convert the Y axis to percent or not
            fig_size: Tuple in inches for pdf size

        Returns:
            None, creates a pdf output
        """
        fig = plt.figure(figsize=fig_size)
        # Plot left chart
        ax1 = fig.add_subplot(121)
        i = 0
        for strategy in strategy_list:
            ax1.plot(strategy.concentration_df[date_col],
                     strategy.concentration_df[col_name_1],
                     color=ReportParameters.STANDARD_19_COLOR_LIST[i],
                     label=strategy.name

                     )
            i = i + 1

        ax1.grid(alpha=0.5)
        ax1.set_title(title_1)

        # Plot Right chart
        ax2 = fig.add_subplot(122)
        i = 0
        for strategy in strategy_list:
            ax2.plot(strategy.concentration_df[date_col],
                     strategy.concentration_df[col_name_2],
                     color=ReportParameters.STANDARD_19_COLOR_LIST[i],
                     label=strategy.name
                     )
            i = i + 1

        ax2.grid(alpha=0.5)
        ax2.set_title(title_2)
        ax2.set_ylim(bottom=0)

        # Format Y axis to percent if required
        if format_to_percent:
            ax1.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda j, _: '{:.0%}'.format(j)))

            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda j, _: '{:.0%}'.format(j)))

        # Adjust both the charts to make space for legends and title
        plt.subplots_adjust(bottom=0.3, top=0.8, wspace=0.2, hspace=0.2)
        # Legends
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=handles,
            labels=labels,
            loc='upper center',
            bbox_to_anchor=(1.1, -0.1),
            ncol=3
        )
        # Title
        fig.suptitle(
            sup_title_text,
            fontsize=ReportParameters.COMPARISON_CHART_TITLE_FONTSIZE,
            y=ReportParameters.COMPARISON_CHART_TITLE_Y_LOCATION,
            x=ReportParameters.COMPARISON_CHART_TITLE_X_LOCATION,
            horizontalalignment='left'
        )
        plt.close()
        # Save PDf
        fig.savefig(output_filename, format='pdf')

    @classmethod
    def get_historical_portfolio_chars_data(
            cls,
            rsa_list: list,
            excel_output: bool = False
    ) -> list:
        """
        Get formatted  data for the historical portfolio characteristics table
        Args:
            rsa_list: list of StrategyRelativeAnalytics objects
            excel_output: Bool to flag excel ( True) or pdf (False) output
        Returns:
            list of formatted  data table for  historical portfolio
            characteristics table

        """
        temp = [ReportParameters.HISTORICAL_PORT_CHARACTERISCS_LIST_NAMES]
        for x in rsa_list:
            df = x.all_data
            temp.append(
                [
                    x.name,
                    cls.format_for_float(x=np.mean(df['PE'].copy()),
                                         excel_format=excel_output),
                    cls.format_for_float(x=np.mean(df['PB'].copy()),
                                         excel_format=excel_output),
                    cls.format_for_float(x=np.mean(df['PS'].copy()),
                                         excel_format=excel_output),
                    cls.format_for_float(x=np.mean(df['PD'].copy()),
                                         excel_format=excel_output),
                    cls.format_for_float(x=np.mean(
                        df[rsa_params.RELATIVE_PE_COL_NAME].copy()),
                        excel_format=excel_output),
                    cls.format_for_float(x=np.mean(
                        df[rsa_params.RELATIVE_PB_COL_NAME].copy()),
                        excel_format=excel_output),
                    cls.format_for_float(x=np.mean(
                        df[rsa_params.RELATIVE_PS_COL_NAME].copy()),
                        excel_format=excel_output),
                    cls.format_for_float(x=np.mean(
                        df[rsa_params.RELATIVE_PD_COL_NAME].copy()),
                        excel_format=excel_output),
                    cls.format_for_float(x=np.mean(
                        df[rsa_params.REL_MEASURE_COL_NAME].copy()),
                        excel_format=excel_output)
                ]
            )

        return temp

    @classmethod
    def get_recent_portfolio_chars_data(cls,
                                        strategy_list: list,
                                        excel_output: bool = False) -> list:
        """
        Get formatted  data for the recent portfolio characteristics table
        Args:
            strategy_list: list of StrategyAnalytics objects
            excel_output: Bool to flag excel (True) or pdf (False) output
        Returns:
            list of formatted data table for recent portfolio
            characteristics table

        """
        temp = [ReportParameters.RECENT_PORTFOLIO_LIST_NAMES]
        for x in strategy_list:
            recent_characteristics = StrategyAnalytics. \
                get_recent_portfolio_characteristics(x.all_data)
            temp.append(
                [
                    x.name,
                    cls.format_for_float(x=recent_characteristics[
                        'recent_PE'],
                                         excel_format=excel_output),
                    cls.format_for_float(x=recent_characteristics[
                        'recent_PB'],
                                         excel_format=excel_output),
                    cls.format_for_float(x=recent_characteristics[
                        'recent_PS'],
                                         excel_format=excel_output),
                    cls.format_for_float(x=recent_characteristics[
                        'recent_div_yield'],
                                         excel_format=excel_output),
                    cls.format_for_float(x=recent_characteristics[
                        'recent_PCF'],
                                         excel_format=excel_output),
                    cls.format_for_float(x=recent_characteristics[
                        'recent_composite_fundamental'],
                                         excel_format=excel_output)
                ]
            )
        return temp

    @staticmethod
    def create_SR_IR_chart(relative_strategy_list,
                           output_filename,
                           sup_title_text='Net of Cost Sharpe Ratios and '
                                          'Information Ratios',
                           title_1='SR vs IR',
                           title_2='SR vs IR(net of cost)',
                           y_label_1='SR',
                           y_label_2='SR (net of cost)',
                           x_label_1='IR',
                           x_label_2='IR (net of cost)',
                           footer_text='Cost calculations are made with '
                                       'respect to $10B AUM',
                           fig_size: tuple = (11, 8.5)
                           ):
        """
        Creates the SR/IR chart PDF
        Args:
            relative_strategy_list:
            output_filename:
            sup_title_text:
            title_1:
            title_2:
            y_label_1:
            y_label_2:
            x_label_1:
            x_label_2:
            footer_text:
            fig_size:

        Returns:

        """
        fig = plt.figure(figsize=fig_size)
        # ----------------------------------------------------------------------
        # Draw Left Chart
        ax1 = fig.add_subplot(121)

        ax1.set_xlabel(x_label_1)

        ax1.set_ylabel(y_label_1)

        i = 0
        for rsa_obj in relative_strategy_list:
            ax1.scatter(
                x=rsa_obj.current_IR,
                y=rsa_obj.strategy.HistITDSharpe,
                color=ReportParameters.STANDARD_19_COLOR_LIST[i],
                edgecolors='#000000',
                marker="o",
                linewidths=1,
                label=rsa_obj.name
            )
            i = i + 1

        # If the auto range is within 0 & 1 fix it within 0 & 1
        lower1, upper1 = ax1.get_ylim()
        left1, right1 = ax1.get_xlim()
        if upper1 < 1:
            upper1 = 1
        if right1 < 1:
            right1 = 1
        if lower1 > 0:
            lower1 = 0
        if left1 > 0:
            left1 = 0

        ax1.set_ylim(bottom=lower1, top=upper1)
        ax1.set_xlim(left=left1, right=right1)

        # format X & Y axis labels
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.2f}'.format(j)))

        ax1.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.2f}'.format(j)))

        ax1.grid(alpha=0.5)
        ax1.set_title(title_1)

        # ----------------------------------------------------------------------
        # Draw Right chart
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(x_label_2)
        ax2.set_ylabel(y_label_2)

        i = 0
        for rsa_obj in relative_strategy_list:
            ax2.scatter(
                x=rsa_obj.IR_net_cost,
                y=rsa_obj.strategy.SR_net_cost,
                color=ReportParameters.STANDARD_19_COLOR_LIST[i],
                edgecolors='#000000',
                marker="o",
                linewidths=1,
                label=rsa_obj.name
            )
            i = i + 1

        # If the auto range is within 0 & 1 fix it within 0 & 1
        lower2, upper2 = ax2.get_ylim()
        left2, right2 = ax2.get_xlim()

        if upper2 < 1:
            upper2 = 1
        if right2 < 1:
            right2 = 1
        if lower2 > 0:
            lower2 = 0
        if left2 > 0:
            left2 = 0

        ax2.set_ylim(bottom=lower2, top=upper2)
        ax2.set_xlim(left=left2, right=right2)

        # Format X & Y axis
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.2f}'.format(j)))
        ax2.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.2f}'.format(j)))

        ax2.grid(alpha=0.5)
        ax2.set_title(title_2)

        # ----------------------------------------------------------------------
        # Adjust chart size
        plt.subplots_adjust(bottom=0.3, top=0.8, wspace=0.4, hspace=0.4)
        handles, labels = ax1.get_legend_handles_labels()
        # Legends
        ax1.legend(
            handles=handles[::-1],
            labels=labels[::-1],
            loc='upper center',
            bbox_to_anchor=(1.1, -0.18),
            ncol=4
        )
        # Title
        fig.suptitle(
            sup_title_text,
            fontsize=ReportParameters.COMPARISON_CHART_TITLE_FONTSIZE,
            y=ReportParameters.COMPARISON_CHART_TITLE_Y_LOCATION,
            x=ReportParameters.COMPARISON_CHART_TITLE_X_LOCATION,
            horizontalalignment='left'
        )
        # Footer
        ax1.text(x=0,
                 y=-0.12,
                 s=footer_text,
                 size=ReportParameters.COMPARISON_CHART_FOOTER_FONTSIZE,
                 transform=ax1.transAxes)
        plt.close()
        # Save PDF
        fig.savefig(output_filename, format='pdf')

    @classmethod
    def year_by_year_return(
            cls,
            strategy_list,
            cumprod_column,
            ret_col_name_1yr,
            excel_output: bool = False
    ) -> list:
        """
        Calculates year by year return for all strategies and returns a 2d list
        Args:
            strategy_list:
            cumprod_column:
            ret_col_name_1yr:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:

        """
        # partial start year
        partial_start_year_returns = []
        # partial start year
        partial_last_year_returns = []

        # get valid 12m excess return index
        year_list = []
        index_list = []
        # ----------------------------------------------------------------------
        # for complete years
        min_valid_date = strategy_list[0].all_data[ret_col_name_1yr]. \
            first_valid_index()
        max_valid_date = strategy_list[0].all_data[ret_col_name_1yr]. \
            last_valid_index()
        if min_valid_date.month == 1:
            year = min_valid_date.year
            year_list.append(str(year - 1))
            index_list.append(pd.to_datetime(datetime.date(year=year,
                                                           month=1,
                                                           day=1)))
        # ----------------------------------------------------------------------
        else:  # for partial start year
            year = min_valid_date.year
            partial_start_year_returns.append(str(year - 1) + '*')
            partial_year_date = pd.to_datetime(datetime.date(year=year,
                                                             month=1,
                                                             day=1))
            # partial first year returns:
            for x in strategy_list:
                partial_start_year_returns.append(
                    float(x.all_data[x.all_data.index == partial_year_date][
                              cumprod_column].copy().values - 1))

        for i in range(min_valid_date.year + 1, max_valid_date.year + 1):
            year_list.append(str(i - 1))
            index_list.append(pd.to_datetime(datetime.date(year=i,
                                                           month=1,
                                                           day=1)))
        # ----------------------------------------------------------------------
        col_name_list = ['Year']
        df = pd.DataFrame(pd.Series(year_list))
        for x in strategy_list:
            df[x.name] = pd.Series(x.all_data[x.all_data.index.isin(
                index_list)][ret_col_name_1yr].copy().values)
            col_name_list.append(x.name)
        # ----------------------------------------------------------------------
        # partial last year
        max_date = max(strategy_list[0].all_data.index)
        max_complete_year_date = max(index_list)
        if max_date > max_complete_year_date:
            partial_last_year_returns.append(str(max_date.year) + '*')
            for x in strategy_list:
                partial_last_year_returns.append(
                    float((x.all_data[x.all_data.index == max_date][
                               cumprod_column].copy().values / float(
                        x.all_data[x.all_data.index == max_complete_year_date][
                            cumprod_column].copy().values)) - 1))

        # ----------------------------------------------------------------------
        if max_date > max_complete_year_date:
            data_list = [col_name_list] + [partial_start_year_returns] + \
                        df.values.tolist() + [partial_last_year_returns]
        else:
            data_list = [col_name_list] + [partial_start_year_returns] + \
                        df.values.tolist()

        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                data_list[i][j] = cls.format_for_percent(
                    x=data_list[i][j],
                    excel_format=excel_output)

        return data_list

    @classmethod
    def year_by_year_strategy_minus_benchmark_return(
            cls,
            relative_strategy_list: list,
            strategy_minus_benchmark_ret_col_name: str,
            cum_monthly_return_column_name: str,
            excel_output: bool = False
    ) -> list:
        """
        Calculates year by year strategy minus benchmark return for all
        strategies and returns a 2d list
        Args:
            relative_strategy_list:
            strategy_minus_benchmark_ret_col_name:
            cum_monthly_return_column_name:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:

        """
        # partial start year
        partial_start_year_returns = []
        # partial start year
        partial_last_year_returns = []

        # get valid 12m excess return index
        year_list = []
        index_list = []

        # for complete years
        min_valid_date = relative_strategy_list[0].all_data[
            strategy_minus_benchmark_ret_col_name].first_valid_index()
        max_valid_date = relative_strategy_list[0].all_data[
            strategy_minus_benchmark_ret_col_name].last_valid_index()
        if min_valid_date.month == 1:
            year = min_valid_date.year
            year_list.append(str(year - 1))
            index_list.append(
                pd.to_datetime(datetime.date(year=year, month=1, day=1)))
        else:  # for partial start year
            year = min_valid_date.year
            partial_start_year_returns.append(str(year - 1) + '*')
            partial_year_date = pd.to_datetime(
                datetime.date(year=year, month=1, day=1))

            # partial first year returns:

            for x in relative_strategy_list:
                partial_start_year_returns.append(
                    float(x.all_data[x.all_data.index == partial_year_date][
                              cum_monthly_return_column_name].copy().values - 1)
                    - float(x.all_data[x.all_data.index == partial_year_date][
                                cum_monthly_return_column_name + '_B'].copy().
                            values - 1)

                )

        for i in range(min_valid_date.year + 1, max_valid_date.year + 1):
            year_list.append(str(i - 1))
            index_list.append(
                pd.to_datetime(datetime.date(year=i, month=1, day=1)))

        col_name_list = ['Year']
        df = pd.DataFrame(pd.Series(year_list))
        for x in relative_strategy_list:
            df[x.name] = pd.Series(
                x.all_data[x.all_data.index.isin(index_list)][
                    strategy_minus_benchmark_ret_col_name].copy().values)
            col_name_list.append(x.name)

        max_date = max(relative_strategy_list[0].all_data.index)
        max_complete_year_date = max(index_list)
        if max_date > max_complete_year_date:
            partial_last_year_returns.append(str(max_date.year) + '*')
            for x in relative_strategy_list:
                partial_last_year_returns.append(
                    float((x.all_data[x.all_data.index == max_date][
                               cum_monthly_return_column_name].copy().values /
                           float(x.all_data[x.all_data.index ==
                                            max_complete_year_date]
                                 [cum_monthly_return_column_name].copy().values)
                           ) - 1) -
                    float((x.all_data[x.all_data.index == max_date][
                               cum_monthly_return_column_name + '_B'].copy()
                           .values / float(x.all_data[x.all_data.index ==
                                                      max_complete_year_date]
                                           [
                                               cum_monthly_return_column_name + '_B'].copy()
                                           .values)) - 1)
                )

            data_list = [col_name_list] + [partial_start_year_returns] + \
                        df.values.tolist() + [partial_last_year_returns]
        else:
            data_list = [col_name_list] + [partial_start_year_returns] + \
                        df.values.tolist()

        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                data_list[i][j] = cls.format_for_percent(
                    x=data_list[i][j],
                    excel_format=excel_output)

        return data_list

    @staticmethod
    def recent_allocation_bar_chart(strategy_list: list,
                                    output_filename: str,
                                    chart_type: str,
                                    bottom: float = 0.3,
                                    top: float = 0.8
                                    ):
        """
        creates recent allocation bar chart ( Region/ Country/ Sector) for all
        strategies
        Args:
            strategy_list:
            output_filename:
            chart_type:
            bottom:
            top:

        Returns:

        """

        benchmark = strategy_list[0]
        column_list_x = []
        name_list = []
        df_list = []
        new_col_list = []
        for i in range(len(strategy_list)):
            x = strategy_list[i]
            if chart_type == 'Sector':
                strategy_df = x.sector_allocation_df
            elif chart_type == 'Region':
                strategy_df = x.region_allocation_df
            elif chart_type == 'Country':
                strategy_df = x.country_allocation_df

            else:
                strategy_df = x.sector_allocation_df
            strategy_df['name'] = i
            name_list.append(x.name)
            column_list_x.append(i)
            df_list.append(strategy_df)

        benchmark_df = df_list[0]

        for i in range(1, len(df_list)):
            new_col_list = new_col_list + list(set(df_list[i].columns) -
                                               set(benchmark_df.columns))

        if chart_type == 'Country':
            new_col_list = list(set(new_col_list))
            all_col_list = sorted(new_col_list + list(benchmark_df.columns))
            for new_col in new_col_list:
                benchmark_df[new_col] = 0
            benchmark_df = benchmark_df[all_col_list].copy()

        df_list = [benchmark_df] + df_list[1:]

        recent_df_list = []
        for x in df_list:
            recent_df_list.append(x.iloc[-1])

        text1 = ''
        text2 = ''
        text3 = ''
        for i in range(len(name_list)):

            if i % 3 == 0:
                text1 = text1 + '(' + str(i) + '): ' + name_list[i] + '\n'
            if i % 3 == 1:
                text2 = text2 + '(' + str(i) + '): ' + name_list[i] + '\n'
            if i % 3 == 2:
                text3 = text3 + '(' + str(i) + '): ' + name_list[i] + '\n'

        # Add sort argument to pd.concat to remove argument in pandas 0.23
        combined_df = pd.concat(objs=recent_df_list,
                                axis=1).transpose()

        combined_df.fillna(0, inplace=True)
        x = column_list_x
        y = combined_df.drop(['name'].copy(), axis=1)
        xtick_list = []
        for item in column_list_x:
            xtick_list.append('(' + str(item) + ')')

        color_list = []
        column_list = []
        for i in (list(y.columns)):
            color_list.append(
                ReportParameters.COLOR_DICT[i.lower()])
            column_list.append(i)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)

        bottom_list = []
        for i in range(len(strategy_list)):
            bottom_list.append(0)
        bottom_list = pd.Series(bottom_list)

        for i in range(len(list(y.columns))):
            ax.bar(x, y[column_list[i]],
                   bottom=bottom_list,
                   label=column_list[i],
                   color=color_list[i])
            bottom_list = bottom_list + y[column_list[i]].values
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_list)
        ax.set_ylim(bottom=0, top=1)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.0%}'.format(j)))

        plt.subplots_adjust(bottom=bottom, top=top, left=0.1, right=0.7,
                            hspace=0.2)

        end_date = benchmark.last_date - pd.Timedelta(1, unit='D')
        fig.suptitle(
            t='Recent Allocation: ' + chart_type + '\n' + benchmark.location +
              ', Latest Snapshot:' + end_date.strftime("%b %Y"),
            fontsize=ReportParameters.COMPARISON_CHART_TITLE_FONTSIZE,
            y=ReportParameters.COMPARISON_CHART_TITLE_Y_LOCATION,
            x=ReportParameters.COMPARISON_CHART_TITLE_X_LOCATION,
            horizontalalignment='left'
        )

        ax.set_xlabel('Strategies')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles[::-1],
            labels=labels[::-1],
            loc='upper center',
            bbox_to_anchor=(1.2, 1)
        )

        ax.set_ylim(bottom=0, top=1)
        ax.text(x=0.0,
                y=-0.15,
                s=text1,
                va='top',
                size=9,
                transform=ax.transAxes)

        ax.text(x=0.33,
                y=-0.15,
                s=text2,
                va='top',
                size=9,
                transform=ax.transAxes)

        ax.text(x=.67,
                y=-0.15,
                s=text3,
                va='top',
                size=9,
                transform=ax.transAxes)

        plt.close()
        fig.savefig(output_filename, format='pdf')

    @classmethod
    def get_mom_down_side_risk(cls,
                               sra_obj: StrategyRelativeAnalytics,
                               excel_output: bool = False) -> list:
        """
        Gives the formatted data for Momentum & Down-side risks of strategies
        Args:
            sra_obj:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:
            list of formatted values

        """
        return [
            sra_obj.name,
            cls.format_for_percent(x=sra_obj.excess_ret_distribution['avg'],
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.excess_ret_distribution['std_dev'],
                                   excel_format=excel_output),
            cls.format_for_float(x=sra_obj.excess_ret_distribution['skewness'],
                                 excel_format=excel_output),
            cls.format_for_float(x=sra_obj.excess_ret_distribution['kurtosis'],
                                 excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.excess_ret_distribution[
                'up_side_dev'], excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.excess_ret_distribution[
                'down_side_dev'], excel_format=excel_output),
            cls.format_for_float(x=sra_obj.excess_ret_distribution[
                'beta_results']['Mkt_RF'], excel_format=excel_output),
            cls.format_for_float(x=sra_obj.excess_ret_distribution[
                'up_side_beta_results']['Mkt_RF'], excel_format=excel_output),
            cls.format_for_float(x=sra_obj.excess_ret_distribution[
                'down_side_beta_results']['Mkt_RF'], excel_format=excel_output)
        ]

    @classmethod
    def get_var_win_rate(cls,
                         sra_obj: StrategyRelativeAnalytics,
                         excel_output: bool = False) -> list:
        """
        Gives the formatted data for VaR, Win-rate  of strategies
        Args:
            sra_obj:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:
            list of formatted values
        """
        return [
            sra_obj.name,
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['worst_return_1m'],
                excel_format=excel_output) + ' (' +
            sra_obj.excess_ret_distribution['worst_return_1m_date'].strftime(
                "%b %Y") + ')',
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['worst_return_12m'],
                excel_format=excel_output) + ' (' +
            sra_obj.excess_ret_distribution['worst_return_12m_date'].strftime(
                "%b %Y") + ')',
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['worst_return_36m'],
                excel_format=excel_output) + ' (' +
            sra_obj.excess_ret_distribution['worst_return_36m_date'].strftime(
                "%b %Y") + ')',
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['worst_return_60m'],
                excel_format=excel_output) + ' (' +
            sra_obj.excess_ret_distribution['worst_return_60m_date'].strftime(
                "%b %Y") + ')',
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['5_percentile_1yr'],
                excel_format=excel_output,
                decimals=1
            ),
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['5_percentile_3yr'],
                excel_format=excel_output,
                decimals=1
            ),
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['5_percentile_5yr'],
                excel_format=excel_output,
                decimals=1
            ),
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['win_rate_1yr'],
                excel_format=excel_output,
                decimals=1
            ),
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['win_rate_3yr'],
                excel_format=excel_output,
                decimals=1
            ),
            cls.format_for_percent(
                x=sra_obj.excess_ret_distribution['win_rate_5yr'],
                excel_format=excel_output,
                decimals=1
            )
        ]

    @classmethod
    def get_hist_tracking_error_2d_list(cls,
                                        strategy_list: list,
                                        col_name: str,
                                        excel_output: bool = False
                                        ) -> list:
        """
        Get historical trackick error of all strategies with each other
        Args:
            strategy_list:
            col_name:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:

        """
        n = len(strategy_list)

        data_list = [['Strategy'] + ['(' + str(i) + ')' for i in range(
            len(strategy_list))]]
        for i in range(n):
            temp = [strategy_list[i].name + ' (' + str(i) + ')']
            for j in range(n):
                temp.append(
                    cls.format_for_percent(
                        x=cls.get_hist_annualized_tracking_error(
                            strategy_return_series=strategy_list[i].
                                all_data[col_name].copy(),
                            benchmark_return_series=strategy_list[j].
                                all_data[col_name].copy()),
                        excel_format=excel_output
                    )
                )
            data_list.append(temp)

        return data_list

    @classmethod
    def get_corr_2d_list(cls,
                         strategy_list: list,
                         col_name,
                         excel_output: bool = False
                         ) -> list:
        """
        Get historical correlation of excess returns for all strategies with
        each other
        Args:
            strategy_list:
            col_name:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:

        """
        df = pd.DataFrame(
            strategy_list[0].all_data[col_name].copy())
        df.rename(columns={df.columns[0]: strategy_list[0].name},
                  inplace=True)
        for i in range(1, len(strategy_list)):
            df[strategy_list[i].name] = strategy_list[i].all_data[
                col_name].copy()

        df = df.iloc[1:, :]
        a = df.corr()
        columns = ['Strategy'] + ['(' + str(i + 1) + ')' for i in range(
            len(a.columns))]

        rows = [[cls.format_for_float(x=i, excel_format=excel_output) for i in
                 row] for row in a.itertuples()]

        for i in range(len(rows)):
            rows[i][0] = rows[i][0] + ' (' + str(i + 1) + ')'

        # 'Excess_Ret_over_benchmark'

        return [columns] + rows

    @classmethod
    def get_output_excess_return_decomposition(
            cls,
            sra_obj: StrategyRelativeAnalytics,
            excel_output: bool = False
    ) -> list:
        """
        Gives the formatted values for excess return decomposition table
        Args:
            sra_obj:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:
            list
        """
        return [
            sra_obj.name,
            cls.format_for_percent(
                x=sra_obj.log_excess_ret_decomposition['log_excess_return'],
                excel_format=excel_output),
            cls.format_for_percent(
                x=sra_obj.log_excess_ret_decomposition['log_excess_div_return'],
                excel_format=excel_output),
            cls.format_for_percent(
                x=sra_obj.log_excess_ret_decomposition[
                    'log_excess_growth_PE_return'],
                excel_format=excel_output),
            cls.format_for_percent(
                x=sra_obj.log_excess_ret_decomposition[
                    'log_excess_earnings_growth_return'],
                excel_format=excel_output)
        ]

    @classmethod
    def get_output_comparison_formatted(
            cls,
            sra_obj: StrategyRelativeAnalytics,
            excel_output: bool = False
    ) -> list:
        """
        generates the formatted list of forecasting excess return table

        Args:
            sra_obj: StrategyRelativeAnalytics obj
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:
            the formatted list for comparison pdf
        """
        return [
            sra_obj.name,
            cls.format_for_percent(x=sra_obj.current_hist_itd_excess_gross,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_hist_itd_struct_alpha,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_reval_alpha,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_hist_itd_tracking_error,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_rel_val_agg,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_rel_val_agg_50th,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_z_log_rel_val_agg,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_hist_itd_struct_beta,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_expected_excess_return,
                                   excel_format=excel_output),
            cls.format_for_percent(x=sra_obj.current_exp_5yr_excess_SE,
                                   excel_format=excel_output)
        ]

    @classmethod
    def get_annualized_returns_nolag_concat_table_formatted(
            cls,
            sra_obj: StrategyRelativeAnalytics,
            excel_output: bool = False
    ) -> list:
        """
        generates the formatted list for annualized returns pdf using
        contemporaneous fundamentals concatted by lagged fundamentals
        Args:
            sra_obj:
            excel_output: Bool to flag excel (True) or pdf (False) output
        Returns:
            list of

        """
        return ([
            [
                sra_obj.name,
                'Excess Returns',
                cls.format_for_percent(
                    x=sra_obj.current_hist_1yr_nolag_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_hist_3yr_nolag_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_hist_5yr_nolag_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_hist_10yr_nolag_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.strategy.Hist10YrVol,
                    excel_format=excel_output),
                cls.format_for_float(
                    x=sra_obj.strategy.Hist10YrSharpe,
                    excel_format=excel_output)
            ],
            [
                '',
                'Change in Valuation',
                cls.format_for_percent(
                    x=sra_obj.current_percent_change_1yr_nolag_concat_relval,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_percent_change_3yr_nolag_concat_relval,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_percent_change_5yr_nolag_concat_relval,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_percent_change_10yr_nolag_concat_relval,
                    excel_format=excel_output),
                '',
                ''
            ]
        ])

    @classmethod
    def get_annualized_returns_table_formatted(
            cls,
            sra_obj: StrategyRelativeAnalytics,
            excel_output: bool = False
    ) -> list:
        """
        generates the formatted list for annualized returns pdf using lagged
        fundamentals

        Args:
            sra_obj: StrategyRelativeAnalytics obj
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:
            the formatted list for annualized returns pdf
        """
        return ([
            [
                sra_obj.name,
                'Excess Returns',
                cls.format_for_percent(
                    x=sra_obj.current_hist_1yr_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_hist_3yr_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_hist_5yr_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.current_hist_10yr_excess_return_annualized,
                    excel_format=excel_output),
                cls.format_for_percent(
                    x=sra_obj.strategy.Hist10YrVol,
                    excel_format=excel_output),
                cls.format_for_float(
                    x=sra_obj.strategy.Hist10YrSharpe,
                    excel_format=excel_output)
            ],
            [
                '',
                'Change in Valuation',
                '{:.2%}'.format(sra_obj
                                .current_percent_change_1yr_rel_val),
                '{:.2%}'.format(sra_obj
                                .current_percent_change_3yr_rel_val),
                '{:.2%}'.format(sra_obj
                                .current_percent_change_5yr_rel_val),
                '{:.2%}'.format(sra_obj
                                .current_percent_change_10yr_rel_val),
                '',
                ''
            ]
        ])

    @classmethod
    def get_trading_costs_formatted(
            cls,
            strategy: StrategyAnalytics,
            excel_output: bool = False
    ) -> list:
        """
        TODO: check cost , dynamic AUM
        Args:

            strategy:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:


        """
        return ([
            strategy.name,
            # No immediate function solution to handle this case
            '{:,}'.format(int(round(strategy.trading_cost["WAMC"]))),
            cls.format_for_float(
                x=strategy.trading_cost["EffN"],
                excel_format=excel_output),
            cls.format_for_float(
                x=strategy.trading_cost["TO"],
                excel_format=excel_output),
            cls.format_for_float(
                x=strategy.trading_cost["TO_Concentration"],
                excel_format=excel_output),
            # No immediate function solution to handle this case
            '{:,}'.format(int(round(strategy.trading_cost["Volume"]))),
            cls.format_for_float(
                x=strategy.trading_cost["Tilt"],
                excel_format=excel_output),
            cls.format_for_float(
                x=strategy.trading_cost["Cost"] * 10000,
                excel_format=excel_output),
            cls.format_for_float(
                x=strategy.trading_cost["Capacity"],
                excel_format=excel_output),
            # No immediate function solution to handle this case
            '{:,}'.format(strategy.trading_cost["Rebalance_freq"])
        ])

    @classmethod
    def get_performance_benchmark_formatted(
            cls,
            strategy: StrategyAnalytics,
            excel_output: bool = False
    ) -> list:
        """
        TODO:

        Args:
            strategy:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:


        """
        SR = strategy.HistITDSharpe
        cost = strategy.trading_cost["Cost"]
        vol = strategy.HistITDVol

        return ([
            strategy.name,
            cls.format_for_percent(
                x=strategy.current_nominal_itd_return,
                excel_format=excel_output),
            cls.format_for_percent(
                x=vol,
                excel_format=excel_output),
            cls.format_for_float(
                x=SR,
                excel_format=excel_output),
            '',
            '',
            '',
            cls.format_for_percent(
                x=cost,
                excel_format=excel_output),
            cls.format_for_float(
                x=(SR * vol - cost) / vol,
                excel_format=excel_output),
            ''
        ])

    @classmethod
    def reg_result_formatting(cls,
                              coefficient: float,
                              pvalue: float,
                              to_float: bool = True,
                              excel_output: bool = False) -> str:
        """
        Adds relevant number of * to coefficient based on significance level
        using the p-value
        Args:
            coefficient: the coefficient to be compared
            pvalue: the pvalue
            to_float: Bool to convert it to float or percent
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:
            str

        """
        #  float or percent formatting
        if to_float:
            prefix = cls.format_for_float(
                x=coefficient,
                excel_format=excel_output)
        else:
            prefix = cls.format_for_percent(
                x=coefficient,
                excel_format=excel_output)
        # Suffix based on pvalue
        if pvalue <= 0.001:
            suffix = str('***')
        elif pvalue <= 0.01:
            suffix = str('** ')
        elif pvalue <= 0.05:
            suffix = str('*  ')
        else:
            suffix = str('   ')

        return str(prefix) + suffix

    @classmethod
    def get_factor_loading_formatted(
            cls,
            strategy: StrategyAnalytics,
            df: pd.DataFrame,
            col_list: list,
            excel_output: bool = False,
            standardized_flag: bool = False
    ) -> list:
        """
        gives factor attributions from any regression in a 2d list
        Args:
            strategy:
            col_list:
            excel_output: Bool to flag excel (True) or pdf (False) output
            standardized_flag:

        Returns:
            2d list  the
        """
        results = StrategyAnalytics.get_regression_results(
            df=df,
            return_col=StrategyAnalyticsParameters.MONTHLY_RETURN_COL_NAME,
            rf_col='RF',
            col_list=col_list,
            standardized_x=standardized_flag
        )

        temp = [
            strategy.name,
            cls.reg_result_formatting(
                coefficient=results['alpha'],
                pvalue=results['alpha_pvalue'],
                to_float=False,
                excel_output=excel_output
            ),
            cls.format_for_float(
                x=results['alpha_tstat'],
                excel_format=excel_output)
        ]
        for name in col_list:
            temp.append(
                cls.reg_result_formatting(
                    coefficient=results[name],
                    pvalue=results[name + '_pvalue'],
                    excel_output=excel_output
                )
            )

        temp.append(cls.format_for_float(
            x=results['adj_rsquared'],
            excel_format=excel_output)
        )
        return temp



    @classmethod
    def get_performance_formatted(
            cls,
            sra_obj: StrategyRelativeAnalytics,
            excel_output: bool = False) -> list:
        """
        Gives the formatted performance table values
        TODO: check COST and AUM relation
        Args:
            sra_obj:
            excel_output: Bool to flag excel (True) or pdf (False) output

        Returns:

        """
        SR = sra_obj.strategy.HistITDSharpe
        IR = sra_obj.current_IR
        cost = sra_obj.strategy.trading_cost["Cost"]
        vol = sra_obj.strategy.HistITDVol
        TE = sra_obj.current_hist_itd_tracking_error
        ER = sra_obj.current_hist_itd_excess_gross

        return ([
            sra_obj.name,
            cls.format_for_percent(
                x=sra_obj.strategy.current_nominal_itd_return,
                excel_format=excel_output),
            cls.format_for_percent(
                x=vol,
                excel_format=excel_output),
            cls.format_for_float(
                x=SR,
                excel_format=excel_output),
            cls.format_for_percent(
                x=ER,
                excel_format=excel_output),
            cls.format_for_percent(
                x=TE,
                excel_format=excel_output),
            cls.format_for_float(
                x=IR,
                excel_format=excel_output),
            cls.format_for_percent(
                x=cost,
                excel_format=excel_output),
            cls.format_for_float(
                x=(SR * vol - cost) / vol,
                excel_format=excel_output),
            cls.format_for_float(
                x=(ER - cost) / TE,
                excel_format=excel_output)
        ])

    @classmethod
    def create_analyses_lists(cls,
                              strategy_list,
                              relative_strategy_list,
                              region,
                              start_date,
                              end_date,
                              no_lag_date,
                              temp_folder,
                              rdp,
                              excel_output: bool,
                              excel_file_name: str
                              ):
        """
        Creates Pdf / excel output for all tables based on analyses switch and
        excel flag
        Args:
            strategy_list:
            relative_strategy_list:
            region:
            start_date:
            end_date:
            no_lag_date:
            temp_folder:
            rdp:
            excel_output:
            excel_file_name:
        Returns:

        """
        benchmark = strategy_list[0]
        no_of_relative_strategies = len(relative_strategy_list)
        no_of_strategies = len(strategy_list)

        if rdp.analyses_set['recent_portfoloio_characteristics'].switch:
            rdp.analyses_set['recent_portfoloio_characteristics'].values \
                = cls.get_recent_portfolio_chars_data(
                strategy_list=strategy_list)
            if excel_output:
                rdp.analyses_set[
                    'recent_portfoloio_characteristics'].excel_output = True
                rdp.analyses_set[
                    'recent_portfoloio_characteristics'].excel_values \
                    = cls.get_recent_portfolio_chars_data(
                    strategy_list=strategy_list,
                    excel_output=True
                )

        if rdp.analyses_set['hist_avg_portfoloio_characteristics'].switch:
            rdp.analyses_set['hist_avg_portfoloio_characteristics'].values \
                = cls.get_historical_portfolio_chars_data(
                rsa_list=relative_strategy_list)
            if excel_output:
                rdp.analyses_set[
                    'hist_avg_portfoloio_characteristics'].excel_output = True
                rdp.analyses_set[
                    'hist_avg_portfoloio_characteristics'].excel_values \
                    = cls.get_historical_portfolio_chars_data(
                    rsa_list=relative_strategy_list,
                    excel_output=excel_output
                )

        if rdp.analyses_set['excess_return_correlation'].switch:
            rdp.analyses_set['excess_return_correlation'].values \
                = cls.get_corr_2d_list(
                strategy_list=relative_strategy_list,
                col_name='Excess_Ret_over_benchmark'
            )
            if excel_output:
                rdp.analyses_set[
                    'excess_return_correlation'].excel_output = True
                rdp.analyses_set['excess_return_correlation'].excel_values \
                    = cls.get_corr_2d_list(
                    strategy_list=relative_strategy_list,
                    col_name='Excess_Ret_over_benchmark',
                    excel_output=excel_output
                )

        if rdp.analyses_set['tracking_error'].switch:
            rdp.analyses_set['tracking_error'].values \
                = cls.get_hist_tracking_error_2d_list(
                strategy_list=strategy_list,
                col_name=StrategyAnalyticsParameters.MONTHLY_RETURN_COL_NAME
            )
            if excel_output:
                rdp.analyses_set['tracking_error'].excel_output = True
                rdp.analyses_set['tracking_error'].excel_values \
                    = cls.get_hist_tracking_error_2d_list(
                    strategy_list=strategy_list,
                    col_name=StrategyAnalyticsParameters.MONTHLY_RETURN_COL_NAME,
                    excel_output=excel_output
                )

        if rdp.analyses_set[
            'year_on_year_strategy_minus_benchmark_returns'].switch:
            rdp.analyses_set['year_on_year_strategy_minus_benchmark_returns']. \
                values = cls.year_by_year_strategy_minus_benchmark_return(
                relative_strategy_list=relative_strategy_list,
                cum_monthly_return_column_name
                =StrategyAnalyticsParameters.CUMULATIVE_RETURN_COL_NAME,
                strategy_minus_benchmark_ret_col_name
                =rsa_params.
                    STRATEGY_MINUS_BENCHMARK_12M_RETURN_COL_NAME)
            if excel_output:
                rdp.analyses_set['year_on_year_strategy_minus_benchmark_returns'
                ].excel_output = True

                rdp.analyses_set['year_on_year_strategy_minus_benchmark_returns'
                ].excel_values \
                    = cls.year_by_year_strategy_minus_benchmark_return(
                    relative_strategy_list=relative_strategy_list,
                    cum_monthly_return_column_name
                    =StrategyAnalyticsParameters.CUMULATIVE_RETURN_COL_NAME,
                    strategy_minus_benchmark_ret_col_name
                    =rsa_params.STRATEGY_MINUS_BENCHMARK_12M_RETURN_COL_NAME,
                    excel_output=excel_output
                )

        if rdp.analyses_set['year_on_year_total_return'].switch:
            rdp.analyses_set['year_on_year_total_return'].values \
                = cls.year_by_year_return(
                strategy_list=strategy_list,
                cumprod_column
                =StrategyAnalyticsParameters.CUMULATIVE_RETURN_COL_NAME,
                ret_col_name_1yr='Historical1YrReturn')

            if excel_output:
                rdp.analyses_set[
                    'year_on_year_total_return'].excel_output = True
                rdp.analyses_set['year_on_year_total_return'].excel_values \
                    = cls.year_by_year_return(
                    strategy_list=strategy_list,
                    cumprod_column
                    =StrategyAnalyticsParameters.CUMULATIVE_RETURN_COL_NAME,
                    ret_col_name_1yr='Historical1YrReturn',
                    excel_output=excel_output
                )

        if rdp.analyses_set['hist_abs_rel_perf'].switch:
            rdp.analyses_set['hist_abs_rel_perf'].values.append(
                cls.get_performance_benchmark_formatted(strategy=benchmark))
            if excel_output:
                rdp.analyses_set['hist_abs_rel_perf'].excel_output = True
                rdp.analyses_set['hist_abs_rel_perf'].excel_values.append(
                    cls.get_performance_benchmark_formatted(
                        strategy=benchmark,
                        excel_output=excel_output))

        for strategy in strategy_list:
            if rdp.analyses_set['market_impact_cost'].switch:
                rdp.analyses_set['market_impact_cost'].values.append(
                    cls.get_trading_costs_formatted(strategy)
                )

            if rdp.analyses_set['attributions_ff3'].switch:
                rdp.analyses_set['attributions_ff3'].values.append(
                    cls.get_factor_loading_formatted(
                        df=strategy.ff3_plus_mom_base,
                        strategy=strategy,
                        col_list=['Mkt_RF', 'SMB', 'HML']
                    )
                )
                if excel_output:
                    rdp.analyses_set['attributions_ff3'].excel_output = True
                    rdp.analyses_set['attributions_ff3'].excel_values.append(
                        cls.get_factor_loading_formatted(
                            df=strategy.ff3_plus_mom_base,
                            strategy=strategy,
                            col_list=['Mkt_RF', 'SMB', 'HML'],
                            excel_output=excel_output
                        )
                    )

            if rdp.analyses_set['attributions_ff3_standardized'].switch:
                rdp.analyses_set['attributions_ff3_standardized'].values.append(
                    cls.get_factor_loading_formatted(
                        df=strategy.ff3_plus_mom_base,
                        strategy=strategy,
                        col_list=['Mkt_RF', 'SMB', 'HML'],
                        standardized_flag=True
                    )
                )
                if excel_output:
                    rdp.analyses_set['attributions_ff3_standardized'].excel_output = True
                    rdp.analyses_set['attributions_ff3_standardized'].excel_values.append(
                        cls.get_factor_loading_formatted(
                            df=strategy.ff3_plus_mom_base,
                            strategy=strategy,
                            col_list=['Mkt_RF', 'SMB', 'HML'],
                            excel_output=excel_output,
                            standardized_flag=True
                        )
                    )

            if rdp.analyses_set['attributions_ff5'].switch:
                rdp.analyses_set['attributions_ff5'].values.append(
                    cls.get_factor_loading_formatted(
                        df=strategy.ff5_plus_mom_base,
                        strategy=strategy,
                        col_list=['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
                    )
                )
                if excel_output:
                    rdp.analyses_set['attributions_ff5'].excel_output = True
                    rdp.analyses_set['attributions_ff5'].excel_values.append(
                        cls.get_factor_loading_formatted(
                            df=strategy.ff5_plus_mom_base,
                            strategy=strategy,
                            col_list=['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
                        )
                    )

                if rdp.analyses_set['attributions_ff5_standardized'].switch:
                    rdp.analyses_set['attributions_ff5_standardized'].values.append(
                        cls.get_factor_loading_formatted(
                            df=strategy.ff5_plus_mom_base,
                            strategy=strategy,
                            col_list=['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA'],
                            standardized_flag=True
                        )
                    )
                    if excel_output:
                        rdp.analyses_set['attributions_ff5_standardized'].excel_output = True
                        rdp.analyses_set['attributions_ff5_standardized'].excel_values.append(
                            cls.get_factor_loading_formatted(
                                df=strategy.ff5_plus_mom_base,
                                strategy=strategy,
                                col_list=['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA'],
                                excel_output=True,
                                standardized_flag=True
                            )
                        )

            if rdp.analyses_set['attributions_carhart4'].switch:
                rdp.analyses_set['attributions_carhart4'].values.append(
                    cls.get_factor_loading_formatted(
                        df=strategy.ff3_plus_mom_base,
                        strategy=strategy,
                        col_list=['Mkt_RF', 'SMB', 'HML', 'WML']

                    )
                )
                if excel_output:
                    rdp.analyses_set[
                        'attributions_carhart4'].excel_output = True
                    rdp.analyses_set['attributions_carhart4'].excel_values. \
                        append(
                        cls.get_factor_loading_formatted(
                            df=strategy.ff3_plus_mom_base,
                            strategy=strategy,
                            col_list=['Mkt_RF', 'SMB', 'HML', 'WML'],
                            excel_output=excel_output

                        )
                    )
            if rdp.analyses_set['attributions_carhart4_standardized'].switch:
                rdp.analyses_set['attributions_carhart4_standardized'].values.append(
                    cls.get_factor_loading_formatted(
                        df=strategy.ff3_plus_mom_base,
                        strategy=strategy,
                        col_list=['Mkt_RF', 'SMB', 'HML', 'WML'],
                        standardized_flag=True
                    )
                )
                if excel_output:
                    rdp.analyses_set[
                        'attributions_carhart4_standardized'].excel_output = True
                    rdp.analyses_set['attributions_carhart4_standardized'].excel_values. \
                        append(
                        cls.get_factor_loading_formatted(
                            df=strategy.ff3_plus_mom_base,
                            strategy=strategy,
                            col_list=['Mkt_RF', 'SMB', 'HML', 'WML'],
                            excel_output=excel_output,
                            standardized_flag=True
                        )
                    )


        for relative_strategy in relative_strategy_list:
            if rdp.analyses_set['hist_abs_rel_perf'].switch:
                rdp.analyses_set['hist_abs_rel_perf'].values.append(
                    cls.get_performance_formatted(relative_strategy)
                )
                if excel_output:
                    rdp.analyses_set['hist_abs_rel_perf'].excel_output = True
                    rdp.analyses_set['hist_abs_rel_perf'].excel_values.append(
                        cls.get_performance_formatted(
                            relative_strategy,
                            excel_output=excel_output
                        )
                    )

            if rdp.analyses_set['performance_at_yearly_horizon'].switch:
                rdp.analyses_set['performance_at_yearly_horizon'].values \
                    = rdp.analyses_set['performance_at_yearly_horizon'].values \
                      + cls.get_annualized_returns_nolag_concat_table_formatted(
                    relative_strategy)

                if excel_output:
                    rdp.analyses_set[
                        'performance_at_yearly_horizon'].excel_output = True
                    rdp.analyses_set['performance_at_yearly_horizon'] \
                        .excel_values = \
                        rdp.analyses_set[
                            'performance_at_yearly_horizon'].excel_values \
                        + cls.get_annualized_returns_nolag_concat_table_formatted(
                            sra_obj=relative_strategy,
                            excel_output=excel_output)

            if rdp.analyses_set['excess_return_decomposition'].switch:
                rdp.analyses_set['excess_return_decomposition'].values.append(
                    cls.get_output_excess_return_decomposition(
                        relative_strategy)
                )

                if excel_output:
                    rdp.analyses_set[
                        'excess_return_decomposition'].excel_output = True
                    rdp.analyses_set[
                        'excess_return_decomposition'].excel_values.append(
                        cls.get_output_excess_return_decomposition(
                            relative_strategy,
                            excel_output=excel_output)
                    )

            if rdp.analyses_set['expected_excess_return_table'].switch:
                rdp.analyses_set['expected_excess_return_table'].values.append(
                    cls.get_output_comparison_formatted(relative_strategy)
                )
                if excel_output:
                    rdp.analyses_set[
                        'expected_excess_return_table'].excel_output = True
                    rdp.analyses_set[
                        'expected_excess_return_table'].excel_values.append(
                        cls.get_output_comparison_formatted(
                            relative_strategy,
                            excel_output=excel_output
                        )
                    )

            if rdp.analyses_set['moments_downside_risks'].switch:
                rdp.analyses_set['moments_downside_risks'].values.append(
                    cls.get_mom_down_side_risk(sra_obj=relative_strategy)
                )
                if excel_output:
                    rdp.analyses_set[
                        'moments_downside_risks'].excel_output = True
                    rdp.analyses_set['moments_downside_risks'].excel_values \
                        .append(
                        cls.get_mom_down_side_risk(
                            sra_obj=relative_strategy,
                            excel_output=excel_output
                        )
                    )

            if rdp.analyses_set['worst_period_var_winrate'].switch:
                rdp.analyses_set['worst_period_var_winrate'].values.append(
                    cls.get_var_win_rate(sra_obj=relative_strategy)
                )
                if excel_output:
                    rdp.analyses_set['worst_period_var_winrate'].excel_values \
                        .append(
                        cls.get_var_win_rate(
                            sra_obj=relative_strategy,
                            excel_output=excel_output
                        )
                    )
        # ----------------------------------------------------------------------
        #  Dynamic formatting lists
        if rdp.analyses_set[
            'year_on_year_strategy_minus_benchmark_returns'].switch:
            # Dynamic table formatting based on data and number of strategies
            # Strategy minus Benchmark coloring
            color_style_temp = []
            year_by_year_excess_returns_list = rdp.analyses_set[
                'year_on_year_strategy_minus_benchmark_returns'].values
            for i in range(1, len(year_by_year_excess_returns_list)):
                for j in range(1, len(year_by_year_excess_returns_list[i])):
                    temp_str = year_by_year_excess_returns_list[i][j]
                    temp_val = float(temp_str[:temp_str.find('%')])
                    if temp_val < -10:
                        color_style_temp.append(
                            ('BACKGROUND', (j, i), (j, i), red)
                        )
                    elif temp_val > 10:
                        color_style_temp.append(
                            ('BACKGROUND', (j, i), (j, i), green)
                        )
                    else:
                        continue

            rdp.analyses_set['year_on_year_strategy_minus_benchmark_returns']. \
                add_tbl_styles = rdp.analyses_set[
                                     'year_on_year_strategy_minus_benchmark_returns']. \
                                     add_tbl_styles + color_style_temp

        if rdp.analyses_set['performance_at_yearly_horizon'].switch:
            # Alternate Rows in performance horizons table
            temp_list = [
                ('SPAN', (0, 0), (1, 0)),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black)]
            for i in range(no_of_relative_strategies):
                temp_list.append(('SPAN', (0, int(i * 2 + 1)), (0, i * 2 + 2)))
                temp_list.append(('SPAN', (6, int(i * 2 + 1)), (6, i * 2 + 2)))
                temp_list.append(('SPAN', (7, int(i * 2 + 1)), (7, i * 2 + 2)))

            rdp.analyses_set['performance_at_yearly_horizon'].add_tbl_styles \
                = temp_list

        if rdp.analyses_set['excess_return_correlation'].switch:
            rdp.analyses_set['excess_return_correlation'].col_widths \
                = [2.5 * inch] + ([(7.5 / no_of_relative_strategies) * inch]
                                  * no_of_relative_strategies)

        if rdp.analyses_set['tracking_error'].switch:
            rdp.analyses_set['tracking_error'].col_widths = \
                [2.5 * inch] + ([(7.5 / no_of_strategies) * inch]
                                * no_of_strategies)

        # ----------------------------------------------------------------------
        # Write the table analyses pdf file
        analyses_list = list(rdp.analyses_set.keys())

        for i in range(len(analyses_list)):
            key = analyses_list[i]
            analysis = rdp.analyses_set[key]
            if rdp.analyses_set[key].is_table and rdp.analyses_set[key].switch:
                Report.table_to_pdf(
                    data_2d_list=analysis.values,
                    heading_text=analysis.title_text
                        .replace('%region', region)
                        .replace('%start_date', start_date.strftime("%b %Y"))
                        .replace('%end_date', end_date.strftime("%b %Y"))
                        .replace('%nolagdate', no_lag_date.strftime("%b %Y"))
                        .replace('%benchmark', benchmark.name),
                    output_location_filename=temp_folder + analysis.output_name,
                    footer_text=analysis.footer_text
                        .replace('%nolagdate', no_lag_date.strftime("%b %Y")),
                    add_tbl_styles=analysis.add_tbl_styles,
                    col_widths=analysis.col_widths,
                    number_of_rows_to_repeat_next_page
                    =analysis.number_of_rows_to_repeat_next_page,
                    alternate_background_rows
                    =analysis.alternate_background_rows,
                    data_align=analysis.data_align
                )
        # ----------------------------------------------------------------------
        # Write the table analyses excel file
        if excel_output:
            writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')

            for i in range(len(analyses_list)):
                key = analyses_list[i]
                analysis = rdp.analyses_set[key]
                if analysis.excel_output:
                    sheet_name = analysis.name
                    if len(sheet_name) > 31:
                        sheet_name = analysis.name[:31]
                    df = pd.DataFrame(analysis.excel_values)
                    df.to_excel(excel_writer=writer,
                                sheet_name=sheet_name,
                                header=False,
                                index=False)
            i = 1
            for rsa in relative_strategy_list:
                rsa.all_data.to_excel(excel_writer=writer,
                                      sheet_name='strategy_' + str(i))
                i += 1

            writer.save()

    @classmethod
    def create_output_chart_pdfs(cls,
                                 strategy_list: list,
                                 relative_strategy_list: list,
                                 temp_folder: str,
                                 rdp: RavisualDataParameters):
        """
        Creates Pdf output for all charts based on analyses switch flag
        Args:
            strategy_list:
            relative_strategy_list:
            temp_folder:
            rdp:

        Returns:

        """
        benchmark = strategy_list[0]
        for i in range(1, len(strategy_list)):
            strategy = strategy_list[i]
            cls.create_historical_allocation_pdf(
                sa_obj=strategy,
                benchmark_obj=benchmark,
                type='Country',
                title_text=ReportParameters.COUNTRY_ALLOCATION_CHART_SUP_TITLE,
                output_filename=(str(temp_folder) + rdp.analyses_set[
                    'historical_allocation_country'].output_name +
                                 str(i) + ".pdf")
            )

            cls.create_historical_allocation_pdf(
                sa_obj=strategy,
                benchmark_obj=benchmark,
                type='Sector',
                title_text=ReportParameters.COUNTRY_ALLOCATION_CHART_SUP_TITLE,
                output_filename=(str(temp_folder) + rdp.analyses_set[
                    'historical_allocation_sector'].output_name +
                                 str(i) + ".pdf")
            )

            cls.create_historical_allocation_pdf(
                sa_obj=strategy,
                benchmark_obj=benchmark,
                type='Region',
                title_text=ReportParameters.COUNTRY_ALLOCATION_CHART_SUP_TITLE,
                output_filename=(str(temp_folder) + rdp.analyses_set[
                    'historical_allocation_region'].output_name +
                                 str(i) + ".pdf")
            )

        # Recent allocation Bar charts
        if rdp.analyses_set['recent_allocation_sector'].switch:
            cls.recent_allocation_bar_chart(
                strategy_list=strategy_list,
                output_filename=temp_folder + rdp.analyses_set[
                    'recent_allocation_sector'].output_name,
                chart_type='Sector')

        if rdp.analyses_set['recent_allocation_region'].switch:
            cls.recent_allocation_bar_chart(
                strategy_list=strategy_list,
                output_filename=temp_folder + rdp.analyses_set[
                    'recent_allocation_region'].output_name,
                chart_type='Region')

        if rdp.analyses_set['recent_allocation_country'].switch:
            cls.recent_allocation_bar_chart(
                strategy_list=strategy_list,
                output_filename=temp_folder + rdp.analyses_set[
                    'recent_allocation_country'].output_name,
                chart_type='Country')

        # Forecast Excess Return charts
        for i in range(len(relative_strategy_list)):
            if rdp.analyses_set['expected_excess_return_charts'].switch:
                cls.create_model_comparison_pdf(
                    relative_strategy_list[i],
                    str(temp_folder) + rdp.analyses_set[
                        'expected_excess_return_charts'].output_name + str(i) +
                    ".pdf")

        # Concentration Charts

        if rdp.analyses_set['concentration_holdings'].switch:
            cls.create_concentration_pdf(
                strategy_list=strategy_list,
                output_filename=temp_folder + rdp.analyses_set[
                    'concentration_holdings'].output_name,
                col_name_1='n',
                col_name_2='effn',
                title_1='Number of Holdings',
                title_2='Number of Effective Holdings',
                sup_title_text="Concentration: Effective number of Holdings",
                format_to_percent=False
            )

        if rdp.analyses_set['concentration_weight'].switch:
            cls.create_concentration_pdf(
                strategy_list=strategy_list,
                output_filename=temp_folder + rdp.analyses_set[
                    'concentration_weight'].output_name,
                col_name_1='sumwgttop5',
                col_name_2='sumwgttop10',
                title_1='Weight of Top 5 Holdings',
                title_2='Weight of Top 10 Holdings',
                sup_title_text="Concentration: Weight of Top Holdings",
                format_to_percent=True
            )

        # SR IR Net of Cost Chart
        if rdp.analyses_set['net_of_cost_SR_IR'].switch:
            cls.create_SR_IR_chart(
                relative_strategy_list=relative_strategy_list,
                output_filename=temp_folder + rdp.analyses_set[
                    'net_of_cost_SR_IR'].output_name
            )

    @classmethod
    def create_output(
            cls,
            relative_strategy_list: list,
            strategy_list: list,
            region: str,
            start_date: pd.datetime,
            end_date: pd.datetime,
            temp_folder: str,
            output_location: str,
            output_filename: str,
            no_lag_date: pd.datetime,
            switch_dict: dict,
            excel_filename: str,
            excel_output: bool = False

    ) -> None:
        """
        Driver function which dreates final report after merging all the analyses and charts
        Args:
            relative_strategy_list:
            strategy_list:
            benchmark_name:
            region:
            start_date:
            end_date:
            temp_folder:
            output_location:
            output_filename:
            no_lag_date:
            switch_dict:
            excel_output:
            excel_filename:

        Returns:

        """
        rdp = RavisualDataParameters()

        # Switch on/OFF analyses
        for analysis in list(switch_dict.keys()):
            rdp.analyses_set[analysis].switch = switch_dict[analysis]

        # ----------------------------------------------------------------------
        # Temporary Solution for fixing Date
        end_date = end_date - pd.Timedelta(1, unit='D')
        no_lag_date = no_lag_date - pd.Timedelta(1, unit='D')
        # ----------------------------------------------------------------------
        number_of_strategies = len(strategy_list)
        number_of_relative_strategies = len(relative_strategy_list)
        # ----------------------------------------------------------------------
        cls.create_analyses_lists(
            strategy_list=strategy_list,
            relative_strategy_list=relative_strategy_list,
            temp_folder=temp_folder,
            region=region,
            start_date=start_date,
            end_date=end_date,
            no_lag_date=no_lag_date,
            rdp=rdp,
            excel_output=excel_output,
            excel_file_name=output_location + '\\' + excel_filename
        )

        cls.create_output_chart_pdfs(
            strategy_list=strategy_list,
            relative_strategy_list=relative_strategy_list,
            temp_folder=temp_folder,
            rdp=rdp
        )

        # Disclosures PDf
        Report.create_disclosures_pdf(temp_folder + "//disclosures.pdf")

        # ----------------------------------------------------------------------
        # Create Table of Contents
        filename_list = []
        content_name_dict = {}
        page_num_list = []
        page_num = 1
        for analysis_name in rdp.ANALYSES_ORDER:
            analysis = rdp.analyses_set[analysis_name]
            if analysis.switch:
                analysis.index_page_num = page_num
                page_num_list.append(page_num)
                filename_list = filename_list + glob.glob(
                    temp_folder + analysis.output_name + '*')

                if analysis.index_section_name in content_name_dict.keys():
                    content_name_dict[analysis.index_section_name].append(
                        analysis.index_name
                    )
                else:
                    content_name_dict[analysis.index_section_name] \
                        = [analysis.index_name]

                if analysis.separate_startegy_plots:
                    page_num += number_of_relative_strategies
                else:
                    page_num += Report.get_number_of_pages_pdf(filename_list[-1])

        Report.create_toc_pdf(output_file_name=temp_folder + "//toc.pdf",
                              title_text='Table of Contents',
                              content_name_dict=content_name_dict,
                              page_number_list=page_num_list
                              )
        # ----------------------------------------------------------------------
        # merge pdf
        Report.merge_pdf_files(
            filename_list=filename_list,
            output_location_filename=output_location + '\\' + output_filename
        )

        Report.merge_pdf_files(
            filename_list=[
                temp_folder + "//user_config.pdf",
                temp_folder + "//toc.pdf",
                output_location + '\\' + output_filename,
                temp_folder + "//disclosures.pdf"
            ],
            output_location_filename=output_location + '\\' + output_filename,
            page_number_flag=False
        )

    @staticmethod
    def del_folder(temp_folder: str) -> None:
        """
        Deletes the folder
        Args:
            temp_folder: the name of the folder to be deleted

        Returns: None

        """
        # clean files
        rm_list = glob.glob(temp_folder + "//*")
        for i in range(len(rm_list)):
            os.remove(rm_list[i])

        os.rmdir(temp_folder)

    @staticmethod
    def create_title_page(strategy_df,
                          benchmark_name,
                          benchmark_filename,
                          region,
                          start_date,
                          last_date,
                          output_filename,
                          rf_filename,
                          ff3_mom_filename,
                          ff5_mom_filename
                          ):
        """

        Args:
            strategy_df:
            benchmark_name:
            benchmark_filename:
            region:
            start_date:
            last_date:
            output_filename:
            rf_filename:
            ff3_mom_filename:
            ff5_mom_filename:

        Returns:

        """

        a = datetime.datetime.today()
        h1 = PS(
            name='Heading 1',
            fontSize=14,
            spaceAfter=8

        )

        b1_indented = PS(
            name='Body 1 indented',
            fontSize=12,
            leftIndent=20,
            spaceBefore=4

        )

        b1 = PS(
            name='Body 1',
            fontSize=12,
            spaceBefore=6
        )

        b2 = PS(
            name='Body 2',
            fontSize=8,
            spaceBefore=4,
            leftIndent=40

        )

        elements = []
        elements.append(Paragraph(
            '<b>Simulations of Systematic Equity Strategies</b>\n'.replace(
                '\n','<br />\n'), style=h1))

        elements.append(Paragraph('Report created on: ' + a.strftime(
            "%Y-%m-%d %H:%M") + '<br /><br />', style=b1))

        elements.append(Paragraph(
            '<i><b>Region: </b></i>' + region, style=b1))
        elements.append(Paragraph(
            '<br /><i><b>Benchmark:</b></i> <br />', style=b1))

        elements.append(Paragraph(benchmark_name + '<br />', style=b1_indented))

        elements.append(Paragraph(
            benchmark_filename + '<br />', style=b2))

        elements.append(Paragraph(
            '<i><b>Strategies:</b></i> <br />', style=b1))

        for i in range(len(strategy_df)):
            strategy_name = strategy_df.loc[i, 'Value1']
            strategy_filename = strategy_df.loc[i, 'Value2']
            elements.append(Paragraph(
                strategy_name + '<br />',
                style=b1_indented))
            elements.append(Paragraph(
                strategy_filename + '<br />',
                style=b2))

        elements.append(Paragraph(
            '<i><b>Time Period (Overlapping): </b></i>' + start_date.strftime(
                "%b %Y") + ' - ' + last_date.strftime("%b %Y"), style=b1))

        elements.append(Paragraph(
            '<br /><i><b>FamaFrench 3, Carhart Factor 4 dataset:</b></i> <br />',
            style=b1))
        elements.append(Paragraph(
            ff3_mom_filename, style=b2))

        elements.append(Paragraph(
            '<br /><i><b>FamaFrench 5:</b></i> <br />', style=b1))
        elements.append(Paragraph(
            ff5_mom_filename, style=b2))

        elements.append(Paragraph(
            '<br /><i><b>Risk Free Rate dataset: </b></i><br />', style=b1))
        elements.append(Paragraph(rf_filename, style=b2))

        doc = SimpleDocTemplate(output_filename,
                                pagesize=landscape(letter), rightMargin=inch,
                                leftMargin=inch,
                                topMargin=inch, bottomMargin=inch)

        doc.build(elements)

    @staticmethod
    def create_model_comparison_pdf(rsa_obj: StrategyRelativeAnalytics,
                                    output_filename: str
                                    ):
        """

        Args:
            rsa_obj:
            output_filename:

        Returns:

        """

        top_chart_x = rsa_obj.all_data['date' + '_characteristics']

        top_chart_y1 = rsa_obj.all_data[rsa_params.CUM_PROD_EXCESS_COL_NAME]

        top_chart_y2 = rsa_obj.all_data[rsa_params.REL_MEASURE_COL_NAME]
        max_top_y2 = max(top_chart_y2)

        bottom_chart_yscatter = \
            rsa_obj.all_data[rsa_params.SUBSEQUENT_EXCESS_RET_COL_NAME]

        bottom_chart_x = rsa_obj.all_data[rsa_params.REL_MEASURE_COL_NAME]

        bottom_chart_ycurve = rsa_obj.all_data[
            rsa_params.EXPECTED_EXCESS_RETURN_COL_NAME]
        bottom_chart_point_x = rsa_obj.current_rel_val_agg
        bottom_chart_point_y = rsa_obj.current_expected_excess_return
        bottom_chart_vline = rsa_obj.current_rel_val_agg_50th
        strategy_name = rsa_obj.strategy.name
        strategy_location = rsa_obj.strategy.location
        benchmark_name = rsa_obj.benchmark.name
        benchmark_location = rsa_obj.benchmark.location
        no_of_years = rsa_obj.number_of_subsequent_years

        # Update the date variables
        start_date = min(
            rsa_obj.all_data['date_characteristics'].copy().dropna())
        end_returns_date = max(
            rsa_obj.all_data['date_returns'].copy().dropna())

        # fit figure into relative_analytics US letter size
        fig = plt.figure(figsize=(11, 8.5))

        ax1 = fig.add_subplot(ReportParameters.COMPARISON_SUBPLOT1_POSITION)
        ax1.set_xlabel(ReportParameters.COMPARISON_SUBPLOT1_X_LABEL)
        ax1.set_ylabel(ReportParameters.COMPARISON_SUBPLOT1_Y_LABEL_LINE1,
                       color=ReportParameters.COMPARISON_SUBPLOT1_COLOR1)
        ax1.grid(alpha=0.5, color=ReportParameters.COMPARISON_SUBPLOT1_COLOR1)

        ax1.plot(top_chart_x,
                 top_chart_y1,
                 color=ReportParameters.COMPARISON_SUBPLOT1_COLOR1,
                 label=ReportParameters.COMPARISON_SUBPLOT1_Y_LABEL_LINE1)
        ax1.tick_params(axis='y',
                        labelcolor=ReportParameters.COMPARISON_SUBPLOT1_COLOR1)
        lower1, upper1 = ax1.get_ylim()
        new_upper1 = max(upper1, 2.0,
                         max_top_y2 * 1.05 * (1 / (top_chart_y2[0])))
        ax1.set_ylim(0, new_upper1)
        ax1.plot(top_chart_x,
                 top_chart_y1,
                 color=ReportParameters.COMPARISON_SUBPLOT1_COLOR1,
                 label=ReportParameters.COMPARISON_SUBPLOT1_Y_LABEL_LINE1)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.2f}'.format(j)))

        # instantiate relative_analytics second axes that shares the same x-axis

        ax1_b = ax1.twinx()

        # we already handled the x-label with ax1
        ax1_b.set_ylabel(ReportParameters.COMPARISON_SUBPLOT1_Y_LABEL_LINE2,
                         color=ReportParameters.COMPARISON_SUBPLOT1_COLOR2)
        ax1_b.plot(top_chart_x,
                   top_chart_y2,
                   color=ReportParameters.COMPARISON_SUBPLOT1_COLOR2,
                   label=ReportParameters.COMPARISON_SUBPLOT1_Y_LABEL_LINE2)

        new_upper2 = (top_chart_y2[0]) * new_upper1

        ax1_b.set_ylim(0, new_upper2)

        ax1_b.tick_params(axis='y',
                          labelcolor=ReportParameters.COMPARISON_SUBPLOT1_COLOR2)

        ax2 = fig.add_subplot(ReportParameters.COMPARISON_SUBPLOT2_POSITION)
        # ax2.set_title(ReportParameters.COMPARISON_SUBPLOT2_TITLE)
        ax2.set_xlabel(ReportParameters.COMPARISON_SUBPLOT2_TITLE_X_LABEL)

        ax2.set_ylabel(
            str(ReportParameters.COMPARISON_SUBPLOT2_TITLE_Y_LABEL).replace(
                "%d", str(no_of_years)))
        ax2.scatter(
            x=bottom_chart_x,
            y=bottom_chart_yscatter,
            c=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_COLOR,
            alpha=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_TRANSPARENCY,
            marker=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_MARKER,
            label=str(
                ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_ACTUAL_RET_LABEL).replace(
                "%d", str(no_of_years)))

        ax2.plot(bottom_chart_x,
                 bottom_chart_ycurve,
                 color=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_COLOR,
                 alpha=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_TRANSPARENCY,
                 # marker=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_MARKER,
                 linewidth=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_LINEWIDTH,
                 label=str(
                     ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_PRED_RET_LABEL).replace(
                     "%d", str(no_of_years))
                 )

        ax2.scatter(x=bottom_chart_point_x,
                    y=bottom_chart_point_y,
                    c=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_COLOR,
                    alpha=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_TRANSPARENCY,
                    marker=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_MARKER,
                    linewidths=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_LINEWIDTH,
                    label=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_CURRENT_VAL_LABEL
                    )

        ax2.axvline(x=float(bottom_chart_vline),
                    c=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_COLOR,
                    alpha=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_TRANSPARENCY,
                    marker=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_MARKER,
                    label=ReportParameters.COMPARISON_SUBPLOT2_SCATTER_PLOT_MEDIAN_VAL_LABEL
                    )
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.2f}'.format(j)))

        ax2.grid(alpha=0.5)
        plt.legend()
        ax2.legend(loc=ReportParameters.COMPARISON_SUBPLOT2_LEGEND_LOC,
                   fontsize=ReportParameters.COMPARISON_SUBPLOT2_LEGEND_FONTSIZE,
                   fancybox=ReportParameters.COMPARISON_SUBPLOT2_LEGEND_FANCYBOX,
                   shadow=ReportParameters.COMPARISON_SUBPLOT2_LEGEND_SHADOW
                   )

        # Adjust the size of the figure to fit in the text
        plt.subplots_adjust(bottom=0.25,
                            top=0.8,
                            left=0.2,
                            right=0.8,
                            hspace=0.2)

        ax2.text(x=0,
                 y=-0.4,
                 s=ReportParameters.COMPARISON_CHART_FOOTER_TXT,
                 size=ReportParameters.COMPARISON_CHART_FOOTER_FONTSIZE,
                 transform=ax2.transAxes)

        fig.suptitle(
            ReportParameters.COMPARISON_CHART_SUP_TITLE.replace(
                "%strategyname", str(strategy_name).upper()).replace(
                "%strategylocation", str(strategy_location).upper()).replace(
                "%benchmarkname", str(benchmark_name).upper()).replace(
                "%benchmarklocation", str(benchmark_location).upper()).replace(
                "%startdate", start_date.strftime("%b %Y")).replace(
                "%enddate", end_returns_date.strftime("%b %Y")),
            fontsize=ReportParameters.COMPARISON_CHART_TITLE_FONTSIZE
            ,
            y=ReportParameters.COMPARISON_CHART_TITLE_Y_LOCATION,
            x=ReportParameters.COMPARISON_CHART_TITLE_X_LOCATION,
            horizontalalignment='left'
        )

        plt.close()
        fig.savefig(output_filename, format='pdf')

    @staticmethod
    def create_historical_allocation_pdf(
            sa_obj: StrategyAnalytics,
            benchmark_obj: StrategyAnalytics,
            type: str,
            output_filename: str,
            title_text: str,
            bottom: float=0.2,
            top: float=0.8
    ):
        """

        Args:
            sa_obj:
            benchmark_obj:
            type:
            output_filename:
            title_text:
            bottom:
            top:

        Returns:

        """
        # Allocate relevant Dataframe based on type
        if type == 'Region':
            strategy_allocation_df = sa_obj.region_allocation_df
            benchmark_allocation_df = benchmark_obj.region_allocation_df
        elif type == 'Country':
            strategy_allocation_df = sa_obj.country_allocation_df
            benchmark_allocation_df = benchmark_obj.country_allocation_df
        elif type == 'Sector':
            strategy_allocation_df = sa_obj.sector_allocation_df
            benchmark_allocation_df = benchmark_obj.sector_allocation_df
        else:
            strategy_allocation_df = sa_obj.sector_allocation_df
            benchmark_allocation_df = benchmark_obj.sector_allocation_df

        # columns not in benchmark but in strategy should appear in the
        # legend
        diff_cols_strategy_minus_benchmark = list(
            set(list(strategy_allocation_df.columns)) - set(
                benchmark_allocation_df.columns))

        diff_cols_benchmark_minus_strategy = list(
            set(list(benchmark_allocation_df.columns)) - set(
                strategy_allocation_df.columns))

        for new_col in diff_cols_strategy_minus_benchmark:
            benchmark_allocation_df[new_col] = 0
        for new_col in diff_cols_benchmark_minus_strategy:
            strategy_allocation_df[new_col] = 0

        new_column_list = sorted(list(benchmark_allocation_df.columns))
        benchmark_allocation_df = benchmark_allocation_df[
            new_column_list].copy()
        strategy_allocation_df = strategy_allocation_df[
            new_column_list].copy()

        # Making generic for any number of columns strategy
        temp_list = []
        strategy_color_list = []
        for i in range(len(strategy_allocation_df.columns)):
            col_name = strategy_allocation_df.columns[i]
            temp_list.append(strategy_allocation_df[col_name])
            if col_name.lower() in list(ReportParameters.COLOR_DICT.keys()):
                strategy_color_list.append(
                    ReportParameters.COLOR_DICT[col_name.lower()])
            else:
                strategy_color_list.append('')

        strategy_y = np.vstack(temp_list)
        strategy_x = strategy_allocation_df.index

        # columns not in benchmark add in benchmark
        # Making generic for any number of columns benchmark
        temp_list = []
        benchmark_color_list = []
        for i in range(len(benchmark_allocation_df.columns)):
            col_name = benchmark_allocation_df.columns[i]
            temp_list.append(benchmark_allocation_df[col_name])
            if col_name.lower() in list(ReportParameters.COLOR_DICT.keys()):
                benchmark_color_list.append(
                    ReportParameters.COLOR_DICT[col_name.lower()])
            else:
                benchmark_color_list.append('')

        benchmark_y = np.vstack(temp_list)
        benchmark_x = benchmark_allocation_df.index

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(122)
        ax.stackplot(strategy_x, strategy_y,
                     labels=list(strategy_allocation_df.columns),
                     colors=strategy_color_list)
        ax.set_ylim(bottom=0, top=1)
        # Format y axis to be %
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.0%}'.format(j)))

        # Shrink current width by 30% to fit in the legend on the right
        ax.set_title(sa_obj.name + ',' + sa_obj.location)

        # Put a legend to the right of the current axis and reverse the order
        # so that it matches the order of colors of the area map from top to
        # bottom
        ax2 = fig.add_subplot(121)
        ax2.stackplot(benchmark_x,
                      benchmark_y,
                      labels=list(benchmark_allocation_df.columns),
                      colors=benchmark_color_list)
        ax2.set_ylim(bottom=0, top=1)
        # Format y axis to be %
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda j, _: '{:.0%}'.format(j)))

        ax2.set_title(
            'Benchmark:' + benchmark_obj.name + ',' + benchmark_obj.location)

        plt.subplots_adjust(bottom=bottom,
                            top=top,
                            hspace=0.15,
                            wspace=0.15)

        fig.suptitle(
            title_text.replace(
                "%strategyname", str(sa_obj.name).upper()).replace(
                "%strategylocation", str(sa_obj.location).upper()).replace(
                "%startdate",
                min(strategy_allocation_df.index).strftime("%b %Y")).replace(
                "%enddate",
                max(strategy_allocation_df.index).strftime("%b %Y")),
            fontsize=ReportParameters.COMPARISON_CHART_TITLE_FONTSIZE,
            y=ReportParameters.COMPARISON_CHART_TITLE_Y_LOCATION,
            x=ReportParameters.COMPARISON_CHART_TITLE_X_LOCATION,
            horizontalalignment='left'
        )

        handles, labels = ax2.get_legend_handles_labels()
        box = ax.get_position()
        ax.set_position(
            [box.x0 - box.width * 0.1, box.y0, box.width * 0.95, box.height])

        box2 = ax2.get_position()
        ax2.set_position([box2.x0 - box.width * 0.05,
                          box2.y0,
                          box2.width * 0.95,
                          box2.height])

        ax2.legend(
            handles=handles[::-1],
            labels=labels[::-1],
            loc='upper center',
            bbox_to_anchor=(2.37, 1),
            fontsize='small'
        )

        plt.close()
        fig.savefig(output_filename, format='pdf')

    @staticmethod
    def get_hist_annualized_tracking_error(
            strategy_return_series: pd.Series,
            benchmark_return_series: pd.Series
    ):
        """
        give annualized historical tracking error between two pd.Series
        Args:
            strategy_return_series:
            benchmark_return_series:

        Returns:

        """
        return np.std(
                a=(strategy_return_series - benchmark_return_series),
                ddof=1
            ) * (12 ** 0.5)


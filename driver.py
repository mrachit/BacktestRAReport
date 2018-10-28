import sys
import pandas as pd
from ras.core.reporting.strategy_relative_analytics import StrategyRelativeAnalytics
from ras.core.reporting.strategy_analytics import StrategyAnalytics
import time
from ras.core.reporting.ravisual_report import ProdResearchReport as prr
from ras.core.reporting.user_input_parameter import UserInputParameters
import os


def get_overlapping_dates(benchmark_filename: str,
                          strategy_df: pd.DataFrame,
                          start_date: pd.datetime,
                          end_date: pd.datetime):
    """
    Screen through all the input dates and a get overlapping start date &
    end date between the user input start and end date, benchmark date and all
    the strategy dates. The no lag date is the last date till which we have
    the contemporaneous data in all the strategies and benchmark.
    Args:
        benchmark_filename: excel filename  for benchmark
        strategy_df: Dataframe containing the list of strategies and their
            file names
        start_date:user input analysis start date
        end_date: user input analysis end date

    Returns:
        start_date : pd.datetime
            Overlapping  (input, benchmark & strategies) start date
        end_date: pd.datetime
            Overlapping (input, benchmark & strategies) end date
        no lag date: pd.datetime
            Overlapping (input, benchmark & strategies) last date for
            contemporaneous data
    """
    min_date, max_date, nolag_date = StrategyAnalytics.get_min_max_nolag_date(
        filename=benchmark_filename)
    if min_date > start_date:
        start_date = min_date
    # last date check
    if max_date < end_date:
        end_date = max_date

    for i in range(0, len(strategy_df)):
        min_date, max_date, nolag_date_s = StrategyAnalytics. \
            get_min_max_nolag_date(filename=strategy_df.loc[i, 'Value2'])
        if min_date > start_date:
            start_date = min_date
        # last date check
        if max_date < end_date:
            end_date = max_date
        if nolag_date_s < nolag_date:
            nolag_date = nolag_date_s
    return start_date, end_date, nolag_date


def ravisual_driver_function(
        input_filename: str = None):
    """
    Requires a input csv file for user inputs for the run
    Args:
        input_filename:input csv file for user inputs for the code run.
        Creates the analysis output based on the user input

    Returns:

    """
    input_params = UserInputParameters(input_filename=input_filename)
    strategy_df = input_params.strategy_df
    switch_dict = input_params.switch_dict
    output_location = input_params.output_location
    output_filename = input_params.output_filename
    name_benchmark = input_params.name_benchmark
    benchmark_filename = input_params.benchmark_filename
    comparison_region = input_params.comparison_region
    start_date = input_params.start_date
    end_date = input_params.end_date
    excel_output = input_params.excel_output
    excel_filename = input_params.excel_filename
    rf_filename = input_params.rf_filename
    ff3_mom_filename = input_params.ff3_plus_mom_sas_filename
    ff5_mom_filename = input_params.ff5_plus_mom_sas_filename
    # --------------------------------------------------------------------------
    # Create temp directory required for analysis

    temp_folder = output_location + "\\temp" + str(time.time())
    os.mkdir(temp_folder)

    # --------------------------------------------------------------------------
    # get analyses for each strategy to be compared
    # check date filter

    start_date, end_date, nolag_date = get_overlapping_dates(
        benchmark_filename=benchmark_filename,
        strategy_df=strategy_df,
        start_date=start_date,
        end_date=end_date,
        )

    # Creating User config : Title pdf
    prr.create_title_page(
        strategy_df=strategy_df,
        benchmark_name=name_benchmark,
        benchmark_filename=benchmark_filename,
        region=comparison_region,
        start_date=start_date,
        last_date=end_date-pd.Timedelta(1, unit='D'),
        output_filename=temp_folder + "//user_config.pdf",
        rf_filename=rf_filename,
        ff5_mom_filename=ff5_mom_filename,
        ff3_mom_filename=ff3_mom_filename
        )

    # Creating Benchmark
    benchmark = StrategyAnalytics(name=name_benchmark,
                                  location=comparison_region,
                                  filename=benchmark_filename,
                                  start_date=start_date,
                                  last_date=end_date,
                                  nolag_date=nolag_date,
                                  rf_filename=rf_filename,
                                  ff3_mom_filename=ff3_mom_filename,
                                  ff5_mom_filename=ff5_mom_filename,

                                  )

    # --------------------------------------------------------------------------
    # get analyses for each strategy to be compared
    relative_strategy_list = []
    strategy_list = [benchmark]

    for i in range(0, len(strategy_df)):
        compared_strategy = \
            StrategyAnalytics(name=strategy_df.loc[i, 'Value1'],
                              location=comparison_region,
                              filename=strategy_df.loc[i, 'Value2'],
                              start_date=start_date,
                              last_date=end_date,
                              nolag_date=nolag_date,
                              rf_filename=rf_filename,
                              ff3_mom_filename=ff3_mom_filename,
                              ff5_mom_filename=ff5_mom_filename
                              )
        relative_analytics = StrategyRelativeAnalytics(
            strategy=compared_strategy,
            benchmark=benchmark,
            nolag_date=nolag_date)

        relative_strategy_list.append(relative_analytics)
        strategy_list.append(compared_strategy)

        # relative_analytics.all_data.to_csv(output_location +
        #                                  '//'+relative_analytics.name +'.csv')

    # --------------------------------------------------------------------------
    # Create output
    prr.create_output(
        relative_strategy_list=relative_strategy_list,
        strategy_list=strategy_list,
        region=comparison_region,
        start_date=start_date,
        end_date=end_date,
        temp_folder=temp_folder,
        output_location=output_location,
        output_filename=output_filename,
        no_lag_date=nolag_date,
        switch_dict=switch_dict,
        excel_output=excel_output,
        excel_filename=excel_filename
        )
    # --------------------------------------------------------------------------
    # Delete temp folder
    prr.del_folder(temp_folder=temp_folder)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ravisual_driver_function(sys.argv[1])
    else:
        ravisual_driver_function(None)

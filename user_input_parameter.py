import pandas as pd
from ras.core.reporting.static_file_parameters import StaticFileParameters as file_params


class UserInputParameters:
    def __init__(self, input_filename: str = None):
        """

        Args:
            input_filename:
        """
        # Read input files and create input parameters
        if input_filename is None:
            print("no input file prov")
            df = pd.read_csv(
                "G:\\Research\Interns\\Rachit\\ravisual\\code_user_input\\"
                "newinput.csv")
        else:
            df = pd.read_csv(input_filename)

        # separate dfs
        self.strategy_df = df.loc[df['Parameter Name'] == 'Strategy']. \
            reset_index(drop=True).drop(labels='Parameter Name', axis=1)
        input_params_df = df.set_index('Parameter Name').drop(labels='Strategy',
                                                              axis=0)
        # Switch Analyses on/off
        self.switch_dict = {
            'hist_abs_rel_perf': True,
            'performance_at_yearly_horizon': True,
            'year_on_year_strategy_minus_benchmark_returns': True,
            'year_on_year_total_return': True,
            'market_impact_cost': True,
            'net_of_cost_SR_IR': True,
            'moments_downside_risks': True,
            'worst_period_var_winrate': True,
            'attributions_ff3': True,
            'attributions_carhart4': True,
            'attributions_ff5': True,
            'attributions_custom': True,
            'attributions_ff3_standardized': True,
            'attributions_carhart4_standardized': True,
            'attributions_ff5_standardized': True,
            'attributions_custom_standardized': True,
            'excess_return_decomposition': True,
            'excess_return_correlation': True,
            'tracking_error': True,
            'recent_allocation_sector': True,
            'recent_allocation_region': True,
            'recent_allocation_country': True,
            'historical_allocation_sector': True,
            'historical_allocation_region': True,
            'historical_allocation_country': True,
            'concentration_holdings': True,
            'concentration_weight': True,
            'recent_portfoloio_characteristics': True,
            'hist_avg_portfoloio_characteristics': True,
            'expected_excess_return_table': True,
            'expected_excess_return_charts': True

        }

        # get output parameters
        self.excel_output = True
        self.output_location = input_params_df.loc['Output Folder', 'Value1']
        self.excel_filename = self.output_location + '//excel_output.xlsx'
        self.output_filename = input_params_df.loc['Output Filename', 'Value1']
        self.excel_filename = input_params_df.loc['Excel Output', 'Value1']

        self.comparison_region = input_params_df.loc['Region', 'Value1']
        self.start_date = pd.to_datetime(input_params_df.loc['Start Date',
                                                             'Value1'])
        self.end_date = pd.to_datetime(input_params_df.loc['End Date',
                                                           'Value1'])
        self.name_benchmark = input_params_df.loc['Benchmark', 'Value1']
        self.benchmark_filename = input_params_df.loc['Benchmark', 'Value2']
        self.ff5_plus_mom_sas_filename, self.ff3_plus_mom_sas_filename, \
        self.rf_filename \
            = UserInputParameters.get_sas_filenames(
            region=self.comparison_region)

    @staticmethod
    def get_sas_filenames(region: str):
        """
        Returns  the fama french and rf data set file names based on the region
        Args:
            region:

        Returns:

        """
        if region is None:
            ff5_plus_mom_sas_filename = file_params.US_FF5_plus_mom_data
            ff3_plus_mom_sas_filename = file_params.US_FF3_plus_mom_data
            rf_sas_filename = file_params.US_RF_data
        elif region.lower().strip() == 'europe':
            ff5_plus_mom_sas_filename = file_params.Europe_FF5_plus_mom_data
            ff3_plus_mom_sas_filename = file_params.Europe_FF3_plus_mom_data
            rf_sas_filename = file_params.Europe_RF_data
        elif region.lower().strip() == 'dev':
            ff5_plus_mom_sas_filename = file_params.Global_FF5_plus_mom_data
            ff3_plus_mom_sas_filename = file_params.Global_FF3_plus_mom_data
            rf_sas_filename = file_params.Global_RF_data
        else:  # or region.lower().strip() == 'us'
            ff5_plus_mom_sas_filename = file_params.US_FF5_plus_mom_data
            ff3_plus_mom_sas_filename = file_params.US_FF3_plus_mom_data
            rf_sas_filename = file_params.US_RF_data

        return ff5_plus_mom_sas_filename, ff3_plus_mom_sas_filename, rf_sas_filename
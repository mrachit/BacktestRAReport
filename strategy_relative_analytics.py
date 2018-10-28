import pandas as pd
import numpy as np
from scipy import stats
from ras.core.reporting.strategy_analytics import StrategyAnalytics
from ras.core.reporting.strategy_analytics_parameters import RelativeStrategyAnalyticsParameters\
    as params


class StrategyRelativeAnalytics:

    def __init__(self, strategy, benchmark, nolag_date):
        """
        Step1	Initialize and check date filter between strategy and benchmark
        Step2	Add Relative measures -	Merge the benchmark columns to all_data.
                Calculate Valuation Ratio, Z-score, Excess Historical till date,
                Excess Subsequent return , Excess 1,3,5,10 Historical,
                Change in Relative valuation measures
        Step3	Current measures
        Step4	Expected excess returns
        Step5	Change in relative measures
        Step6	excess return decomposition
        Step7	IR

        TODO: add exit and error warning for benchmark not available.
        TODO: add unit tests
        TODO: Add other Error Handling

        """
        # nolag_date = nolag_date
        self.nolag_date = nolag_date

        # Initialize self Strategy
        self.strategy = strategy
        self.name=strategy.name

        # Initialize benchmark strategy
        self.benchmark = benchmark

        # date check
        self.start_date = max(self.strategy.start_date, benchmark.start_date)
        self.last_date = min(self.strategy.last_date, benchmark.last_date)

        self.number_of_subsequent_years = params.NUMBER_OF_SUBSEQUENT_YEARS

        # add Relative Measures
        self.all_data = StrategyRelativeAnalytics.add_relative_measures(
            strategy_data=self.strategy.all_data,
            benchmark_data=self.benchmark.all_data,
            nolag_date=nolag_date
        )

        # Current Measures
        self.current_hist_itd_excess_gross = \
            self.all_data.loc[self.last_date, params.HISTORICAL_EXCESS_ITD_COL_NAME]
        self.current_reval_alpha = \
            StrategyRelativeAnalytics.get_revaluation_alpha(df=self.all_data)

        self.current_hist_itd_struct_alpha = \
            self.current_hist_itd_excess_gross - self.current_reval_alpha
        self.current_hist_itd_tracking_error = \
            self.get_historical_itd_tracking_error(df=self.all_data)
        self.current_rel_val_agg = \
            self.all_data[params.REL_MEASURE_COL_NAME].iloc[-1]
        self.current_rel_val_agg_50th = \
            np.median(self.all_data[params.REL_MEASURE_COL_NAME])
        self.current_z_log_rel_val_agg = self.all_data.loc[
            self.last_date, params.Z_SCORE_COL_NAME]
        self.current_hist_itd_struct_beta = \
            StrategyRelativeAnalytics.get_struct_beta(
                self.all_data[params.SUBSEQUENT_EXCESS_RET_COL_NAME],
                self.all_data[params.Z_SCORE_COL_NAME])

        # expected future return
        self.all_data[params.EXPECTED_EXCESS_RETURN_COL_NAME] = \
            StrategyRelativeAnalytics.add_expected_return(
            input_x_col=self.all_data[params.Z_SCORE_COL_NAME].copy(),
            struct_alpha=self.current_hist_itd_struct_alpha,
            struct_beta=self.current_hist_itd_struct_beta
            )

        #
        self.current_expected_excess_return = \
            self.current_hist_itd_struct_alpha + \
            self.current_hist_itd_struct_beta * \
            self.current_z_log_rel_val_agg

        self.current_exp_5yr_excess_SE = StrategyRelativeAnalytics.get_sample_std_dev(
            pd.Series(self.all_data[params.EXPECTED_EXCESS_RETURN_COL_NAME] -
                      self.all_data[params.SUBSEQUENT_EXCESS_RET_COL_NAME]))

        # Current Excess Returns Annualized
        self.current_hist_1yr_nolag_excess_return_annualized = self.all_data.loc[
            self.nolag_date,
            params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace('%year', str(1))]
        self.current_hist_3yr_nolag_excess_return_annualized = self.all_data.loc[
            self.nolag_date, params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
                '%year', str(3))]
        self.current_hist_5yr_nolag_excess_return_annualized = self.all_data.loc[
            self.nolag_date, params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
                '%year', str(5))]
        self.current_hist_10yr_nolag_excess_return_annualized \
            = self.all_data.loc[self.nolag_date, params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
                '%year', str(10))]

        # Current Excess Returns nominal
        self.current_hist_1yr_nolag_excess_return_nominal = \
            self.current_hist_1yr_nolag_excess_return_annualized
        self.current_hist_3yr_nolag_excess_return_nominal = \
            ((1 + self.current_hist_3yr_nolag_excess_return_annualized) ** 3) - 1
        self.current_hist_5yr_nolag_excess_return_nominal = \
            ((1 + self.current_hist_5yr_nolag_excess_return_annualized) ** 5) - 1
        self.current_hist_10yr_nolag_excess_return_nominal = \
            ((1 + self.current_hist_10yr_nolag_excess_return_annualized) ** 10) - 1

        # Current Excess Returns Annualized
        self.current_hist_1yr_excess_return_annualized = \
        self.all_data.loc[
            self.last_date,
            params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace('%year',
                                                                  str(1))]
        self.current_hist_3yr_excess_return_annualized = \
        self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
                '%year', str(3))]
        self.current_hist_5yr_excess_return_annualized = \
        self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
                '%year', str(5))]
        self.current_hist_10yr_excess_return_annualized \
            = self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
                '%year', str(10))]

        # Current Excess Returns nominal
        self.current_hist_1yr_excess_return_nominal = \
            self.current_hist_1yr_excess_return_annualized
        self.current_hist_3yr_excess_return_nominal = \
            (
            (1 + self.current_hist_3yr_excess_return_annualized) ** 3) - 1
        self.current_hist_5yr_excess_return_nominal = \
            (
            (1 + self.current_hist_5yr_excess_return_annualized) ** 5) - 1
        self.current_hist_10yr_excess_return_nominal = \
            ((
             1 + self.current_hist_10yr_excess_return_annualized) ** 10) - 1

        # Current Change in Relative Measures
        self.current_percent_change_1yr_rel_val = self.all_data.loc[
            self.last_date, params.REL_MEASURE_CHANGE_COL_NAME.replace('%year',
                                                                       str(1))]
        self.current_percent_change_3yr_rel_val = self.all_data.loc[
            self.last_date, params.REL_MEASURE_CHANGE_COL_NAME.replace('%year',
                                                                       str(3))]
        self.current_percent_change_5yr_rel_val = self.all_data.loc[
            self.last_date, params.REL_MEASURE_CHANGE_COL_NAME.replace('%year',
                                                                       str(5))]
        self.current_percent_change_10yr_rel_val = self.all_data.loc[
            self.last_date, params.REL_MEASURE_CHANGE_COL_NAME.replace('%year',
                                                                       str(10))]

        # Current Change in NOLAG Relative Measures
        self.current_percent_change_1yr_nolag_rel_val = self.all_data.loc[
            self.nolag_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(1))]
        self.current_percent_change_3yr_nolag_rel_val = self.all_data.loc[
            self.nolag_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(3))]
        self.current_percent_change_5yr_nolag_rel_val = self.all_data.loc[
            self.nolag_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(5))]
        self.current_percent_change_10yr_nolag_rel_val = self.all_data.loc[
            self.nolag_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(10))]
        #combined nolag measures
        self.current_percent_change_1yr_nolag_concat_relval = self.all_data.loc[
            self.last_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(1))]

        self.current_percent_change_3yr_nolag_concat_relval = self.all_data.loc[
            self.last_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(3))]
        self.current_percent_change_5yr_nolag_concat_relval = self.all_data.loc[
            self.last_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(5))]
        self.current_percent_change_10yr_nolag_concat_relval = self.all_data.loc[
            self.last_date, params.REL_MEASURE_NOLAG_CHANGE_COL_NAME.replace('%year',
                                                                       str(10))]

        # get excess return decomposition
        self.log_excess_ret_decomposition \
            = (StrategyRelativeAnalytics
               .get_excess_return_decomposition(
                                                benchmark=benchmark,
                                                strategy=strategy
                                                )
            )
        # IR
        self.current_IR \
            = self.current_hist_itd_excess_gross / self.current_hist_itd_tracking_error
        self.IR_net_cost= (self.current_hist_itd_excess_gross - self.strategy.cost)/self.current_hist_itd_tracking_error

        self.excess_ret_distribution=StrategyRelativeAnalytics.get_excess_ret_distribution(
            df= self.all_data.copy(),
            excess_ret_col=params.EXCESS_RETURN_COL_NAME,
            hist_1yr_excess_ret_col=params.HISTORICAL_YEAR_RETURN_COL_NAME.replace('%year',str(1)),
            hist_3yr_excess_ret_col= params.HISTORICAL_YEAR_RETURN_COL_NAME.replace('%year',str(3)),
            hist_5yr_excess_ret_col= params.HISTORICAL_YEAR_RETURN_COL_NAME.replace('%year',str(5)),
            hist_10yr_excess_ret_col=params.HISTORICAL_YEAR_RETURN_COL_NAME.replace('%year',str(10)),
            strategy_obj= self.strategy
        )

    @staticmethod
    def get_excess_return_decomposition(
            benchmark,
            strategy
    ) -> dict:
        """
        Returns  the excess

        Args:
            benchmark:
            strategy:

        Returns:

        """

        return {
            'log_excess_return':
                strategy.return_decomposition['log_return'] -
                benchmark.return_decomposition['log_return'],
            'log_excess_div_return':
                strategy.return_decomposition['log_div_return'] -
                benchmark.return_decomposition['log_div_return'],
            'log_excess_growth_PE_return':
                strategy.return_decomposition['log_growth_PE_return'] -
                benchmark.return_decomposition['log_growth_PE_return'],
            'log_excess_earnings_growth_return':
                strategy.return_decomposition['log_growth_earnings_return'] -
                benchmark.return_decomposition['log_growth_earnings_return']
               }

    @staticmethod
    def add_expected_return(input_x_col, struct_alpha, struct_beta):
        """

        Args:
            input_x_col:
            struct_alpha:
            struct_beta:

        Returns:

        """
        return struct_alpha + (struct_beta * input_x_col)

    @staticmethod
    def get_sample_std_dev(ds):
        """

        :param ds: pandas DataSeries object with numeric
        :return:
        """

        ds.dropna()
        return np.std(ds, ddof=1)

    @classmethod
    def get_revaluation_alpha(cls, df: pd.DataFrame):
        """

        Args:
            df:

        Returns:

        """
        time = pd.DataFrame({
            'TimeInYears': df[params.MONTHS_TO_DATE_COL_NAME] / 12.0})
        result = stats.linregress(time['TimeInYears'],
                                  df[params.LOG_REL_MEASURE_COL_NAME])
        return result[0]

    @classmethod
    def get_struct_beta(cls, actual_subsequent_excess_return, zscore):
        """
        Gives the beta coefficient of the regression

        Args:
            actual_subsequent_excess_return: 'Y' of the regression.
                pandas.DataSeries object which has same number of observations
                as zscore
            zscore: 'X' of the regression.
                pandas.DataSeries object which has same number of observations
                as actual_subsequent_excess_return

        Returns:
            Beta coefficient of the regression of Y vs X
        """

        # Note: Need same length pd.Series as inputs
        a = pd.concat([actual_subsequent_excess_return, zscore], axis=1)
        a = a.dropna()
        a.columns = ['Y', 'X']
        result = stats.linregress(a['X'], a['Y'])
        return result[0]

    @classmethod
    def add_relative_measures(cls, strategy_data, benchmark_data, nolag_date):
        """
        Merge the benchmark columns to all_data.
        Calculate Valuation Ratio, Z-score, Excess Historical till date,
        Excess Subsequent return , Excess 1,3,5,10 Historical,
        Change in Relative valuation measures

        Args:
            strategy_data:
            benchmark_data:

        Returns:

        """

        # Join benchmark data
        combined_df = strategy_data.join(
            other=benchmark_data[params.BENCHMARK_COL_LIST],
            how='left',
            rsuffix='_B')

        # calculate relative measures
        combined_df[params.RELATIVE_PB_COL_NAME] = \
            combined_df['PB'] / combined_df['PB_B']
        combined_df[params.RELATIVE_PS_COL_NAME] = \
            combined_df['PS'] / combined_df['PS_B']
        combined_df[params.RELATIVE_PE_COL_NAME] = \
            combined_df['PE'] / combined_df['PE_B']
        combined_df[params.RELATIVE_PD_COL_NAME] = \
            combined_df['PD'] / combined_df['PD_B']

        # NOLAG MEASURES
        combined_df[params.RELATIVE_PB_NOLAG_COL_NAME] = \
            combined_df['nolag_PB'] / combined_df['nolag_PB_B']
        combined_df[params.RELATIVE_PS_NOLAG_COL_NAME] = \
            combined_df['nolag_PS'] / combined_df['nolag_PS_B']
        combined_df[params.RELATIVE_PE_NOLAG_COL_NAME] = \
            combined_df['nolag_PE'] / combined_df['nolag_PE_B']
        combined_df[params.RELATIVE_PD_NOLAG_COL_NAME] = \
            combined_df['nolag_PD'] / combined_df['nolag_PD_B']



        combined_df[params.EXCESS_RETURN_COL_NAME] = \
            ((1+combined_df[params.MONTHLY_RETURN_COL_NAME])/  \
            (1+combined_df[params.MONTHLY_RETURN_COL_NAME + '_B'])) -1


        # Relative valuation Ration as geometric mean of individual measures
        combined_df[params.REL_MEASURE_COL_NAME] = \
            (combined_df[['RelPB', 'RelPS', 'RelPE', 'RelPD']].product(axis=1,
             skipna=True)) ** (1 / (combined_df[['RelPB', 'RelPS', 'RelPE',
                                                  'RelPD']].count(axis=1)))

        combined_df[params.REL_MEASURE_NOLAG_COL_NAME] = \
            (combined_df[[params.RELATIVE_PB_NOLAG_COL_NAME,
                          params.RELATIVE_PE_NOLAG_COL_NAME,
                          params.RELATIVE_PD_NOLAG_COL_NAME,
                          params.RELATIVE_PS_NOLAG_COL_NAME]].product(axis=1,
             skipna=True)) ** (1 /
                               (combined_df[[params.RELATIVE_PB_NOLAG_COL_NAME,
                          params.RELATIVE_PE_NOLAG_COL_NAME,
                          params.RELATIVE_PD_NOLAG_COL_NAME,
                          params.RELATIVE_PS_NOLAG_COL_NAME]].count(axis=1)))

        # filling NOLAG measures with lagged measures
        combined_df.loc[combined_df.index>nolag_date,params.REL_MEASURE_NOLAG_COL_NAME] = \
            combined_df.loc[
                combined_df.index > nolag_date, params.REL_MEASURE_COL_NAME]


        combined_df[params.LOG_REL_MEASURE_COL_NAME] = np.log(combined_df[
                                                params.REL_MEASURE_COL_NAME])

        sigma_log_rel_val_agg = cls.get_sample_std_dev(
            combined_df[params.LOG_REL_MEASURE_COL_NAME])

        mean_log_rel_val_agg = np.mean(
            combined_df[params.LOG_REL_MEASURE_COL_NAME])

        combined_df[params.Z_SCORE_COL_NAME] = \
            (combined_df[params.LOG_REL_MEASURE_COL_NAME] - mean_log_rel_val_agg) \
            / sigma_log_rel_val_agg

        combined_df[params.HISTORICAL_EXCESS_ITD_COL_NAME] = \
            ((1 + combined_df[params.HISTORICAL_NOM_ITD_COL_NAME]) /
             (1 + combined_df[params.HISTORICAL_NOM_ITD_COL_NAME + '_B'])) - 1

        combined_df[params.SUBSEQUENT_EXCESS_RET_COL_NAME] = \
            ((1 + combined_df[params.SUBSEQUENT_RETURN_COL_NAME]) /
             (1 + combined_df[params.SUBSEQUENT_RETURN_COL_NAME + '_B'])) - 1


        combined_df[params.STRATEGY_MINUS_BENCHMARK_12M_RETURN_COL_NAME] \
            = combined_df[params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                                                '%year', str(1))] - \
              combined_df[params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                                                '%year', str(1))+ '_B']



        cls.add_wealth_over_benchmark(combined_df)

        cls.add_hist_excess_return(df=combined_df, hist_years=1)
        cls.add_hist_excess_return(df=combined_df, hist_years=3)
        cls.add_hist_excess_return(df=combined_df, hist_years=5)
        cls.add_hist_excess_return(df=combined_df, hist_years=10)

        cls.add_change_rel_measure(df=combined_df, hist_years=1)
        cls.add_change_rel_measure(df=combined_df, hist_years=3)
        cls.add_change_rel_measure(df=combined_df, hist_years=5)
        cls.add_change_rel_measure(df=combined_df, hist_years=10)


        cls.add_change_rel_measure(
            df=combined_df, hist_years=1,
            rel_measure_col_name=params.REL_MEASURE_NOLAG_COL_NAME,
            change_measure_col_name=params.REL_MEASURE_NOLAG_CHANGE_COL_NAME)

        cls.add_change_rel_measure(
            df=combined_df, hist_years=3,
            rel_measure_col_name=params.REL_MEASURE_NOLAG_COL_NAME,
            change_measure_col_name=params.REL_MEASURE_NOLAG_CHANGE_COL_NAME)

        cls.add_change_rel_measure(
            df=combined_df, hist_years=5,
            rel_measure_col_name=params.REL_MEASURE_NOLAG_COL_NAME,
            change_measure_col_name=params.REL_MEASURE_NOLAG_CHANGE_COL_NAME)

        cls.add_change_rel_measure(
            df=combined_df, hist_years=10,
            rel_measure_col_name=params.REL_MEASURE_NOLAG_COL_NAME,
            change_measure_col_name=params.REL_MEASURE_NOLAG_CHANGE_COL_NAME)



        return combined_df


    @staticmethod
    def get_excess_ret_distribution(
            df,
            excess_ret_col,
            hist_1yr_excess_ret_col: str,
            hist_3yr_excess_ret_col: str,
            hist_5yr_excess_ret_col: str,
            hist_10yr_excess_ret_col: str,
            strategy_obj: StrategyAnalytics
            ) -> dict:
        """

        Args:
            df:
            excess_ret_col:

        Returns:

        """
        excess_ret_series = df[excess_ret_col].copy()[1:]
        avg=np.mean(excess_ret_series)*12
        std_dev=np.std(excess_ret_series,ddof=1)* np.sqrt(12)
        skewness = stats.skew(excess_ret_series)
        kurtosis = stats.kurtosis(excess_ret_series)

        semi_series= excess_ret_series[excess_ret_series <=avg]
        up_side_series= excess_ret_series[excess_ret_series >0]
        down_side_series =  excess_ret_series[excess_ret_series <= 0]

        semi_deviation = np.std(semi_series,ddof=1)* np.sqrt(12)
        up_side_deviation = np.std(up_side_series, ddof=1)* np.sqrt(12)
        down_side_deviation = np.std(down_side_series, ddof=1)* np.sqrt(12)
        worst_return_char_date= excess_ret_series.idxmin()
        worst_return_date= worst_return_char_date -pd.Timedelta(1,unit='D')
        worst_return = excess_ret_series[worst_return_char_date]

        excess_1year_returns= df[hist_1yr_excess_ret_col].copy().dropna()

        excess_3year_returns = df[hist_3yr_excess_ret_col].copy().dropna()

        excess_5year_returns = df[hist_5yr_excess_ret_col].copy().dropna()


        worst_return_last_char_12m_date = excess_1year_returns.idxmin()
        worst_return_last_12m_date = worst_return_last_char_12m_date - pd.Timedelta(1,Unit='D')
        worst_return_12m = excess_1year_returns.loc[worst_return_last_char_12m_date]

        worst_return_last_char_36m_date = excess_3year_returns.idxmin()
        worst_return_last_36m_date = worst_return_last_char_36m_date - pd.Timedelta(1,Unit='D')
        worst_return_36m = excess_3year_returns.loc[worst_return_last_char_36m_date]

        worst_return_last_char_60m_date = excess_5year_returns.idxmin()
        worst_return_last_60m_date = worst_return_last_char_60m_date - pd.Timedelta(1,Unit='D')
        worst_return_60m = excess_5year_returns.loc[worst_return_last_char_60m_date]


        beta_results= StrategyAnalytics.get_regression_results(df=strategy_obj.ff5_plus_mom_base,
                                                 return_col=params.MONTHLY_RETURN_COL_NAME,
                                                 rf_col='RF',
                                                 col_list=['Mkt_RF']
                                                 )
        up_side_beta_results= StrategyAnalytics.get_regression_results(df=strategy_obj.ff5_plus_mom_base[strategy_obj.ff5_plus_mom_base['Mkt_RF']>0],
                                                 return_col=params.MONTHLY_RETURN_COL_NAME,
                                                 rf_col='RF',
                                                 col_list=['Mkt_RF']
                                                 )

        down_side_beta_results= StrategyAnalytics.get_regression_results(df=strategy_obj.ff5_plus_mom_base[strategy_obj.ff5_plus_mom_base['Mkt_RF']<=0],
                                                 return_col=params.MONTHLY_RETURN_COL_NAME,
                                                 rf_col='RF',
                                                 col_list=['Mkt_RF']
                                                 )

        win_rate_1yr = len(excess_1year_returns[excess_1year_returns>0])/ len(excess_1year_returns)

        win_rate_3yr = len(excess_3year_returns[excess_3year_returns>0])/ len(excess_3year_returns)

        win_rate_5yr = len(excess_5year_returns[excess_5year_returns>0])/ len(excess_5year_returns)

        percentile_5_1m= np.percentile(excess_ret_series,5)
        percentile_5_1yr = np.percentile(excess_1year_returns, 5)
        percentile_5_3yr = np.percentile(df[hist_3yr_excess_ret_col].copy().dropna(), 5)
        percentile_5_5yr = np.percentile(df[hist_5yr_excess_ret_col].copy().dropna(), 5)

        return {
            'avg': avg,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'semi_dev': semi_deviation,
            'up_side_dev': up_side_deviation,
            'down_side_dev': down_side_deviation,
            'worst_return_1m_date': worst_return_date,
            'worst_return_1m': worst_return,
            'worst_return_12m_date': worst_return_last_12m_date,
            'worst_return_12m':worst_return_12m,
            'worst_return_36m_date': worst_return_last_36m_date,
            'worst_return_36m': worst_return_36m,
            'worst_return_60m_date': worst_return_last_60m_date,
            'worst_return_60m': worst_return_60m,

            '5_percentile_1m':percentile_5_1m,
            '5_percentile_1yr': percentile_5_1yr,
            '5_percentile_3yr': percentile_5_3yr,
            '5_percentile_5yr': percentile_5_5yr,
            # '5_percentile_10yr': np.percentile(
            # pd.Series(df[hist_10yr_excess_ret_col].copy()), 5,axis=0),
            'beta_results': beta_results,
            'up_side_beta_results': up_side_beta_results,
            'down_side_beta_results': down_side_beta_results,
            'win_rate_1yr': win_rate_1yr,
            'win_rate_3yr': win_rate_3yr,
            'win_rate_5yr': win_rate_5yr
        }

    @staticmethod
    def get_historical_portfolio_characteristics(df):
        """

        Args:
            df:

        Returns:

        """


        np.mean(df['PE'].copy())
        np.mean(df['PB'].copy())
        np.mean(df['PS'].copy())
        np.mean(df['PD'].copy())

        np.mean(df[params.RELATIVE_PE_COL_NAME].copy())
        np.mean(df[params.RELATIVE_PB_COL_NAME].copy())
        np.mean(df[params.RELATIVE_PS_COL_NAME].copy())
        np.mean(df[params.RELATIVE_PD_COL_NAME].copy())
        np.mean(df[params.REL_MEASURE_COL_NAME].copy())

    @classmethod
    def get_historical_itd_tracking_error(cls, df):
        """

        Args:
            df:

        Returns:

        """
        # Calculate the Sample standard deviation
        return (np.std(df[params.MONTHLY_RETURN_COL_NAME] -
                       df[params.MONTHLY_RETURN_COL_NAME + '_B'], ddof=1)
                * (12 ** 0.5)
                )

    @classmethod
    def add_wealth_over_benchmark(cls, df):
        """

        Args:
            df:

        Returns:

        """
        col_name = params.CUMULATIVE_RETURN_COL_NAME
        df[params.CUM_PROD_EXCESS_COL_NAME] = df[col_name] / df[col_name + '_B']

    # self.all_data[
        # 'Historical' + str(hist_years) + 'YrReturn_B']

    @classmethod
    def add_hist_excess_return(cls, df: pd.DataFrame, hist_years: int):
        """

        Args:
            df:
            hist_years:

        Returns:

        """
        hist_ret_col_name = params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                                                '%year', str(hist_years))
        hist_excess_ret_col_name = params.HISTORICAL_YEAR_EXCESS_RETURN_COL_NAME.replace(
            '%year', str(hist_years))

        df[hist_excess_ret_col_name] = ((1 + df[hist_ret_col_name]) / (1 +
                                                df[hist_ret_col_name+'_B'])) - 1

    @classmethod
    def add_change_rel_measure(
            cls,
            df: pd.DataFrame,
            hist_years: int,
            rel_measure_col_name: str=params.REL_MEASURE_COL_NAME,
            change_measure_col_name: str = params.REL_MEASURE_CHANGE_COL_NAME
            ):
        """
        Annualized change in relative measure

        Args:
            df:
            hist_years:
            rel_measure_col_name:

        Returns:

        """
        column_name = change_measure_col_name.replace(
                                        '%year', str(hist_years))
        df[column_name] = np.nan
        new_data = df[rel_measure_col_name].copy()
        months = 12 * hist_years
        df[column_name] = ((new_data / new_data.shift(months)) ** (1.0 / hist_years)) - 1

import pandas as pd
import numpy as np
from ras.core.reporting.strategy_analytics_parameters import \
    StrategyAnalyticsParameters as params
from scipy import stats
import statsmodels.api as sm


class StrategyAnalytics:
    def __init__(self,
                 name: str,
                 location: str,
                 filename: str,
                 start_date: pd.datetime,
                 last_date: pd.datetime,
                 nolag_date: pd.datetime,
                 rf_filename: str,
                 ff3_mom_filename: str,
                 ff5_mom_filename: str,
                 custom_regression_flag: bool = False,
                 custom_regression_filename: str = '',
                 custom_regression_date_col: str = ''
                 ):
        """
        Used to initialize and build all the attributes of StrategyAnalytics
        object

        The class objects are also passed to Strategy Relative analytics to
        calculate the relative measures

        Step1	initialize name,loc, start_date, end_date,
        Step2	read data
        Step3   Apply relevant date filters
        Step4   add analyses columns: cumulative returns, Months till date, historical returns, subsequent returns,
        Step5	Store current historical returns
        Step6   get trading costs, concentration data, returns decomposition
        Step7	Get performance Measures
        Step8	Get different datframes: allocation,concentration,
        famafrench+ mom with returns

        Args:
            name: Name of the Strategy
            location: Location/ Region of the strategy
            filename: the filename ( including path) of the time series file
                (standard output file of the strategy)
            start_date:
            last_date:
            nolag_date:
            rf_filename:
            ff3_mom_filename:
            ff5_mom_filename:
        """
        # ----------------------------------------------------------------------
        # Initialization

        self.name = name
        self.location = location
        self.filename = filename
        self.start_date_input = pd.to_datetime(start_date)
        self.start_date = pd.to_datetime(start_date)
        self.last_date = pd.to_datetime(last_date)
        self.number_of_subsequent_years = params.NUMBER_OF_SUBSEQUENT_YEARS

        # ----------------------------------------------------------------------
        # Read data

        returns_df = StrategyAnalytics.read_data(
            excel_filename=filename,
            tab_name=params.RETURNS_TAB_NAME,
            col_list=params.KEEP_RETURNS_COL_LIST
        )
        returns_df['date'] = pd.to_datetime(returns_df['date'])

        returns_df = StrategyAnalytics.get_rf_data(df=returns_df,
                                                   rf_sas_filename=rf_filename)

        characteristics_df = StrategyAnalytics.read_data(
            excel_filename=filename,
            tab_name=params.CHARACTERISTICS_TAB_NAME,
            col_list=params.KEEP_CHARACTERISTICS_COL_LIST
        )

        characteristics_df['date'] = pd.to_datetime(characteristics_df['date'])

        self.all_data = \
            StrategyAnalytics.read_char_return_input(
                returns_df=returns_df,
                characteristics_df=characteristics_df)

        # ----------------------------------------------------------------------
        # Fix date issues! for characteristics. DO it for returns too

        # check start date
        min_date = min(self.all_data['date_characteristics'])
        if min_date > self.start_date:
            self.start_date = min_date

        # check last date
        max_date = max(self.all_data['date_characteristics'])
        if max_date < self.last_date:
            self.last_date = max_date

        # date filter
        # start date filter

        self.all_data = StrategyAnalytics.get_filtered_data_on_column(
            df=self.all_data,
            col_name='date_characteristics',
            value=self.start_date,
            relation='ge'
        )

        # last date filter
        self.all_data = StrategyAnalytics.get_filtered_data_on_column(
            df=self.all_data,
            col_name='date_characteristics',
            value=self.last_date,
            relation='le'
        )

        # ----------------------------------------------------------------------
        # Some initialization for making sure no return is added when the month
        # just starts

        # set first row as zero return since it is prev month return
        self.all_data.loc[self.start_date, params.MONTHLY_RETURN_COL_NAME] = 0.0
        self.all_data.loc[self.start_date, 'ret_ex_div'] = 0.0
        self.all_data.loc[self.start_date, 'RF'] = 0.0
        # ----------------------------------------------------------------------
        # add useful columns for later analyses

        # add Cumulative Returns
        self.all_data[params.CUMULATIVE_RETURN_COL_NAME] = \
            np.cumprod(1 + self.all_data[params.MONTHLY_RETURN_COL_NAME].copy())

        self.all_data['CumProdRF'] = \
            np.cumprod(1 + self.all_data['RF'].copy())

        # add Months
        StrategyAnalytics.add_months_to_date(
            df=self.all_data,
            date_col='date_characteristics',
            start_date_val=start_date,
            new_col_name=params.MONTHS_TO_DATE_COL_NAME
        )

        # add historical Nominal returns
        StrategyAnalytics.add_historical_itd_nominal_return(
            df=self.all_data,
            months_to_date_col=params.MONTHS_TO_DATE_COL_NAME,
            cumprodreturn_col=params.CUMULATIVE_RETURN_COL_NAME,
            historical_nominal_itd_col_name=params.HISTORICAL_NOM_ITD_COL_NAME
        )

        # add Subsequent Returns

        StrategyAnalytics.add_subsequent_return(
            df=self.all_data,
            number_of_years=params.NUMBER_OF_SUBSEQUENT_YEARS,
            cumprodreturn_col=params.CUMULATIVE_RETURN_COL_NAME,
            subsequent_returns_col_name=params.SUBSEQUENT_RETURN_COL_NAME
        )

        # ----------------------------------------------------------------------
        # Historical Returns by year

        StrategyAnalytics.add_hist_return(
            df=self.all_data,
            cumprodreturn_col=params.CUMULATIVE_RETURN_COL_NAME,
            hist_years=1
        )

        StrategyAnalytics.add_hist_return(
            df=self.all_data,
            cumprodreturn_col=params.CUMULATIVE_RETURN_COL_NAME,
            hist_years=3
        )

        StrategyAnalytics.add_hist_return(
            df=self.all_data,
            cumprodreturn_col=params.CUMULATIVE_RETURN_COL_NAME,
            hist_years=5
        )

        StrategyAnalytics.add_hist_return(
            df=self.all_data,
            cumprodreturn_col=params.CUMULATIVE_RETURN_COL_NAME,
            hist_years=10
        )

        # Current Nominal Returns
        self.current_hist_1yr_return = self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                '%year', str(1))]
        self.current_hist_3yr_return = self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                '%year', str(3))]
        self.current_hist_5yr_return = self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                '%year', str(5))]
        self.current_hist_10yr_return = self.all_data.loc[
            self.last_date, params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
                '%year', str(10))]

        # get trading costs
        self.trading_cost = StrategyAnalytics.get_turnover_data(
            file_name=filename,
            tab_name='Turnovers_and_Costs',
            start_date=self.start_date,
            last_date=self.last_date
        )
        # get concentration data
        self.concentration_df = StrategyAnalytics.get_concentration_df(
            filename=filename,
            start_date=self.start_date,
            last_date=self.last_date
        )

        # get returns decomposition

        self.return_decomposition = StrategyAnalytics.get_return_decomposition(
            df=self.all_data,
            nolag_date=nolag_date
        )

        # get Performance Measures
        self.HistITDVol = np.std(
            self.all_data[params.MONTHLY_RETURN_COL_NAME].copy().dropna(),
            ddof=1) * np.sqrt(12)

        self.HistITDVol_nolag = np.std(
            self.all_data[self.all_data.index <= nolag_date][
                params.MONTHLY_RETURN_COL_NAME].copy().dropna(),
            ddof=1) * np.sqrt(12)

        self.current_nominal_itd_return = self.all_data.loc[
            self.last_date, params.HISTORICAL_NOM_ITD_COL_NAME]

        self.current_nominal_nolag_itd_return = self.all_data.loc[
            nolag_date, params.HISTORICAL_NOM_ITD_COL_NAME]

        self.current_nominal_ITD_RF = self.all_data.loc[self.last_date,
                                                        'HistITDRF']

        self.current_nominal_nolag_ITD_RF = self.all_data.loc[nolag_date,
                                                              'HistITDRF']

        self.HistITDSharpe = \
            (self.current_nominal_itd_return - self.current_nominal_ITD_RF
             ) / self.HistITDVol

        self.HistITDSharpe_nolag = \
            (
                self.current_nominal_nolag_itd_return - self.current_nominal_nolag_ITD_RF
            ) / self.HistITDVol_nolag

        self.cost = self.trading_cost["Cost"]
        self.SR_net_cost = (
                               self.HistITDSharpe * self.HistITDVol - self.cost) / self.HistITDVol

        # 10yr calculations

        cum10RF = np.cumprod(1 + self.all_data['RF'].tail(120).copy())
        # Do not display if 10 yr  data not available
        if self.all_data.loc[self.last_date, params.MONTHS_TO_DATE_COL_NAME] < (
            12 * 10):
            cum10RF = np.nan

        cum10RF_nolag = np.cumprod(
            1 + self.all_data[self.all_data.index <= nolag_date]['RF'].tail(
                120).copy())
        self.Hist10YrRF = (cum10RF[-1] ** (12 / len(cum10RF))) - 1
        self.Hist10YrRF_nolag = (cum10RF_nolag[-1] ** (
            12 / len(cum10RF_nolag))) - 1

        cum10Returns = np.cumprod(
            1 + self.all_data[params.MONTHLY_RETURN_COL_NAME].tail(120).copy())
        cum10Returns_nolag = np.cumprod(
            1 + self.all_data[self.all_data.index <= nolag_date][
                params.MONTHLY_RETURN_COL_NAME].tail(120).copy())

        self.Hist10YrReturns = (cum10Returns[-1] ** (12 / len(cum10RF))) - 1

        self.Hist10YrReturns_nolag = (cum10Returns_nolag[-1] ** (
            12 / len(cum10RF_nolag))) - 1

        self.Hist10YrVol = np.std(
            self.all_data[params.MONTHLY_RETURN_COL_NAME].tail(
                120).copy().dropna(),
            ddof=1) * np.sqrt(12)

        self.Hist10YrVol_nolag = np.std(
            self.all_data[self.all_data.index <= nolag_date][
                params.MONTHLY_RETURN_COL_NAME].tail(120).copy().dropna(),
            ddof=1) * np.sqrt(12)

        self.Hist10YrSharpe = (
                                  self.Hist10YrReturns - self.Hist10YrRF) / self.Hist10YrVol
        self.Hist10YrSharpe_nolag = (
                                        self.Hist10YrReturns_nolag - self.Hist10YrRF_nolag) / self.Hist10YrVol_nolag

        # ----------------------------------------------------------------------
        # Get Sector / Country / Region Allocations Dataframes

        self.sector_allocation_df = self.get_allocation_df(
            filename=self.filename,
            tab_name='Sector_Allocation',
            start_date=self.start_date,
            last_date=self.last_date,
            re_normalize=True
        )

        self.country_allocation_df = self.get_allocation_df(
            filename=self.filename,
            tab_name='Country_Allocation',
            start_date=self.start_date,
            last_date=self.last_date,
            re_normalize=True
        )

        region_allocation_df = self.get_allocation_df(
            filename=self.filename,
            tab_name='Region_Allocation',
            start_date=self.start_date,
            last_date=self.last_date,
            re_normalize=True
        )
        self.region_allocation_df = \
            region_allocation_df[params.STANDARD_REGION_LIST]

        # ----------------------------------------------------------------------
        # Get fama French 3,5, databases

        self.ff3_plus_mom_base = StrategyAnalytics.get_fama_french_plus_mom_data(
            df=self.all_data[
                   ['date_returns', params.MONTHLY_RETURN_COL_NAME]].iloc[1:,
               :].copy(),
            ff_sas_filename=ff3_mom_filename
        )

        self.ff5_plus_mom_base = StrategyAnalytics.get_fama_french_plus_mom_data(
            df=self.all_data[
                   ['date_returns', params.MONTHLY_RETURN_COL_NAME]].iloc[1:,
               :].copy(),
            ff_sas_filename=ff5_mom_filename
        )

        # ----------------------------------------------------------------------
        # Get Custom regression database
        if custom_regression_flag:
            self. custom_regression_df = StrategyAnalytics.get_custom_regression_data(
                df=self.all_data[
                       ['date_returns', params.MONTHLY_RETURN_COL_NAME]].iloc[1:,
                   :].copy(),
                custom_csv_filename=custom_regression_filename,
                custom_reg_date_col=custom_regression_date_col
            )

    @staticmethod
    def get_custom_regression_data(
            df: pd.DataFrame,
            custom_csv_filename: str,
            custom_reg_date_col: str,
            df_date_returns_col: str ='date_returns'
    ) -> pd.DataFrame:
        """

        Args:
            df:
            custom_csv_filename:
            custom_reg_date_col:

        Returns:

        """
    #     Read csv , identify date column,


    # Merge with df
        df_custom = pd.read_csv(io=custom_csv_filename)
        df_custom['date']= pd.to_datetime(df_custom[custom_reg_date_col])

        combined_df = pd.merge(left=df,
                               right=df_custom,
                               how='left',
                               left_on=[df_date_returns_col],
                               right_on=['date'],
                               sort=True,
                               copy=True,
                               indicator=False
                               )
        # forward fill all columns data
        combined_df.fillna(method='ffill', inplace=True)

        return combined_df

    @staticmethod
    def get_recent_portfolio_characteristics(
            df: pd.DataFrame) -> dict:
        """
        Returns a dictionary with recent PD, PE, PS, PD, PCF, composite
        valuation ratio

        Args:
            df:

        Returns:


        """

        recent_PB = df['PB'].iloc[-1]
        recent_PE = df['PE'].iloc[-1]
        recent_PS = df['PS'].iloc[-1]
        recent_div_yield = df[params.DIV_YIELD_INPUT_COL_NAME].iloc[-1]
        recent_PCF = df[params.PCF_COL_NAME].iloc[-1]
        recent_composite_fundamental = stats.gmean(
            [recent_PB, recent_PE, recent_PS, 1 / recent_div_yield])

        return {
            'recent_PB': recent_PB,
            'recent_PE': recent_PE,
            'recent_PS': recent_PS,
            'recent_div_yield': recent_div_yield,
            'recent_PCF': recent_PCF,
            'recent_composite_fundamental': recent_composite_fundamental
        }

    @staticmethod
    def get_min_max_nolag_date(
            filename: str):
        """
        Returns the minimum, maximum and no lag date in the characterics tab
        Args:
            filename: filename of the time series file

        Returns:

        """
        characteristics_df = StrategyAnalytics.read_data(
            excel_filename=filename,
            tab_name=params.CHARACTERISTICS_TAB_NAME,
            col_list=params.KEEP_CHARACTERISTICS_COL_LIST
        )

        characteristics_df['date'] = pd.to_datetime(characteristics_df['date'])

        min_date = min(characteristics_df['date'])
        max_date = max(characteristics_df['date'])

        nolag_date = characteristics_df.loc[pd.Series(characteristics_df[
                                                          params.PE_NOLAG_INPUT_COL_NAME].copy()).last_valid_index()][
            'date']

        return min_date, max_date, nolag_date

    @staticmethod
    def convert_col_names_lower(
            df: pd.DataFrame) -> None:
        """
        Converts the column names of the input data frame to lowercase
        the column names are updated in place

        Args:
            df: pandas data frame

        Returns:
            the column names are updated in place
            returns None

        """
        df.columns = map(str.lower, df.columns)
        return None

    @staticmethod
    def convert_col_names_upper(df: pd.DataFrame) -> None:
        """
        Converts the column names of the input data frame to uppercase
        the column names are updated in place

        Args:
            df: pandas data frame

        Returns:
            the column names are updated in place
            returns None

        """
        df.columns = map(str.upper, df.columns)

    @classmethod
    def read_data(cls, excel_filename: str,
                  tab_name: str,
                  col_list: list = None) -> pd.DataFrame:
        """
        Reads the excel file_tab , converts the column names to lower case and
        returns relative_analytics pandas.DataFrame

        Args:
            excel_filename: Excel filename which is to be read and supported
                            by pandas.read_excel
            tab_name: Excel tab_name  which is to be loaded
            col_list: subsets the column list

        Returns:
            pandas DataFrame object
        """

        df = pd.read_excel(io=excel_filename, sheet_name=str(tab_name))
        cls.convert_col_names_lower(df)
        if col_list is not None:
            df = df[col_list]

        return df

    @staticmethod
    def get_filtered_data_on_column(
            df: pd.DataFrame,
            col_name: str,
            value,
            relation: str = 'ge') -> pd.DataFrame:
        """
        returns filtered data Frame based on column and a reference value and  operation ( 'ge','le','gt','lt','eq')
        TODO: use op
        Args:
            df: input pandas.DataFrame
            col_name: column name on which the filter is to be applied
            value: the value to be compared
            relation: Default 'ge' - takes 'gt', 'lt','le', 'eq'
                    compares the column values with the value

        Examples:
        relation: 'ge' ==>
            Returns df with rows such that all values in df['col_name'] >= value


        Returns:
            filtered pandas.DataFrame

        """
        if relation == 'ge':
            return df[df[col_name] >= value].reindex()
        elif relation == 'gt':
            return df[df[col_name] > value].reindex()
        elif relation == 'lt':
            return df[df[col_name] < value].reindex()
        elif relation == 'le':
            return df[df[col_name] <= value].reindex()
        elif relation == 'eq':
            return df[df[col_name] == value].reindex()
        else:
            return df

    @classmethod
    def read_char_return_input(cls,
                               returns_df: pd.DataFrame,
                               characteristics_df: pd.DataFrame
                               ) -> pd.DataFrame:
        """
        Reads input from Returns and Characteristics and stores the same in
        relative_analytics combined dataframe.
        The index is set to the date in characteristics column
        Assumes the date column in Returns and Characteristics tab
        TODO: Check using rename instead of new columns and deleting old columns

        Args:
            returns_df:
            characteristics_df:

        Returns:
            returns combined DataFrame after reading Returns and Characteristics
            data

        """
        # ----------------------------------------------------------------------
        # Merge Dfs

        # date match conversion step. Making sure the column for join has same
        # date
        characteristics_df['newdate_temp'] = \
            pd.to_datetime(characteristics_df['date']) - pd.Timedelta(1,
                                                                      unit='D')

        combined_df = pd.merge(left=characteristics_df,
                               right=returns_df,
                               how='left',
                               on=None,
                               left_on='newdate_temp',
                               right_on='date',
                               sort=True,
                               suffixes=('_characteristics', '_returns'),
                               copy=True,
                               indicator=False
                               )
        # del combined_df['newdate_temp']

        # set the characteristics date as index. This is the first date
        combined_df.set_index(
            pd.DatetimeIndex(combined_df['date_characteristics']), inplace=True)
        combined_df['date_characteristics'] = combined_df.index

        # sort index so that we have correct CumProdReturn calculation
        combined_df.sort_index(
            inplace=True)  # set the characteristics date as index. This is the first date
        combined_df.set_index(
            pd.DatetimeIndex(combined_df['date_characteristics']), inplace=True)
        combined_df['date_characteristics'] = combined_df.index

        # sort index so that we have correct CumProdReturn calculation
        combined_df.sort_index(inplace=True)

        # Renaming columns
        combined_df.rename(columns={
            params.PB_INPUT_COL_NAME: 'PB',
            params.PS_INPUT_COL_NAME: 'PS',
            params.PE_INPUT_COL_NAME: 'PE',
            params.PE_NOLAG_INPUT_COL_NAME: 'nolag_PE',
            params.PS_NOLAG_INPUT_COL_NAME: 'nolag_PS',
            params.PB_NOLAG_INPUT_COL_NAME: 'nolag_PB',

            params.MONTHLY_RETURN_INPUT_COL_NAME: params.MONTHLY_RETURN_COL_NAME
        }, inplace=True)

        # some additional processing
        combined_df['PD'] = 1 / combined_df[params.DIV_YIELD_INPUT_COL_NAME]
        combined_df['nolag_PD'] = 1 / combined_df[
            params.DIV_YIELD_NOLAG_INPUT_COL_NAME]

        min_date = min(combined_df.index)
        combined_df.loc[min_date, params.MONTHLY_RETURN_COL_NAME] = 0
        combined_df.loc[min_date, 'RF'] = 0

        return combined_df

    @staticmethod
    def add_months_to_date(df: pd.DataFrame,
                           date_col: str,
                           start_date_val: pd.datetime,
                           new_col_name: str):
        """
        Adds relative_analytics new column with months from the start date

        Args:
            df:
            date_col:
            start_date_val:
            new_col_name:

        Returns:

        """
        df[new_col_name] = ((df[date_col].dt.year - start_date_val.year) * 12) \
                           + (df[date_col].dt.month - start_date_val.month)

    @staticmethod
    def add_historical_itd_nominal_return(df: pd.DataFrame,
                                          months_to_date_col: str,
                                          cumprodreturn_col: str,
                                          historical_nominal_itd_col_name: str
                                          ) -> None:
        """
        Adds Historical till date nominal return column inplace to
        relative_analytics DataFrame

        Args:
            df:
            months_to_date_col:
            cumprodreturn_col:
            historical_nominal_itd_col_name:

        Returns:

        """
        df[historical_nominal_itd_col_name] \
            = df[cumprodreturn_col] ** (12.0 / df[months_to_date_col]) - 1

    @staticmethod
    def add_subsequent_return(df,
                              number_of_years,
                              cumprodreturn_col,
                              subsequent_returns_col_name):
        """
        Adds subsequent annualized return for the next number of years
        Args:
            df:
            number_of_years:
            cumprodreturn_col:
            subsequent_returns_col_name:

        Returns:

        """

        number_of_months = int(number_of_years * 12)

        df[subsequent_returns_col_name] = ((df[cumprodreturn_col].shift(
            -number_of_months) / df[cumprodreturn_col]) ** (1.0 /number_of_years)) - 1

    @staticmethod
    def add_hist_return(
            df: pd.DataFrame,
            cumprodreturn_col: str,
            hist_years: int):
        """
        Computes and adds historical returns for <hist_years> based on
        cumulative returns column. The name of column is calculated busing
        StrategyAnalyticsParameters.HISTORICAL_YEAR_RETURN_COL_NAME
        Args:
            df:
            cumprodreturn_col:
            hist_years:

        Returns:

        """
        hist_ret_col_name = params.HISTORICAL_YEAR_RETURN_COL_NAME.replace(
            '%year', str(hist_years))

        number_of_months = int(hist_years) * 12
        df[hist_ret_col_name] = ((df[cumprodreturn_col] / df[
            cumprodreturn_col].shift(number_of_months)) ** (
                                     1.0 / (hist_years * 1.0))) - 1

    @classmethod
    def get_filtered_read_data(cls,
                               file_name: str,
                               tab_name: str,
                               start_date,
                               last_date,
                               date_col,
                               col_list: list
                               ) -> pd.DataFrame:
        """
        Returns the filtered data from excel filename in the tabname while
        filtering based on the start and end date and capturing the given list
        of columns.
        Args:
            file_name:
            tab_name:
            start_date:
            last_date:
            date_col:
            col_list:

        Returns:

        """

        df = cls.read_data(
            excel_filename=file_name,
            tab_name=tab_name,
            col_list=col_list
        )

        df[date_col] = pd.to_datetime(df[date_col])

        # Fix date issues! Making sure overlap between characteristics and
        # returns

        # check start date
        min_date = min(df[date_col])
        if min_date > start_date:
            start_date = min_date

        # check last date
        max_date = max(df[date_col])
        if max_date < last_date:
            last_date = max_date

        # date filter
        # start date filter

        df = cls.get_filtered_data_on_column(
            df=df,
            col_name=date_col,
            value=start_date,
            relation='ge'
        )

        # last date filter
        df = cls.get_filtered_data_on_column(
            df=df,
            col_name=date_col,
            value=last_date,
            relation='le'
        )
        return df

    @classmethod
    def get_turnover_data(cls,
                          file_name: str,
                          tab_name: str,
                          start_date,
                          last_date,
                          col_list: list = None,
                          aum_in_billions: float = 10.0,
                          bps_cost: float = 50.0
                          ) -> dict:
        """
        Returns the dictionary with trading costs  & turnover data for the
        strategy
        TODO: code fix for rebalance freq less than once in an year
        Args:
            file_name:
            tab_name:
            col_list:
            aum_in_billions:
            start_date:
            last_date:

        Returns:

        """
        df_to = cls.get_filtered_read_data(
            file_name=file_name,
            tab_name=tab_name,
            start_date=start_date,
            last_date=last_date,
            col_list=col_list,
            date_col='date'
        )
        # Assumption: needs at least 1 rebalancing in an year
        rebalance_annual_freq = len(df_to['month'].unique())

        # current portfolio Volume

        if df_to['one_way_to'].iloc[0] < 1:
            avg_to = np.average(df_to['one_way_to'])
            # do it for last 5 years

        else:
            avg_to = np.average(df_to['one_way_to'][1:])

        avg_eff_to = np.average(df_to['eff_to_estimate'])
        # do it for last 5 years
        cost_avg = np.average(df_to['cost_by_summation'].tail(
            int(5 * rebalance_annual_freq)))

        targetted_aum_billions = df_to['targetted_aum'].iloc[-1] / 1000.0

        transaction_cost = ((cost_avg * rebalance_annual_freq) *
                            (aum_in_billions / targetted_aum_billions)
                            )

        bps = bps_cost
        portfolio_volume = df_to['total_vol'].iloc[-1]
        tilt = df_to['portfolio_vol_tilt'].iloc[-1]
        turnover = avg_to * rebalance_annual_freq
        capacity = (bps / 10000.0) * (1.0 / transaction_cost) * aum_in_billions
        turnover_concentration = avg_eff_to / (2.0 * avg_to)
        wamc = df_to['wamc'].iloc[-1]
        effn = 1.0 / (df_to['herfindahl'].iloc[-1])

        # cost_of_strategy = \
        #     (0.03*aum_in_billions * (1.0/rebalance_annual_freq) *
        #      (1.0/portfolio_volume) * tilt * turnover * turnover_concentration
        #      )

        return {"TO": turnover,
                "TO_Concentration": turnover_concentration,
                "Volume": portfolio_volume,
                "Tilt": tilt,
                "Cost": transaction_cost,
                "Capacity": capacity,
                "Rebalance_freq": rebalance_annual_freq,
                "WAMC": wamc,
                "EffN": effn
                }

    @staticmethod
    def get_return_decomposition(
            df: pd.DataFrame,
            nolag_date: pd.datetime) -> dict:
        """
        returns the dictionary with return decomposition of the strategy
        Args:
            df:
            nolag_date:

        Returns:

        """

        df['price'] = np.cumprod(1 + df['ret_ex_div'].copy()) * 100
        df['AnnualizedReturnITD'] = (df[params.CUMULATIVE_RETURN_COL_NAME] ** (
            12 / df[params.MONTHS_TO_DATE_COL_NAME])) - 1

        #  Log Div Return
        df['div_ret'] = ((df[params.MONTHLY_RETURN_COL_NAME] - df['ret_ex_div'])
                         * (df['price'].shift(1) / df['price']))
        df['CumulativeDivReturn'] = np.cumprod(1 + df['div_ret'].copy())

        df['AnnualizedDivReturnITD'] = (
                                           df['CumulativeDivReturn'] ** (
                                               12 / df[
                                                   params.MONTHS_TO_DATE_COL_NAME])) - 1

        #  log PE growth return
        df['PE_nolag'] = df['price_to_earnings_nolag']
        df['change_PE_no_lag'] = df['PE_nolag'] / df['PE_nolag'].shift(1) - 1
        df['change_PE_no_lag'].fillna(0)
        df['growth_PE_no_lag'] = np.cumprod(1 + df['change_PE_no_lag'].copy())

        df['AnnualizedGrowthPEReturnITD'] = (df['growth_PE_no_lag'] ** (12.0 /
                                                                        df[
                                                                            params.MONTHS_TO_DATE_COL_NAME])
                                             ) - 1

        # log Earnings Growth Return
        df['Earnings'] = df['price'] / df['PE_nolag']
        df['Earnings_growth'] = df['Earnings'] / (
            df['price'].iloc[0] / df['PE_nolag'].iloc[0])

        df['AnnualizedGrowthEarningsITD'] = (df['Earnings_growth'] ** (12 /
                                                                       df[
                                                                           params.MONTHS_TO_DATE_COL_NAME])) - 1

        geometric_growth_PE_return = df.loc[
            nolag_date, 'AnnualizedGrowthPEReturnITD']

        log_growth_PE_return = (
            np.log((1 + geometric_growth_PE_return) ** (1 / 12)
                   ) * 12)

        geometric_div_return = df.loc[nolag_date, 'AnnualizedDivReturnITD']
        log_div_return = (np.log((1 + geometric_div_return) ** (1 / 12)) * 12)

        geometric_return = df.loc[nolag_date, 'AnnualizedReturnITD']
        log_return = (np.log((1 + geometric_return) ** (1 / 12)) * 12)

        geometric_growth_earnings_return = df.loc[pd.Series(df[
                                                                'AnnualizedGrowthEarningsITD'].copy()).last_valid_index(), 'AnnualizedGrowthEarningsITD']
        log_growth_earnings_return = \
            (np.log((1 + geometric_growth_earnings_return) ** (1 / 12)) * 12)

        return {
            'log_return': log_return,
            'log_div_return': log_div_return,
            'log_growth_PE_return': log_growth_PE_return,
            'log_growth_earnings_return': log_growth_earnings_return
        }

    @staticmethod
    def get_allocation_df(
            filename: str,
            tab_name: str,
            start_date: pd.datetime,
            last_date: pd.datetime,
            re_normalize=True
    ) -> pd.DataFrame:
        """
        Returns a filted(on dates) allocation DataFrame ( Sector / Region
        / Country). By default the rows are renormalized to sum up to 100%.
        Args:
            filename:
            tab_name:
            start_date:
            last_date:
            re_normalize:

        Returns:

        """
        df = pd.read_excel(filename,
                           tab_name)
        df.rename(index=str, columns={"Date": "date"}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

        # Date Filter
        df = StrategyAnalytics.get_filtered_data_on_column(
            df=df,
            col_name='date',
            value=start_date,
            relation='ge'
        )

        # last date filter
        df = StrategyAnalytics.get_filtered_data_on_column(
            df=df,
            col_name='date',
            value=last_date,
            relation='le'
        )

        df.set_index('date', inplace=True)

        a = df.iloc[:, 2:].copy()
        a = a.fillna(0)

        if re_normalize:
            a = a.div(a.sum(axis=1), axis=0)

        return a

    @staticmethod
    def get_rf_data(
            df: pd.DataFrame,
            rf_sas_filename: str) -> pd.DataFrame:
        """
        Add risk free rate  from the sas file ( Assumes comn name to be 'RF')
        to the data frame
        Args:
            df:
            rf_sas_filename:

        Returns:

        """

        df_rf = pd.read_sas(filepath_or_buffer=rf_sas_filename)

        df_rf['Month'] = df_rf['Month'].astype('int')
        df_rf['Year'] = df_rf['Year'].astype('int')

        df['Month'] = df['date'].dt.month
        df['Year'] = df['date'].dt.year

        combined_df = pd.merge(left=df,
                               right=df_rf,
                               how='left',
                               on=['Year', 'Month'],
                               sort=True,
                               suffixes=('', '_rf'),
                               copy=True,
                               indicator=False
                               )
        # forward fill RF data

        combined_df['RF'].fillna(method='ffill', inplace=True)

        return combined_df

    @staticmethod
    def get_fama_french_plus_mom_data(
            df: pd.DataFrame,
            ff_sas_filename: str,
            df_returns_col: str = 'date_returns'
    ) -> pd.DataFrame:
        """
        Returns the DataFrame for Fama French and Carhart four regressions by
        merging the factors data to the d\input data Frame.
        Specific to the sas dataset formats from the data team.
        Args:
            df:

        Returns:

        """
        df_ffm = pd.read_sas(filepath_or_buffer=ff_sas_filename)

        df_ffm['Month'] = df_ffm['Month'].astype('int')
        df_ffm['Year'] = df_ffm['Year'].astype('int')

        df['Month'] = df[df_returns_col].dt.month
        df['Year'] = df[df_returns_col].dt.year

        combined_df = pd.merge(left=df,
                               right=df_ffm,
                               how='left',
                               on=['Year', 'Month'],
                               sort=True,
                               suffixes=('', '_ff5m'),
                               copy=True,
                               indicator=False
                               )
        # forward fill all columns data
        combined_df.fillna(method='ffill', inplace=True)

        return combined_df


    @staticmethod
    def get_regression_results(df: pd.DataFrame,
                               return_col: str,
                               rf_col: str,
                               col_list: list,
                               standardized_x: bool = False,
                               standardized_y: bool = False,
                               no_standardized_col: list = list()
                               ) -> dict:
        """
        Runs regression based on the returns_col minus rf col on the given
        Factors list.
        Args:
            df:
            return_col:
            rf_col:
            col_list:
            standardized_x:
            standardized_y:
            no_standardized_col:

        Returns:

        """
        df = df[col_list + [return_col, rf_col]].copy().dropna()
        Y = df[return_col] - df[rf_col]
        X = df[col_list]

        if standardized_y:
            stdev_Y = np.std(Y)
            Y = Y.copy() / stdev_Y
        if standardized_x:
            temp_X = X.copy()
            for colname in X.columns:
                if colname in no_standardized_col:
                    continue
                stdev_col = np.std(X[colname])
                temp_X[colname] = (X[colname].copy()) / stdev_col

        X = sm.add_constant(X)
        fit = sm.OLS(Y, X).fit()

        result_df = pd.concat([fit.params, fit.pvalues, fit.tvalues], axis=1)
        result_df.columns = ['coefficient', 'pvalues', 'tvalues']

        results = {'alpha': result_df.loc['const', 'coefficient'] * 12,
                   'alpha_tstat': result_df.loc['const', 'tvalues'],
                   'alpha_pvalue': result_df.loc['const', 'pvalues'],
                   'adj_rsquared': fit.rsquared_adj,
                   'rsquared': fit.rsquared
                   }
        for name in col_list:
            results[name] = result_df.loc[name, 'coefficient']
            results[name + '_tstat'] = result_df.loc[name, 'tvalues']
            results[name + '_pvalue'] = result_df.loc[name, 'pvalues']

        return results

    @staticmethod
    def get_concentration_df(filename: str,
                             start_date,
                             last_date,
                             tab_name: str = params.CONCENTRATIONS_TAB_NAME
                             ) -> pd.DataFrame:
        """
        Returns concentration DataFrame.
        Args:
            filename:
            start_date:
            last_date:
            tab_name:

        Returns:

        """

        df = StrategyAnalytics.read_data(
            excel_filename=filename,
            tab_name=tab_name
        )

        df['date'] = pd.to_datetime(df['date'])

        df = StrategyAnalytics.get_filtered_data_on_column(
            df=df,
            col_name='date',
            value=start_date,
            relation='ge'
        )

        df = StrategyAnalytics.get_filtered_data_on_column(
            df=df,
            col_name='date',
            value=last_date,
            relation='le'
        )

        return df


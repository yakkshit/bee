"""
Parser Façade Module.
Maintains 100% backward compatibility across all project files while
delegating core mathematical engines and parsers to modules/core/.
"""
from modules.core.time_parser import DEFAULT_CSV, parse_time_data, filter_post_starvation_data, is_time_zero
from modules.core.probit_engine import calculate_lt_percentiles
from modules.core.bee_extractor import extract_individual_bee_survival
from modules.core.batch_parser import parse_batch_df, standardize_observation_df
from modules.core.cohort_manager import (
    AGREED_COHORTS,
    COHORT_CODE_NAMES,
    COHORT_PALETTE,
    COHORT_MARKERS,
    clean_cohort_name,
    get_cohort_code,
    get_default_4_cohort_dataframes,
    get_active_cohort_dictionary,
    COHORT_SUCROSE_2UL,
    COHORT_ACETONE_5PCT_2UL,
    COHORT_ACETONE_10PCT_2UL,
    COHORT_ACETONE_5PCT_5UL
)

__all__ = [
    'DEFAULT_CSV',
    'parse_time_data',
    'filter_post_starvation_data',
    'is_time_zero',
    'parse_batch_df',
    'standardize_observation_df',
    'calculate_lt_percentiles',
    'extract_individual_bee_survival',
    'AGREED_COHORTS',
    'COHORT_CODE_NAMES',
    'COHORT_PALETTE',
    'COHORT_MARKERS',
    'clean_cohort_name',
    'get_cohort_code',
    'get_default_4_cohort_dataframes',
    'get_active_cohort_dictionary',
    'COHORT_SUCROSE_2UL',
    'COHORT_ACETONE_5PCT_2UL',
    'COHORT_ACETONE_10PCT_2UL',
    'COHORT_ACETONE_5PCT_5UL'
]

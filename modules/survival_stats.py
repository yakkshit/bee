"""
Survival Statistics Façade Module.
Maintains 100% backward compatibility for all existing scripts and tabs
while delegating core logic to modular sub-packages in modules/survival/.
"""
from modules.parser import extract_individual_bee_survival
from modules.survival.log_rank import render_log_rank_analysis
from modules.survival.cox_regression import render_cox_regression_analysis
from modules.survival.forest_plot import render_hazard_ratio_forest_plot
from modules.survival.math_cards import render_logrank_math_card, render_cox_math_card

__all__ = [
    'render_log_rank_analysis',
    'render_cox_regression_analysis',
    'render_hazard_ratio_forest_plot',
    'render_logrank_math_card',
    'render_cox_math_card',
    'extract_individual_bee_survival'
]

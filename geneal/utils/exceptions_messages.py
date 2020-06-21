exception_messages = {
    "InvalidSelectionStrategy": lambda selection_strategy, allowed_selection_strategies: f"{selection_strategy} is not a valid selection strategy. "
    f"Available options are {', '.join(allowed_selection_strategies)}.",
    "InvalidPopulationSize": "The population size must be larger than 2",
    "InvalidExcludedGenes": lambda excluded_genes: f"{excluded_genes} is not a valid input for excluded_genes",
}

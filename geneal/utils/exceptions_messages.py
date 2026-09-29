exception_messages = {
    "InvalidSelectionStrategy": lambda selection_strategy, allowed_selection_strategies: f"{selection_strategy} is not a valid selection strategy. "
    f"Available options are {', '.join(allowed_selection_strategies)}.",
    "InvalidPopulationSize": "The population size must be larger than 2",
    "InvalidExcludedGenes": lambda excluded_genes: f"{excluded_genes} is not a valid input for excluded_genes",
    "NoFitnessFunction": "A fitness function must be provided as an argument or defined as a method, "
    "or calculate_fitness must be overridden. Alternatively, drive the solver with ask() and tell().",
    "InvalidInitialPopulation": lambda shape, pop_size, n_genes: f"initial_population has shape {shape}, "
    f"but it must be (k, {n_genes}) with 1 <= k <= pop_size ({pop_size})",
}

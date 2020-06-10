from geneal.applications.template import TemplateChildClass


class TestTemplateChildClass:
    def test_template_child_class(self):

        tsp_solver = TemplateChildClass(n_genes=10)

        tsp_solver.fitness_function("dummy_input")
        tsp_solver.initialize_population("dummy_input", "dummy_input")
        tsp_solver.create_offspring(
            "dummy_input", "dummy_input", "dummy_input", "dummy_input"
        )
        tsp_solver.mutate_population("dummy_input", "dummy_input")

        assert getattr(tsp_solver, "initialize_population", None) is not None
        assert getattr(tsp_solver, "create_offspring", None) is not None
        assert getattr(tsp_solver, "mutate_population", None) is not None

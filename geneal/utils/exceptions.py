class NoFitnessFunction(Exception):
    def __init__(self, *args, **kwargs):
        self.message = args[0] if args else "NoFitnessFunction"

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return self.__class__.__name__


class InvalidInput(Exception):
    def __init__(self, *args, **kwargs):
        self.message = args[0] if args else "InvalidInput"

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return self.__class__.__name__

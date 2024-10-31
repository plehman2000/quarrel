import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell
def __():
    import adalflow
    return (adalflow,)


@app.cell
def __():
    #The optimization requires users to have at least one dataset, an evaluator, and define optimizor to use. This section we will briefly cover the datasets and evaluation metrics supported in the library.
    return


@app.cell
def __():
    from adalflow.core import Component, Generator
    from adalflow.components.model_client import OllamaClient


    input_template = r"""<SYS> You are an agent that returns true or false if a text supports a given claim. Return a JSON indicating if the text supports the claim </SYS> Claim: {{claim}}\nText: {{text}}"""

    class IsInformative(Component):
        def __init__(self):
            super().__init__()
            self.doc = Generator(
                template=input_template,
                model_client=OllamaClient(),
                model_kwargs={"model": "llama3.2"},
            )

        def call(self, text: str, claim : str) -> str:
            return self.doc(prompt_kwargs={"claim": claim,"text": text}).data
    return Component, Generator, IsInformative, OllamaClient, input_template


@app.cell
def __(IsInformative):
    model = IsInformative()
    print(model(text="Fauci likes beaches", claim="Fauci supported vaccines"))
    return (model,)


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

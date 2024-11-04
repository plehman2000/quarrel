import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell
def __():
    import adalflow
    import marimo as mo
    return adalflow, mo


@app.cell
def __():
    #The optimization requires users to have at least one dataset, an evaluator, and define optimizor to use. This section we will briefly cover the datasets and evaluation metrics supported in the library.
    return


@app.cell
def __(mo):
    mo.md(r"""## Model""")
    return


@app.cell
def __():
    from adalflow.core import Component, Generator
    from adalflow.components.model_client import OllamaClient

    MODEL = "llama3.2"
    # MODEL = "dolphin-phi"
    # MODEL = 'dolphin-llama3'

    input_template = r"""<SYS> You are an agent that returns true or false if a text is relevant for a given claim. Only return the word "true" or the word "false" </SYS> Claim: {{claim}}\nText: {{text}}"""

    class IsInformative(Component):
        def __init__(self):
            super().__init__()
            self.doc = Generator(
                template=input_template,
                model_client=OllamaClient(),
                model_kwargs={"model": MODEL},
            )

        def call(self, text: str, claim : str) -> str:
            return self.doc(prompt_kwargs={"claim": claim,"text": text}).data
    return (
        Component,
        Generator,
        IsInformative,
        MODEL,
        OllamaClient,
        input_template,
    )


@app.cell
def __(IsInformative):
    model = IsInformative()
    print(model(text="Fauci always goes to the beach", claim="Fauci hates beaches"))
    return (model,)


@app.cell
def __(mo):
    mo.md(r"""## Data Class and Dataset""")
    return


@app.cell
def __():
    from dataclasses import dataclass, field
    from adalflow.core import DataClass, required_field
    from typing import Dict

    @dataclass
    class IsInformativeInput(DataClass):
        statement: str = field(
            metadata={"desc": "A statement or text chunk to be evaluated"}
        )
        claim: str = field(
            metadata={"desc": "The claim the statement may be relevant for evaluating"}
        )

    @dataclass
    class IsInformativeOutput(DataClass):
        response: bool = field(
            metadata={"desc": "Whether the text is relevant to the claim the claim (true) or not (false)"}
        )

    @dataclass
    class IsInformativeExample(DataClass):
        inputs: IsInformativeInput = field(
            metadata={"desc": "Input containing text and claim"}
        )
        outputs: IsInformativeOutput = field(
            metadata={"desc": "Output containing if text is relevant to claim"}
        )

        @classmethod
        def from_dict(cls, data: Dict) -> "IsInformativeExample":
            # Convert string 'true'/'false' to boolean
            response_str = data["outputs"]["is_supportive"].lower()
            if response_str == "true":
                is_supportive = True
            elif response_str == "false":
                is_supportive = False
            else:
                is_supportive = None
            
            return cls(
                inputs=IsInformativeInput(
                    statement=data["inputs"]["statement"],
                    claim=data["inputs"]["claim"]
                ),
                outputs=IsInformativeOutput(
                    response=is_supportive
                )
            )

    return (
        DataClass,
        Dict,
        IsInformativeExample,
        IsInformativeInput,
        IsInformativeOutput,
        dataclass,
        field,
        required_field,
    )


@app.cell
def __():
    true_examples = [
        {
            "inputs": {
                "statement": "Carbon dioxide levels in the atmosphere are at an all-time high.",
                "claim": "Climate change is real"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Eating fruits and vegetables is associated with better health outcomes.",
                "claim": "Healthy eating promotes wellness"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Vaccination rates have significantly reduced the incidence of diseases like measles.",
                "claim": "Vaccines are effective"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Research indicates that smoking increases the risk of lung cancer.",
                "claim": "Smoking is harmful"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Studies show that sleep deprivation can impair cognitive function.",
                "claim": "Sleep is important for health"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Renewable energy sources like solar and wind are becoming more cost-effective.",
                "claim": "Renewable energy is viable"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Regular physical activity is linked to improved mental health.",
                "claim": "Exercise benefits mental health"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "The ozone layer protects the Earth from harmful ultraviolet radiation.",
                "claim": "The ozone layer is important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Climate models predict continued warming if greenhouse gas emissions are not reduced.",
                "claim": "Climate change is a serious issue"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Public health campaigns have successfully reduced smoking rates.",
                "claim": "Public health initiatives are effective"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Ocean acidification is caused by increased CO2 in the atmosphere.",
                "claim": "Ocean health is affected by climate change"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Mental health disorders can be effectively treated with therapy and medication.",
                "claim": "Mental health care is important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Diets high in sugar can lead to obesity and diabetes.",
                "claim": "Sugar can be harmful"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Wildlife conservation efforts help protect endangered species.",
                "claim": "Conservation is crucial"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Exercise is linked to improved cardiovascular health.",
                "claim": "Exercise is beneficial for the heart"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Electric vehicles produce fewer emissions than traditional gasoline cars.",
                "claim": "Electric cars are better for the environment"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Healthy relationships can enhance emotional well-being.",
                "claim": "Social connections are important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Antibiotics can effectively treat bacterial infections.",
                "claim": "Antibiotics are useful medications"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Rising sea levels threaten coastal communities due to climate change.",
                "claim": "Climate change poses risks to society"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Good nutrition plays a key role in overall health.",
                "claim": "Nutrition is fundamental for health"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Mental health awareness campaigns have increased understanding and support for those affected.",
                "claim": "Mental health awareness is important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Research shows that hobbies can improve life satisfaction.",
                "claim": "Hobbies enhance well-being"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Global temperatures are rising due to human activities.",
                "claim": "Human activity affects climate"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Fruits and vegetables provide essential vitamins and minerals.",
                "claim": "Healthy diets include fruits and vegetables"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Increased exercise can help manage weight effectively.",
                "claim": "Exercise aids in weight management"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "The use of fossil fuels contributes to air pollution.",
                "claim": "Fossil fuels are harmful to air quality"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Proper sanitation and hygiene reduce the spread of infectious diseases.",
                "claim": "Sanitation is vital for public health"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Regular check-ups can lead to early detection of health issues.",
                "claim": "Preventive care is important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Studies show that climate change is affecting wildlife habitats.",
                "claim": "Climate change impacts ecosystems"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Biodiversity is essential for ecosystem health and stability.",
                "claim": "Biodiversity is important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Chronic stress can lead to a variety of health problems.",
                "claim": "Stress management is essential"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Recycling helps reduce waste and conserve resources.",
                "claim": "Recycling is beneficial for the environment"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Access to clean water is critical for health and sanitation.",
                "claim": "Clean water is a basic necessity"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Eating a balanced diet contributes to overall health.",
                "claim": "Balanced diets are important"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Air quality affects respiratory health.",
                "claim": "Air quality is important for health"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Urban green spaces improve mental health and well-being.",
                "claim": "Green spaces are valuable in cities"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Community engagement enhances social cohesion.",
                "claim": "Community involvement is beneficial"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        },
        {
            "inputs": {
                "statement": "Pollinator populations are declining, which threatens food production.",
                "claim": "Pollinators are essential for agriculture"
            },
            "outputs": {
                "is_supportive": 'true'
            }
        }
    ]

    false_examples = [
        {
            "inputs": {
                "statement": "It rained heavily last week in my city.",
                "claim": "Climate change is real"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I had pizza for dinner last night.",
                "claim": "Healthy eating promotes wellness"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My friend jogs every morning.",
                "claim": "Exercise is good for health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "The stars were particularly bright last night.",
                "claim": "Exercise is good for health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I watched a documentary about dolphins yesterday.",
                "claim": "Vaccines are effective"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I forgot to bring my lunch to work today.",
                "claim": "Healthy eating promotes wellness"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "The bakery around the corner makes the best cookies.",
                "claim": "Smoking is harmful"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My cat loves to play with string.",
                "claim": "Sleep is important for health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My neighbor's garden is filled with flowers.",
                "claim": "Renewable energy is viable"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I recently finished reading a novel.",
                "claim": "Public health initiatives are effective"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "The weather forecast predicts sunshine for tomorrow.",
                "claim": "Ocean health is affected by climate change"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I found a $20 bill on the ground yesterday.",
                "claim": "Exercise benefits mental health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My favorite color is blue.",
                "claim": "The ozone layer is important"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I have a collection of stamps from different countries.",
                "claim": "Healthy diets include fruits and vegetables"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I need to wash my car this weekend.",
                "claim": "Climate change is a serious issue"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I have plans to visit a museum next month.",
                "claim": "Public health initiatives are effective"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "The flowers in my garden are blooming beautifully.",
                "claim": "Climate change impacts ecosystems"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I tried a new recipe for dinner.",
                "claim": "Antibiotics are useful medications"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I prefer summer over winter.",
                "claim": "Air quality is important for health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I plan to take a trip next summer.",
                "claim": "Stress management is essential"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My dog barks at the mailman.",
                "claim": "Social connections are important"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I watch TV every evening.",
                "claim": "Renewable energy is viable"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I went for a walk in the park yesterday.",
                "claim": "Vaccines are effective"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I recently bought new shoes.",
                "claim": "Biodiversity is important"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I finished a jigsaw puzzle last weekend.",
                "claim": "Hobbies enhance well-being"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I heard a great song on the radio.",
                "claim": "Community involvement is beneficial"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My friend baked a cake for her birthday.",
                "claim": "Healthy diets include fruits and vegetables"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "The coffee shop on the corner has great pastries.",
                "claim": "Public health initiatives are effective"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I am learning to play the guitar.",
                "claim": "Exercise is beneficial for the heart"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My sister has a collection of postcards.",
                "claim": "Electric cars are better for the environment"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I saw a movie about aliens last night.",
                "claim": "Pollinators are essential for agriculture"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I enjoy taking photographs of nature.",
                "claim": "Mental health care is important"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I got a haircut last week.",
                "claim": "Nutrition is fundamental for health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "My favorite season is autumn.",
                "claim": "Stress management is essential"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I enjoy reading science fiction novels.",
                "claim": "Good nutrition plays a key role in overall health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I went to the beach last weekend.",
                "claim": "Air quality is important for health"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I tried a new coffee blend today.",
                "claim": "Exercise aids in weight management"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        },
        {
            "inputs": {
                "statement": "I watched a documentary about space last night.",
                "claim": "Mental health awareness is important"
            },
            "outputs": {
                "is_supportive": 'false'
            }
        }]


    return false_examples, true_examples


@app.cell
def __(IsInformativeExample, true_examples):
    example = IsInformativeExample.from_dict(true_examples[0])

    return (example,)


@app.cell
def __(example, false_examples, true_examples):
    # Create an instance
    examples = false_examples + true_examples
    print(example)
    return (examples,)


@app.cell
def __(mo):
    mo.md(r"""# Eval""")
    return


@app.cell
def __(OllamaClient):
    from adalflow.eval.llm_as_judge import LLMasJudge, DefaultLLMJudge



    llm_judge = DefaultLLMJudge(model_client=OllamaClient(),
    model_kwargs={"model": "dolphin-llama3"},
    jugement_query="Does the predicted answer means the same as the ground truth answer? Say True if yes, False if no."
    )
    llm_evaluator = LLMasJudge(llm_judge=llm_judge)
    print(llm_judge)

    return DefaultLLMJudge, LLMasJudge, llm_evaluator, llm_judge


if __name__ == "__main__":
    app.run()

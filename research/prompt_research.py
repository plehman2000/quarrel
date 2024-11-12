import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import dspy
    from dspy.evaluate import Evaluate
    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
    # Configure Ollama as the backend
    lm = dspy.OllamaLocal(
        model="dolphin-llama3",
        temperature=0.3,  # Lower temperature for more consistent classification
    )
    dspy.settings.configure(lm=lm)

    return (
        BootstrapFewShot,
        BootstrapFewShotWithRandomSearch,
        Evaluate,
        dspy,
        lm,
    )


@app.cell
def __(dspy):


    # Training examples properly structured for DSPy
    train_data = [
        dspy.Example(
            inputs={
                "statement": "The average global temperature has increased by 1°C since pre-industrial times.",
                "claim": "Climate change is real"
            },
            outputs={
                "is_supportive": '{"response":"true"}'
            }
        ),
        dspy.Example(
            inputs={
                "statement": "My neighbor said it was cold yesterday.",
                "claim": "Climate change is real"
            },
            outputs={
                "is_supportive": '{"response":"false"}'
            }
        ),
        dspy.Example(
            inputs={
                "statement": "Studies show regular exercise reduces the risk of heart disease.",
                "claim": "Exercise is good for health"
            },
            outputs={
                "is_supportive": '{"response":"true"}'
            }
        ),
        dspy.Example(
            inputs={
                "statement": "The sky is blue.",
                "claim": "Exercise is good for health"
            },
            outputs={
                "is_supportive": '{"response":"false"}'
            }
        ),
        # Additional training examples
        dspy.Example(
            inputs={
                "statement": "Carbon emissions are leading to more extreme weather events.",
                "claim": "Climate change is real"
            },
            outputs={
                "is_supportive": '{"response":"true"}'
            }
        ),
        dspy.Example(
            inputs={
                "statement": "It rained yesterday.",
                "claim": "Climate change is real"
            },
            outputs={
                "is_supportive": '{"response":"false"}'
            }
        ),
        dspy.Example(
            inputs={
                "statement": "Eating fruits and vegetables is associated with better health outcomes.",
                "claim": "Healthy eating promotes well-being"
            },
            outputs={
                "is_supportive": '{"response":"true"}'
            }
        ),
        dspy.Example(
            inputs={
                "statement": "The restaurant serves pizza.",
                "claim": "Healthy eating promotes well-being"
            },
            outputs={
                "is_supportive": '{"response":"false"}'
            }
        )
    ]


    return (train_data,)


if __name__ == "__main__":
    app.run()







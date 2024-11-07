import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import ollama
    return (ollama,)


@app.cell
def __(ollama):
    def determine_informative_bespoke(doc, claim):

        prompt = f"""Document: {doc}\n Claim: {claim}"""
        response = ollama.generate(model="bespoke-minicheck", prompt=prompt)
        output = response['response']
        if output == "Yes":
            return {"response" : "true"}
        else:
            return {"response" : "false"}
            

    return (determine_informative_bespoke,)


@app.cell
def __(determine_informative_bespoke):
    document = """Donald Trump vowed to “rescue” the Denver suburb of Aurora, Colorado, from the rapists, “blood thirsty criminals,” and “most violent people on earth” he insists are ruining the “fabric” of the country and its culture: immigrants.

    Trump’s message in Aurora, a city that has become a central part of his campaign speeches in the final stretch to Election Day, marks another example of how the former president has escalated his xenophobic and racist rhetoric against migrants and minority groups he says are genetically predisposed to commit crimes. The supposed threat migrants pose is the core part of the former president’s closing argument, as he promises his base that he’s the one who can save the country from a group of people he calls “animals,” “stone cold killers,” the “worst people,” and the “enemy from within.”"""

    claim = "Donald Trump is Racist"

    r = determine_informative_bespoke(document, claim)
    print(r)
    return claim, document, r


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __():
    from prover import Prover 
    prover = Prover()
    return Prover, prover


@app.cell
def __(prover):
    claim = "The minecraft youtuber Dream is a pedophile"
    out = None
    for x in prover.run(proposition_claim=claim, use_small_model=False):
        out = x
        print(out['status'])
    arg1_w_claims = out['arg1_w_claims']
    arg2_w_claims = out['arg2_w_claims']
    print(arg1_w_claims, arg2_w_claims)
    print(f"Winning Claim: {out['victor']}")
    return arg1_w_claims, arg2_w_claims, claim, out, x


@app.cell
def __(i):
    i
    return


@app.cell
def __(out):
    out # should \grou[ all chunks together, just in case there is one informative source?]
    ####
    #"proposition_query":
    #"Why do you believe that most dogs are friendly?"
    #FUCKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK
    return


app._unparsable_cell(
    r"""
    {
    \"proposition_claim\":
    \"The minecraft youtuber Dream is a pedophile\"
    \"status\":
    \"Complete\"
    \"progress\":
    100
    \"opposition_claim\":
    \"The minecraft youtuber Dream is not a pedophile.\"
    \"proposition_query\":
    \"Why do you believe that most dogs are friendly?\"
    \"opposition_query\":
    \"Why isn't the Minecraft youtuber Dream considered a pedophile?\"
    \"prop_final_args\":
    []
    \"prop_chunk\":
    {
    0:
    []
    1:
    []2:
    []
    }
    \"opp_final_args\":
    []
    \"opp_chunks\":
    {
    0:
    []
    1:
    []2:
    []
    }
    \"arg1_w_claims\":
    \"Claim:The minecraft youtuber Dream is a pedophile\n\"
    \"arg2_w_claims\":
    \"Claim: The minecraft youtuber Dream is not a pedophile.\n\"
    \"victor\":
    \"The minecraft youtuber Dream is not a pedophile.\"
    \"final_judge\":
    {
    \"argument\":
    \"2\"
    }
    }
    """,
    name="__"
)


if __name__ == "__main__":
    app.run()

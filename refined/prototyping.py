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
    oppclaim = "The minecraft youtuber Dream is not a pedophile"
    out = None
    import time
    start_time = time.time()
    for x in prover.run(proposition_claim=claim,opposition_claim = oppclaim, use_small_model=False):
        out = x
        print(out['status'])
        print(time.time() - start_time )
        start_time = time.time()
    arg1_w_claims = out['arg1_w_claims']
    arg2_w_claims = out['arg2_w_claims']
    print(arg1_w_claims, arg2_w_claims)
    print(f"Winning Claim: {out['victor']}")
    return (
        arg1_w_claims,
        arg2_w_claims,
        claim,
        oppclaim,
        out,
        start_time,
        time,
        x,
    )


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


if __name__ == "__main__":
    app.run()

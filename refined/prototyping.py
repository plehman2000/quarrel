import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __():
    from prover import prover_F
    return (prover_F,)


@app.cell
def __(prover_F):
    out = None
    import time
    start_time = time.time()
    for x in prover_F(
    proposition_claim="Donald Trump is racist",
        opposition_claim = "Donald Trump is not racist",
        use_small_model=False, 
        n_websites=6,
        n_chunks_needed_per_cluster=1,
        n_argument_clusters=2
    ):
        out = x
        print(out['status'])
        print(f"Time Take: {time.time() - start_time}" )
        start_time = time.time()
    arg1_w_claims = out['arg1_w_claims']
    arg2_w_claims = out['arg2_w_claims']
    print(arg1_w_claims, "\n" + arg2_w_claims)
    print(f"Winning Claim: {out['victor']}")
    return arg1_w_claims, arg2_w_claims, out, start_time, time, x


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

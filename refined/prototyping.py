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
        proposition_claim="Donald Trump is not racist",
        opposition_claim="Donald Trump is racist",
        use_small_model=False,
        n_websites=40,
        n_chunks_needed_per_cluster=3,
        n_argument_clusters=2,
    ):
        out = x
        print(out["status"])
        print(f"Time Take: {time.time() - start_time}")
        start_time = time.time()
    arg1_w_claims = out["arg1_w_claims"]
    arg2_w_claims = out["arg2_w_claims"]
    print(arg1_w_claims, "\n" + arg2_w_claims)
    print(f"Winning Claim: {out['victor']}")
    return arg1_w_claims, arg2_w_claims, out, start_time, time, x


app._unparsable_cell(
    r"""
    WHAT THE FUCK:{0: ['search_document: . Native American groups criticized him for making derogatory remarks about tribes seeking to build casinos in the 1990s. Trump was also a leading voice of the “birther” conspiracy that baselessly claimed former President Barack Obama was from Africa and not an American citizen', 'search_document: .\n\nTribe leaders at the time called out the remarks as racist. The National Indian Gaming Association filed a Federal Communications Commission complaint after Trump made similar remarks on Don Imus’ talk radio show', 'search_document: .\n\nTrump has a Jewish daughter and grandchildren, yet left Jews out of a Holocaust remembrance statement and referred to one Jewish group as “negotiators.”\n\nHe said an Indiana judge could not rule on a border case because of his Mexican heritage. He funded ads that associated Native Americans with drug use and crime'], 1: ['search_document: .\n\nFirst, Donald Trump’s support in the 2016 campaign was clearly driven by racism, sexism, and xenophobia. While some observers have explained Trump’s success as a result of economic anxiety, the data demonstrate that anti-immigrant sentiment, racism, and sexism are much more strongly related to support for Trump', 'search_document: . But there is no excuse for avoiding clear, accurate descriptions of American political dynamics. When the data show that President Trump’s support stems from racist and sexist beliefs, and that his election emboldened Americans to engage in racist behavior, it is the responsibility of social scientists and other political observers to say so', 'search_document: . & Sherif, C. W. Reference Groups (Harper & Row, 1964).\n\nGoogle Scholar\n\nDesjardins, L. Every moment in Trump’s charged relationship with race. PBS News Hour (22 August 2017).\n\nLeonhardt, D. & Philbrick, I. P. Donald Trump’s racism: the definitive list. The New York Times (15 January 2018).\n\nMendelberg, T']}

    """,
    name="__",
)


if __name__ == "__main__":
    app.run()

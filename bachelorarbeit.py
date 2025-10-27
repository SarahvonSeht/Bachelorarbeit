import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import re
    import os
    from langchain_openai import ChatOpenAI
    return ChatOpenAI, mo, os, pl, re


@app.cell
def _(pl):
    # Die zwei Datensätze Willow und Malls
    # Außerdem ein Testdatensatz und eine kleinere Variante davon

    willow = pl.read_excel("data/WillowNLtoFol_Dataset.xlsx")

    malls = pl.read_json("data/MALLS-v0.1-train.json")

    malls = malls.select(
        pl.col.NL.alias("NL_sentence"),
        pl.col.FOL.alias("FOL_expression")
    )

    dataset = malls

    test = pl.read_excel("data/test.xlsx")

    test_small = pl.read_excel("data/test_small.xlsx")
    return dataset, malls, test, willow


@app.cell
def _(pl, willow):
    # Zur Überprüfung, wie viele NL-Sätze nicht mit einem Punkt enden

    filtered = willow.filter(~pl.col("NL_sentence").str.ends_with("."))

    filtered
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #Bereinigen der Individuenkonstanten:

    - alle kleinschreiben

    - die Anführungsstriche entfernen

    - andere Sonderzeichen (z.B. -) entfernen
    """
    )
    return


@app.cell
def _(dataset, pl, re):
    # Alle kleingeschriebenen Individuenkonstanten finden
    # Zum Vergleich nutzen

    def lowercase_individuals(expr: str) -> str:
        # Funktionsnamen finden (alles vor '(')
        func_names = set(re.findall(r'\b\w+(?=\s*\()', expr))
        # Alle Wörter im Ausdruck
        words = re.findall(r'\b\w+\b', expr)
        # Neue Version mit kleingeschriebenen Individuen
        new_expr = expr
        for w in words:
            if len(w) > 1 and w not in func_names:
                new_expr = re.sub(r'\b' + re.escape(w) + r'\b', w[0].lower() + w[1:], new_expr)
        return new_expr

    df_lowercase = dataset.with_columns(
        pl.col("FOL_expression")
        .map_elements(lowercase_individuals, return_dtype=pl.Utf8)  # Datentyp explizit angeben
        .alias("FOL_modified")
    ).filter(
        pl.col("FOL_expression") != pl.col("FOL_modified")
    )

    df_lowercase
    return


@app.cell
def _(dataset, pl):
    # Alle Individuenkonstanten in Anführungsstrichen finden
    # Zum Vergleich nutzen

    df_quotes = dataset.with_columns(
        pl.col.FOL_expression
        .str.extract_all(r'"[^"]*"|`[^`]*`|´[^´]*´|\'[^\']*\'')
        .alias("quoted_words")
    )

    df_quotes.filter(
        pl.col.quoted_words.list.len() > 0 
    )
    return


@app.cell
def _(dataset, pl):
    # Alle Wörter mit anderen Sonderzeichen finden
    # Zum Vergleich nutzen

    df_punctuated = dataset.with_columns(
        pl.col.FOL_expression
        .str.extract_all(r'\S+')  
        .list.eval(pl.element().filter(pl.element().str.contains(r'[-:;!?]')))
        .alias("punctuated_words")
    )

    df_punctuated.filter(
        pl.col.punctuated_words.list.len() > 0
    )
    return


@app.cell
def _(pl):
    # Alle Sonderzeichen entfernen
    # Inkl. Anführungszeichen

    def clean_fol_expression(dataset: pl.DataFrame) -> pl.DataFrame:

        return dataset.with_columns(
            pl.col.FOL_expression
            .str.replace_all(r"[-:;!?\"'`´]", "") 
            .alias("FOL_expression")
        )
    return (clean_fol_expression,)


@app.cell
def _(pl, re):
    # Alle Individuenkonstanten klein schreiben

    def lowercase_individual_constants(dataset: pl.DataFrame) -> pl.DataFrame:

        # 1. Individuenkonstanten finden
        df_changed = dataset.with_columns(
            pl.col.FOL_expression
            .str.replace_all(r'\b\w+\s*\(', '(')   
            .str.replace_all(r'[^\w]', ' ')           
            .str.extract_all(r'\b\w+')
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 1))
            .alias("extracted_words")
        )


        # 2. Ersten Buchstaben klein schreiben
        df_changed = df_changed.with_columns(
            pl.col.extracted_words.list.eval(
                # schreibt den ersten Buchstaben aller extrahierter Wörter klein
                pl.element().str.slice(0, 1).str.to_lowercase() + pl.element().str.slice(1)
            ).alias("lowercase_words")
        )

        # 3. Python-Listen für Ersetzung
        extracted_words_lists = df_changed["extracted_words"].to_list()
        lowercase_words_lists = df_changed["lowercase_words"].to_list()
        original_expressions = dataset["FOL_expression"].to_list()

        # 4. Ersetzung im Originalstring
        modified_expressions = []
        for original, extracted, lowercase in zip(original_expressions, extracted_words_lists, lowercase_words_lists):
            modified_expression = original
            for old, new in zip(extracted, lowercase):
                modified_expression = re.sub(r'\b' + re.escape(old) + r'\b', new, modified_expression)
            modified_expressions.append(modified_expression)

        # 5. Direkt in der Originalspalte speichern
        df_final = dataset.with_columns(
            pl.Series(modified_expressions).alias("FOL_expression")
        )

        return df_final
    return (lowercase_individual_constants,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Adverbien der Zeit""")
    return


@app.cell
def _(malls, pl):
    # Begriffe, die sich nicht sinnvoll in Logik erster Stufe formalisieren lassen
    # Future Work: dasselbe für Modalverben (z.b. can, must, ...) Die sind aufwendiger zu unterscheiden, weil manche Sätze übersetzbar sind und andere weniger gut


    # willow = 0, malls = 22
    # later: (16)
    # recently: (6)
    # der Rest kommt nicht vor
    adverbien_zeit = ["now", "today", "yesterday", "tomorrow", "later", "soon", "recently"]

    # willow = 0, malls = 17
    # permanently: (3) alle im Prädikatennamen
    # temporarily: (15) 
    adverbien_dauer = ["forever", "briefly", "permanently", "temporarily"]

    # willow = 382, malls = 716
    # often: (42) (452) 
    # usually: (49) (177)
    # sometimes: (1) (26)
    # rarely: (1) (2)
    # occasionally: (0) (3)
    adverbien_frequenz = ["often", "usually", "sometimes", "rarely", "occasionally"]

    einzeln = ["recently"]

    df_contains_keyword = malls.with_columns(
                        pl.col.NL_sentence
                        .str.to_lowercase()
                        .str.extract_all(r'\S+')  # Tokenisierung: alle Wörter extrahieren
                        .list.eval(pl.element().is_in(einzeln))
                        .alias("keyword_matches")
    )

    df_contains_keyword.filter(
        pl.col.keyword_matches.list.any()
    )
    return


@app.cell
def _(pl):
    # Alle Sätze mit den festgelegten Adverbien entfernen

    def remove_sentences_with_adverbs(dataset: pl.DataFrame) -> pl.DataFrame:

        adverbien = [
            "now", "today", "yesterday", "tomorrow", "later", "soon", "recently",
            "forever", "briefly", "permanently", "temporarily", "often", "usually",
            "sometimes", "rarely", "occasionally"
        ]

        df_contains_keywords = dataset.with_columns(
            pl.col.NL_sentence
            .str.to_lowercase()
            .str.extract_all(r'\S+') 
            .list.eval(pl.element().is_in(adverbien))
            .alias("keyword_matches")
        )

        # Zeilen entfernen, die mindestens ein Adverb enthalten
        df_filtered = df_contains_keywords.filter(
            ~pl.col("keyword_matches").list.any()
        )

        # Hilfsspalte wieder entfernen
        return df_filtered.drop("keyword_matches")
    return (remove_sentences_with_adverbs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Datensätze automatisch bereinigen

    - Filtert alle Sätze mit Adverbien heraus

    - Schreibt alle Individuenkonstanten groß

    - Entfernt alle Sonderzeichen
    """
    )
    return


@app.cell
def _(
    clean_fol_expression,
    lowercase_individual_constants,
    remove_sentences_with_adverbs,
    willow,
):
    willow_changed = clean_fol_expression(willow)
    willow_changed = lowercase_individual_constants(willow_changed)
    willow_changed = remove_sentences_with_adverbs(willow_changed)
    willow_changed
    return


@app.cell
def _(
    clean_fol_expression,
    lowercase_individual_constants,
    malls,
    remove_sentences_with_adverbs,
):
    malls_changed = clean_fol_expression(malls)
    malls_changed = lowercase_individual_constants(malls_changed)
    malls_changed = remove_sentences_with_adverbs(malls_changed)
    malls_changed
    return (malls_changed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Erkennung der Fehlergruppen""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 1. Ansatz: Definitionen und Few-Shot-Beispielen getrennt""")
    return


@app.cell
def _(
    SYSTEM_PROMPT_CLASSIFICATION_ALLE,
    error_categories_definition,
    few_shot_examples1,
    llm,
    pl,
):
    def process_entries_classification1(sentence: str, translation: str, error_categories: list[dict], few_shot_examples: list[tuple[str, str, str]] = []) -> str:

        few_shot_prompt = ""

        # Few-shot-Beispiele: (NL, inkorrekte FOL, korrekte FOL, Fehlergruppe)
        for ex_sentence, ex_incorrect_translation, ex_correct_translation, ex_label in few_shot_examples:
            few_shot_prompt += (
                f"Sentence: {ex_sentence}\n"
                f"Incorrect Translation: {ex_incorrect_translation}\n"
                f"Correct Translation: {ex_correct_translation}\n"
                f"Error category: {ex_label}\n\n"
            )

        # Fehlerkategorien auflisten
        category_list = "\n".join(
            [f"- {cat['label']}: {cat['description']}" for cat in error_categories]
        )

        # Prompt für den aktuellen Eintrag
        prompt = (
            few_shot_prompt +
            f"Sentence: {sentence}\n"
            f"Translation: {translation}\n\n"
            f"Based on the NL sentence and FOL expression above, does the expression fall into one of the following error categories?\n\n"
            f"{category_list}\n\n"
        )

        #print("🧠 Prompt for classification:\n", prompt)

        messages = [
            ("system", SYSTEM_PROMPT_CLASSIFICATION_ALLE),
            ("human", prompt),
        ]

        ai_msg = llm.invoke(messages)
        return ai_msg.content.strip()


    # Hier mit zufälligen Daten, später anpassen
    # Dann nur noch den TEST Datensatz mit den wichtigsten Fehlergruppen
    # Und danach für den kompleten Datensatz anwendbar
    def process_dataset_classification1(data: pl.DataFrame, sample_size: int, random_seed: int = 3) -> pl.DataFrame:
        sampled_data = data.sample(sample_size, seed=random_seed)

        error_types = sampled_data.map_rows(
            lambda r: process_entries_classification1(
                sentence = r[0],
                translation = r[1],  
                error_categories = error_categories_definition,
                few_shot_examples = few_shot_examples1
            )
        )

        result = pl.concat((sampled_data, error_types), how="horizontal")
        result = result.rename({"map": "error_type"})

        return result

    # Auskommentiert, damit die Anfrage an das LLM nicht automatisch gestartet wird
    #df_classifikation1 = process_dataset_classification1(test, sample_size=5)
    #df_classifikation1
    return


@app.cell
def _():
    # Definitionen der Fehlergruppen
    error_categories_definition = [
        {
            "label": "missing_category",
            "description": "The FOL formula omits a general category or class that is explicitly mentioned in the natural language sentence. For example, if the sentence refers to 'all birds' and the formula only refers to 'eagles', the broader category 'Bird(x)' is missing."
        },
        {
            "label": "overgeneralization",
            "description": "The FOL formula uses overly broad quantification, such as using a universal quantifier (∀) where an existential quantifier (∃) is appropriate. This often leads to stronger claims than stated in the NL sentence."
        },
        {
            "label": "wrong_variable_binding",
            "description": "Variables are introduced or used in a way that mismatches the intended entities or roles in the NL sentence. For example, mixing up who serves and who is served, or binding a variable to the wrong quantifier."
        },
        {
            "label": "free_variable_error",
            "description": "A variable appears in the formula without being quantified (unbound). Every variable used in a logical expression must be bound by a quantifier (e.g., ∀x or ∃y)."
        },
        {
            "label": "biconditional_instead_implication",
            "description": "A biconditional (↔) is incorrectly used when a one-way implication (→) is required. This changes the logic by implying equivalence where only a conditional relationship exists."
        },
        {
            "label": "conjunction_instead_implication",
            "description": "The formula uses a conjunction (∧) instead of a conditional (→), falsely implying that both parts must always be true, rather than one depending on the other."
        },
        {
            "label": "invalid_xor_usage",
            "description": "The XOR operator (⊕) is used incorrectly, particularly in cases involving more than two options. Correct usage should ensure that only one of the options can be true, and not combinations."
        },
        {
            "label": "missing_parentheses",
            "description": "Parentheses are missing or misplaced, which alters the intended grouping of logical operations. This changes the interpretation of operator precedence and sentence meaning."
        },
        {
            "label": "global_negation_error",
            "description": "The entire logical formula is incorrectly negated, altering the intended truth conditions of the sentence."
        },
        {
            "label": "missing_condition_negation",
            "description": "The antecedent (condition part) of an implication is missing a necessary negation, which alters the intended logic."
        },
        {
            "label": "wrong_negation_scope",
            "description": "Negation is applied incorrectly to both the antecedent and the consequent of an implication. This changes the meaning by denying both sides instead of keeping the correct implication structure."
        },
        {
            "label": "neither_nor_translation_error",
            "description": "The NL structure 'neither ... nor ...' is translated as ¬A ∨ ¬B instead of the correct ¬A ∧ ¬B."
        },
        {
            "label": "predicate_naming_error",
            "description": "The predicates used in the FOL formula are incorrect in meaning. The predicate should capture the essential concept concisely and accurately."
        }
    ]
    return (error_categories_definition,)


@app.cell
def _():
    # Few-Shot-Beispiele mit positiver und negativer Übersetzung + zugehörige Fehlergruppe
    few_shot_examples1 = [
        # missing_category
        (
        "All birds that are not both white and black are eagles.",
        "∀x ((¬(White(x) ∧ Black(x))) → Eagles(x))",
        "∀x ((Bird(x) ∧ ¬(White(x) ∧ Black(x))) → Eagles(x))",
        "missing_category"
        ),

        # quantifier_error
        (
        "A cat chases a mouse, catches it, and then eats it.",
        "∀x ∀y (Cat(x) ∧ Mouse(y) → (Chases(x, y) ∧ Catches(x, y) ∧ Eats(x, y)))",
        "∀x (Cat(x) → ∃y (Mouse(y) ∧ Chases(x, y) ∧ Catches(x, y) ∧ Eats(x, y)))",
        "overgeneralization"
        ),

        # wrong_variable_binding
        (
        "Chefs prepare meals for customers, and waiters serve them.",
        "∀x ∀y ∀z (Chef(x) ∧ Customer(y) ∧ Waiter(z) → PreparesMeal(x, y) ∧ Serves(z, x))",
        "∀x (Chef(x) → ∃y (Customer(y) ∧ PreparesMeal(x, y))) ∧ ∀z (Waiter(z) → ∃y (Customer(y) ∧ Serves(z, y)))",
        "wrong_variable_binding"
        ),

        # free_variable_error
        (
        "Drones are used for aerial photography, surveillance, and package delivery.",
        "∀x (Drone(x) → (UsedFor(y) ∧ (AerialPhotography(y) ∨ Surveillance(y) ∨ PackageDelivery(y) ∧ In(x, y))))",
        "∀x (Drone(x) → ∃y (UsedFor(x, y) ∧ (AerialPhotography(y) ∨ Surveillance(y) ∨ PackageDelivery(y))))",
        "free_variable_error"
        ),

        # biconditional_instead_implication
        (
        "An entity is a heavy cube only if it’s not yellow.",
        "∀v (Heavy(v) ∧ Cube(v) ↔ ¬Yellow(v))",
        "∀v (Heavy(v) ∧ Cube(v) → ¬Yellow(v))",
        "biconditional_instead_implication"
        ),

        # conjunction_instead_implication
        (
        "All kittens are not fierce or mean.",
        "∀x (Kitten(x) ∧ (¬Fierce(x) ∨ ¬Mean(x)))",
        "∀x (Kitten(x) → (¬Fierce(x) ∨ ¬Mean(x)))",
        "conjunction_instead_implication"
        ),

        # invalid_xor_usage
        (
        "A movie can be awarded for best cinematography, best original score, or best costume design, but not all three simultaneously.",
        "∀x (Movie(x) ∧ Awarded(x) → (BestCinematographyAward(x) ⊕ BestOriginalScoreAward(x) ⊕ BestCostumeDesignAward(x)))",
        "∀x (Movie(x) ∧ Awarded(x) → ((BestCinematographyAward(x) ∧ ¬BestOriginalScoreAward(x) ∧ ¬BestCostumeDesignAward(x)) ∨ (¬BestCinematographyAward(x) ∧ BestOriginalScoreAward(x) ∧ ¬BestCostumeDesignAward(x)) ∨ (¬BestCinematographyAward(x) ∧ ¬BestOriginalScoreAward(x) ∧ BestCostumeDesignAward(x))))",
        "invalid_xor_usage"
        ),

        # missing_parentheses
        (
        "A person is a musician if and only if they play an instrument or sing, but they do not dissonance.",
        "∀x (Person(x) ∧ Musician(x) ↔ (PlayInstrument(x) ∨ Sing(x) ∧ ¬Dissonance(x)))",
        "∀x (Person(x) ∧ Musician(x) ↔ ((PlayInstrument(x) ∨ Sing(x)) ∧ ¬Dissonance(x)))",
        "missing_parentheses"
        ),

        # global_negation_error
        (
        "If all humans admire John then there are people who do not respect Emma.",
        "¬∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))",
        "∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))",
        "global_negation_error"
        ),

        # missing_condition_negation
        (
        "Unless a country is either poor or rich, it is a developed country.",
        "∀v (Country(x) ∧ (Poor(v) ∨ Rich(v)) → Developed(v))",
        "∀v (Country(x) ∧ ¬(Poor(v) ∨ Rich(v)) → Developed(v))",
        "missing_condition_negation"
        ),

        # wrong_negation_scope
        (
        "All careful persons are alive.",
        "∀x (Person(x) ∧ ¬Careful(x) → ¬Alive(x))",
        "∀x (Person(x) ∧ Careful(x) → Alive(x))",
        "wrong_negation_scope"
        ),

        # neither_nor_translation_error
        (
        "If a house is neither big nor small, it’s affordable.",
        "∀x (House(x) ∧ (¬Big(x) ∨ ¬Small(x)) → Affordable(x))",
        "∀x (House(x) ∧ (¬Big(x) ∧ ¬Small(x)) → Affordable(x))",
        "neither_nor_translation_error"
        ),

        # predicate_naming_error
        (
        "Every individual either studies mathematics or enjoys painting, but not both.",
        "∀x (Individuel(x) → Mathematics(x) ⊕ EnjoyPainting(x))",
        "∀x (Individuel(x) → StudiesMathematics(x) ⊕ EnjoyPainting(x))",
        "predicate_naming_error"
        ),
    ]
    return (few_shot_examples1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 2. Ansatz: Definitionen und Few-Shot-Beispielen immer zusammen""")
    return


@app.cell
def _(SYSTEM_PROMPT_CLASSIFICATION_ALLE, error_categories, llm, pl):
    def format_error_categories_text(categories):
        text = ""
        for c in categories:
            text += f"Error label: {c['label']}\n"
            text += f"Instruction: {c['instruction']}\n"
            text += "Positive example (correct):\n"
            text += f"  Sentence: {c['positive_example']['sentence']}\n"
            text += f"  Formula: {c['positive_example']['formula']}\n"
            text += "Negative example (error):\n"
            text += f"  Sentence: {c['negative_example']['sentence']}\n"
            text += f"  Formula: {c['negative_example']['formula']}\n"
            # Erklärung optional prüfen (falls evtl. mal nicht vorhanden)
            explanation = c['negative_example'].get("explanation", "")
            if explanation:
                text += f"  Explanation: {explanation}\n"
            text += "\n"
        return text


    def process_entries_classification2(sentence: str, translation: str, error_categories: list[dict]) -> str:

        error_text = format_error_categories_text(error_categories)

        prompt = (
            "Below are descriptions of common logical error categories. Each includes:\n"
            "- An instruction that defines the error\n"
            "- A positive (correct) example\n"
            "- A negative (incorrect) example and explanation\n\n"
            f"{error_text}"
            "------------------------------\n"
            f"Sentence:\n{sentence}\n\n"
            f"Formula:\n{translation}\n\n"
        )

        messages = [
            ("system", SYSTEM_PROMPT_CLASSIFICATION_ALLE),
            ("human", prompt),
        ]

        ai_msg = llm.invoke(messages)
        return ai_msg.content.strip()


    def process_dataset_classification2(data: pl.DataFrame, sample_size: int, random_seed: int = 3) -> pl.DataFrame:
        sampled_data = data.sample(sample_size, seed=random_seed)

        error_types = sampled_data.map_rows(
            lambda r: process_entries_classification2(
                sentence = r[0],
                translation = r[1],  
                error_categories = error_categories,
            )
        )

        result = pl.concat((sampled_data, error_types), how="horizontal")
        result = result.rename({"map": "error_type"})

        return result


    # Auskommentiert, damit die Anfrage an das LLM nicht automatisch gestartet wird
    #df_classifikation2 = process_dataset_classification2(test, sample_size=5)
    #df_classifikation2
    return


@app.cell
def _():
    # Definitionen der Fehlergruppen und zu jeder positive und negative Beispiele
    error_categories = [
        {
            "label": "missing_category",
            "instruction": "The FOL formula omits a general category explicitly mentioned in the sentence.",
            "positive_example": {
                "sentence": "All birds that are not both white and black are eagles.",
                "formula": "∀x ((Bird(x) ∧ ¬(White(x) ∧ Black(x))) → Eagles(x))"
            },
            "negative_example": {
                "sentence": "All birds that are not both white and black are eagles.",
                "formula": "∀x ((¬(White(x) ∧ Black(x))) → Eagles(x))",
                "explanation": "The formula only mentions 'Eagle', missing the general 'Bird' category."
            }
        },
        {
            "label": "overgeneralization",
            "instruction": "The formula uses ∀ where ∃ is appropriate, overstating the claim. It only happens when more than one ∀ is used",
            "positive_example": {
                "sentence": "A cat chases a mouse, catches it, and then eats it.",
                "formula": "∀x (Cat(x) → ∃y (Mouse(y) ∧ Chases(x, y) ∧ Catches(x, y) ∧ Eats(x, y)))"
            },
            "negative_example": {
                "sentence": "A cat chases a mouse, catches it, and then eats it.",
                "formula": "∀x ∀y (Cat(x) ∧ Mouse(y) → (Chases(x, y) ∧ Catches(x, y) ∧ Eats(x, y)))",
                "explanation": "The formula incorrectly states that every cat chases, catches, and eats every mouse. This overgeneralizes the original sentence, which allows for each cat to interact with only one (or some) mouse."
            }
        },
        {
            "label": "wrong_variable_binding",
            "instruction": "Variables are incorrectly assigned to roles, reversing or distorting who performs the action and who receives it.",
            "positive_example": {
                "sentence": "Every waiter serves a customer.",
                "formula": "∀x (Waiter(x) → ∃y (Customer(y) ∧ Serves(x, y)))"
            },
            "negative_example": {
                "sentence": "Every waiter serves a customer.",
                "formula": "∀x (Waiter(x) → ∃y (Customer(y) ∧ Serves(y, x)))",
                "explanation": "The formula incorrectly binds the variables: it states that the customer serves the waiter (Serves(y, x)) instead of the other way around."
            }
        },
        {
            "label": "free_variable_error",
            "instruction": "A variable is used without being quantified (e.g., ∀x or ∃x). Applies only to variables represented by single letters.",
            "positive_example": {
                "sentence": "Some dog barks.",
                "formula": "∃x (Dog(x) ∧ Barks(x))"
            },
            "negative_example": {
                "sentence": "Some dog barks.",
                "formula": "Dog(x) ∧ Barks(x)",
                "explanation": "The variable x appears without a quantifier. It is used freely without being introduced properly."
            }
        },
        {
            "label": "biconditional_instead_implication",
            "instruction": "A biconditional (↔) is used where only a one-way implication (→) is correct.",
            "positive_example": {
                "sentence": "If it rains, the ground gets wet.",
                "formula": "Rain(x) → Wet(x)"
            },
            "negative_example": {
                "sentence": "If it rains, the ground gets wet.",
                "formula": "Rain(x) ↔ Wet(x)",
                "explanation": "The formula incorrectly implies that the ground is wet if and only if it rains, which overstates the one-way relationship in the sentence."
            }
        },
        {
            "label": "conjunction_instead_implication",
            "instruction": "A conjunction (∧) is used instead of a conditional (→), changing the meaning to mutual truth instead of dependency.",
            "positive_example": {
                "sentence": "If a person is a doctor, then they went to medical school.",
                "formula": "∀x (Doctor(x) → WentToMedSchool(x))"
            },
            "negative_example": {
                "sentence": "If a person is a doctor, then they went to medical school.",
                "formula": "∀x (Doctor(x) ∧ WentToMedSchool(x))",
                "explanation": "The formula states that both parts must always be true, rather than expressing a conditional relationship."
            }
        },
        {
            "label": "invalid_xor_usage",
            "instruction": "The XOR operator is misused with more than two operands; it should only be applied between two variables at a time.",
            "positive_example": {
                "sentence": "A person can be a student, a teacher, or a researcher — but only one of the three.",
                "formula": "∀x (Person(x) → ((Student(x) ∧ ¬Teacher(x) ∧ ¬Researcher(x)) ∨ (¬Student(x) ∧ Teacher(x) ∧ ¬Researcher(x)) ∨ (¬Student(x) ∧ ¬Teacher(x) ∧ Researcher(x))))"
            },
            "negative_example": {
                "sentence": "A person can be a student, a teacher, or a researcher — but only one of the three.",
                "formula": "∀x (Student(x) ⊕ Teacher(x) ⊕ Researcher(x))",
                "explanation": "The formula incorrectly chains the XOR operator across three predicates. XOR is only well-defined for exactly two operands. For three or more mutually exclusive options, the logic must be expressed using conjunctions and disjunctions of negated combinations."
            }
        },
        {
            "label": "missing_parentheses",
            "instruction": "Parentheses are missing or misplaced, changing the intended grouping of logical operations. This alters how logical operators like ∧, ∨, and ¬ interact.",
            "positive_example": {
                "sentence": "A person is a musician if and only if they play an instrument or sing, but they do not dissonance.",
                "formula": "∀x (Person(x) ∧ Musician(x) ↔ ((PlayInstrument(x) ∨ Sing(x)) ∧ ¬Dissonance(x)))"
            },
            "negative_example": {
                "sentence": "A person is a musician if and only if they play an instrument or sing, but they do not dissonance.",
                "formula": "∀x (Person(x) ∧ Musician(x) ↔ (PlayInstrument(x) ∨ Sing(x) ∧ ¬Dissonance(x)))",
                "explanation": "The formula lacks parentheses around the disjunction. Due to operator precedence, the formula evaluates 'Sing(x) ∧ ¬Dissonance(x)' first, then disjoins that with 'PlayInstrument(x)', resulting in a different logical meaning than intended."
            }
        },
        {
            "label": "global_negation_error",
            "instruction": "The entire formula is unnecessarily negated, inverting its meaning.",
            "positive_example": {
                "sentence": "If all humans admire John then there are people who do not respect Emma.",
                "formula": "∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))"
            },
            "negative_example": {
                "sentence": "If all humans admire John then there are people who do not respect Emma.",
                "formula": "¬∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))",
                "explanation": "Negating the entire formula changes its meaning, incorrectly stating that it is not true that if all humans admire John then some do not respect Emma. This reverses the original implication."
            }
        },
        {
            "label": "missing_condition_negation",
            "instruction": "A necessary negation in the condition part (antecedent) of an implication is missing.",
            "positive_example": {
                "sentence": "Unless a country is either poor or rich, it is a developed country.",
                "formula": "∀x (Country(x) ∧ ¬(Poor(x) ∨ Rich(x)) → Developed(x))"
            },
            "negative_example": {
                "sentence": "Unless a country is either poor or rich, it is a developed country.",
                "formula": "∀v (Country(x) ∧ (Poor(x) ∨ Rich(x)) → Developed(x))",
                "explanation": "The condition is missing the negation. This formula incorrectly implies that countries that are poor or rich are developed, which contradicts the original sentence."
            }
        },
        {
            "label": "wrong_negation_scope",
            "instruction": "Negation is applied to both the condition and the conclusion of an implication, incorrectly reversing the intended logical relationship.",
            "positive_example": {
                "sentence": "All careful persons are alive.",
                "formula": "∀x (Person(x) ∧ Careful(x) → Alive(x))"
            },
            "negative_example": {
                "sentence": "All careful persons are alive.",
                "formula": "∀x (Person(x) ∧ ¬Careful(x) → ¬Alive(x))",
                "explanation": "This formula incorrectly implies that not being careful means not being alive. The original sentence only states that if someone is careful, they are alive — not the reverse."
            }
        },
        {
            "label": "neither_nor_translation_error",
            "instruction": "‘Neither A nor B’ is incorrectly translated using disjunction (∨) instead of conjunction (∧) of negations.",
            "positive_example": {
                "sentence": "Neither the teacher nor the student laughed.",
                "formula": "¬Laughed(Teacher) ∧ ¬Laughed(Student)"
            },
            "negative_example": {
                "sentence": "Neither the teacher nor the student laughed.",
                "formula": "¬Laughed(Teacher) ∨ ¬Laughed(Student)",
                "explanation": "The formula uses a disjunction, which allows one of them to have laughed. The sentence says neither did — so both must not have laughed."
            }
        },
        {
            "label": "predicate_naming_error",
            "instruction": "The predicate names do not accurately reflect the meaning or action described in the natural language sentence.",
            "positive_example": {
                "sentence": "Every individual either studies mathematics or enjoys painting, but not both.",
                "formula": "∀x (Individuel(x) → StudiesMathematics(x) ⊕ EnjoyPainting(x))"
            },
            "negative_example": {
                "sentence": "Every individual either studies mathematics or enjoys painting, but not both.",
                "formula": "∀x (Individuel(x) → Mathematics(x) ⊕ EnjoyPainting(x))",
                "explanation": "The predicate 'Mathematics(x)' names a concept (the subject) instead of the activity. It should describe the individual's action, such as 'StudiesMathematics(x)'."
            }
        }
    ]
    return (error_categories,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3. Ansatz: Jede Fehlergruppe einzeln + mehrere Few-Shot-Beispielen""")
    return


@app.cell
def _(SYSTEM_PROMPT_CLASSIFICATION_EINZELN, llm, pl):
    def process_entries_classification3(sentence: str, translation: str, few_shot_prompt: str) -> str:

        prompt = (
            f"{few_shot_prompt}"
            f"Example:\n"
            f"Sentence: {sentence}\n"
            f"FOL: {translation}\n"
            f"Answer:"

        )

        #print("🧠 Prompt for classification:\n", prompt)

        messages = [
            ("system", SYSTEM_PROMPT_CLASSIFICATION_EINZELN),
            ("human", prompt),
        ]

        ai_msg = llm.invoke(messages)
        return ai_msg.content.strip()


    def process_dataset_classification3(data: pl.DataFrame, few_shot_prompt: str) -> pl.DataFrame:

        error_types = data.map_rows(
            lambda r: process_entries_classification3(
                sentence = r[0],
                translation = r[1],  
                few_shot_prompt = few_shot_prompt,
            )
        )

        result = pl.concat((data, error_types), how="horizontal")
        result = result.rename({"map": "error_type"})

        return result


    def process_dataset_classificationsample3(data: pl.DataFrame, few_shot_prompt: str, sample_size: int, random_seed: int = 42) -> pl.DataFrame:
        sampled_data = data.sample(sample_size, seed=random_seed)

        error_types = sampled_data.map_rows(
            lambda r: process_entries_classification3(
                sentence = r[0],
                translation = r[1],  
                few_shot_prompt = few_shot_prompt,
            )
        )

        result = pl.concat((sampled_data, error_types), how="horizontal")
        result = result.rename({"map": "error_type"})

        return result
    return


@app.cell
def _(pl):
    # Die bereinigten Datensätze anpassen, sodass zur Evaluierung nochmal die Fehlererkennung laufen kann

    willow_cleaned = pl.read_excel("data/willow_cleaned.xlsx")

    willow_cleaned = (willow_cleaned
        .drop(["FOL_expression", "is_equal"])   # löscht die beiden Spalten
        .rename({"FOL_LLM": "FOL_expression"})  # benennt FOL_LLM in FOL_expression um
    )

    malls_cleaned = pl.read_excel("data/malls_cleaned.xlsx")

    malls_cleaned = (malls_cleaned
        .drop(["FOL_expression", "is_equal"])   # löscht die beiden Spalten
        .rename({"FOL_LLM": "FOL_expression"})  # benennt FOL_LLM in FOL_expression um
    )
    return malls_cleaned, willow_cleaned


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fehlende Kategorisierung

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _():
    #df_category = process_dataset_classification3(test, category_prompt).with_row_index("row", 1)
    #df_category
    return


@app.cell
def _():
    category_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL).
    Your task is to check whether the following error category applies.

    Error category: The FOL formula fails to include the single main general category or superclass that the natural language sentence is about.

    Decision process:
    1. Identify the main category in the NL sentence:
       - This is the broadest noun phrase that denotes the group, entity, or objects the sentence is about.
       - All other properties, subclasses, or conditions in the sentence apply to this category.
       - Do not choose a property, condition, or narrower subclass unless no broader category is given.
       - If multiple candidates exist, choose the broadest one explicitly mentioned.
    2. Check if a predicate with this category name (or a close variant) appears in the FOL formula.
    3. If the category is missing, output: "error category does apply."
    4. If the category is present, output: "error category does not apply."

    Examples:

    Sentence: All birds that are not both white and black are eagles.
    FOL: ∀x ((¬(White(x) ∧ Black(x))) → Eagles(x))
    Answer: error category does apply

    Sentence: A musician plays an instrument and performs in a concert.
    FOL: ∀x∀y∀z (Musician(x) ∧ Instrument(y) ∧ Concert(z) → Plays(x, y) ∧ PerformsIn(x, z))
    Answer: error category does not apply

    Sentence: A worker can be a singer, or appreciate Da Vinci's sketches, but not both.
    FOL: ∀x (Singer(x) ⊕ AppreciateDaVinci(x))
    Answer: error category does apply

    Sentence: If an item is either black or purple, it is a square.
    FOL: ∀x ((Black(x) ∨ Purple(x)) → Square(x))
    Answer: error category does apply

    Sentence: Only writers write.
    FOL: ∀v (¬Writer(v) → ¬Write(v))
    Answer: error category does not apply

    Sentence: All dogs bark.
    FOL: ∀x (Dog(x) → Bark(x))
    Answer: error category does not apply

    Sentence: All cats chase mice.
    FOL: ∀x∀y (Chase(x,y) ∧ Mouse(y))
    Answer: error category does apply

    Sentence: Some tall students are athletes.
    FOL: ∃x (Tall(x) ∧ Student(x) ∧ Athlete(x))
    Answer: error category does not apply

    Sentence: If a cat chases a mouse, it is hungry.
    FOL: ∀x∀y (Cat(x) ∧ Mouse(y) ∧ Chases(x,y) → Hungry(x))
    Answer: error category does not apply

    Sentence: A teacher grades a student.
    FOL: ∀x∀y (Teacher(x) ∧ Student(y) → Grades(x,y))
    Answer: error category does not apply

    Sentence: There exists a doctor who treats patients.
    FOL: ∃x∃y (Treats(x,y) ∧ Patient(y))
    Answer: error category does apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Verallgemeinerung

    FUNKTIONIERT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert nur die FOL Übersetzungen heraus, die min zwei ∀ beinhalten
    def overgeneralization_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.count_matches("∀") >= 2
        )
    return (overgeneralization_filtered,)


@app.cell
def _(malls_cleaned, overgeneralization_filtered):
    df_overgeneralization_filtered = overgeneralization_filtered(malls_cleaned)
    df_overgeneralization_filtered
    return


@app.cell
def _():
    # FERTIG
    # willow_changed: 2.614
    #df_willow_overgeneralization = process_dataset_classification3(overgeneralization_filtered(willow_changed), overgeneralization_prompt)

    # Als Excel speichern
    #df_willow_overgeneralization.to_pandas().to_excel("data/willow_overgeneralization.xlsx", index=False)

    #df_willow_overgeneralization
    return


@app.cell
def _():
    # FERTIG
    # malls_changed: 7.880
    #df_malls_overgeneralization = process_dataset_classification3(overgeneralization_filtered(malls_changed), overgeneralization_prompt)

    # Als Excel speichern
    #df_malls_overgeneralization.to_pandas().to_excel("data/malls_overgeneralization.xlsx", index=False)

    #df_malls_overgeneralization
    return


@app.cell
def _():
    # FERTIG
    # willow_cleaned: 2.644
    #df_willow_cleaned_overgeneralization = process_dataset_classification3(overgeneralization_filtered(willow_cleaned), overgeneralization_prompt)

    # Als Excel speichern
    #df_willow_cleaned_overgeneralization.to_pandas().to_excel("data/willow_cleaned_overgeneralization.xlsx", index=False)

    #df_willow_cleaned_overgeneralization
    return


@app.cell
def _():
    # FERTIG
    # malls_cleaned: 3.308
    #df_malls_cleaned_overgeneralization = process_dataset_classification3(overgeneralization_filtered(malls_cleaned), overgeneralization_prompt)

    # Als Excel speichern
    #df_malls_cleaned_overgeneralization.to_pandas().to_excel("data/malls_cleaned_overgeneralization.xlsx", index=False)

    #df_malls_cleaned_overgeneralization
    return


@app.cell
def _():
    overgeneralization_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: The FOL formula uses overly broad quantification by including more than one universal quantifier (∀) over different variables, creating an unintended “for all … and for all …” scope.

    How to decide:
    1. Check if the FOL formula contains two or more universal quantifiers (∀). If not, answer “error category does not apply.” 
    2. Does the NL sentence license universal pairing? Answer “error category does not apply” if the NL sentence clearly intends the statement to range over all cross-pairs of the relevant sets, for example:
    - Universal negatives: “No/none of … (ever) R … / any …”
    - Double universals: “Every … (verb) every …” / “for all X … for all Y …”
    - Equivalent phrasings that deny any pair or assert all pairs.
    3. If the NL sentence does not license universal pairing (e.g., it uses some, at least one, or leaves the second role unspecified), but the FOL uses ∀∀ to quantify all pairs, answer “error category does apply.”
    4. Ignore: argument order/identity, predicate naming, and other logical issues (operators, scope accuracy beyond the ∀∀ vs. NL intent).

    Example:
    Sentence: An elevator transports people or goods between floors of a building.
    FOL: ∀x∀y (Elevator(x) ∧ Building(y) → TransportsBetweenFloors(x, y))
    Answer: error category does apply

    Example:
    Sentence: Loving parents do not neglect their children.
    FOL: ∀x (Parent(x) ∧ Loving(x) → ¬Neglect(x, child))
    Answer: error category does not apply

    Example:
    Sentence: A musician plays an instrument and performs in a concert.
    FOL: ∀x∀y∀z (Musician(x) ∧ Instrument(y) ∧ Concert(z) → Plays(x, y) ∧ PerformsIn(x, z))
    Answer: error category does apply

    Example:
    Sentence: A scientist studies cells using a microscope.
    FOL: ∀x ∀y (Scientist(x) ∧ Microscope(y) → StudiesCellsWith(x, y))
    Answer: error category does apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Falsche Variablenzuordnung

    FUNKTIONIERT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert alle Übersetzungen heraus, deren FOL-Formel ein Komma enthält
    def wrong_binding_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.contains(",")
        )
    return (wrong_binding_filtered,)


@app.cell
def _(test, wrong_binding_filtered):
    df_wrong_binding_filtered = wrong_binding_filtered(test)
    df_wrong_binding_filtered
    return


@app.cell
def _():
    # FERTIG
    # willow_changed: 5.214
    #df_willow_wrong_binding = process_dataset_classification3(wrong_binding_filtered(willow_changed), wrong_binding_prompt)

    # Als Excel speichern
    #df_willow_wrong_binding.to_pandas().to_excel("data/willow_wrong_binding.xlsx", index=False)

    #df_willow_wrong_binding
    return


@app.cell
def _():
    # FERTIG
    # malls_changed: 7.660
    #df_malls_wrong_binding = process_dataset_classification3(wrong_binding_filtered(malls_changed), wrong_binding_prompt)

    # Als Excel speichern
    #df_malls_wrong_binding.to_pandas().to_excel("data/malls_wrong_binding.xlsx", index=False)

    #df_malls_wrong_binding
    return


@app.cell
def _():
    # FERTIG
    # willow_cleaned: 5.425
    #df_willow_cleaned_wrong_binding = process_dataset_classification3(wrong_binding_filtered(willow_cleaned), wrong_binding_prompt)

    # Als Excel speichern
    #df_willow_cleaned_wrong_binding.to_pandas().to_excel("data/willow_cleaned_wrong_binding.xlsx", index=False)

    #df_willow_cleaned_wrong_binding
    return


@app.cell
def _():
    # FERTIG
    # malls_cleaned: 9.359
    #df_malls_cleaned_wrong_binding = process_dataset_classification3(wrong_binding_filtered(malls_cleaned), wrong_binding_prompt)

    # Als Excel speichern
    #df_malls_cleaned_wrong_binding.to_pandas().to_excel("data/malls_cleaned_wrong_binding.xlsx", index=False)

    #df_malls_cleaned_wrong_binding
    return


@app.cell
def _():
    wrong_binding_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: Variables or constants are incorrectly assigned to argument positions in predicates, reversing or distorting the intended roles.

    How to decide:
    1. Identify exactly the roles of the entities in the natural language sentence (who performs the action, who receives it, who is where) using only the literal, direct reading of the sentence.
    2. Examine each predicate and its arguments in the FOL formula.
    3. Compare the variables or constants in each argument position with the roles identified in step 1.
    4. Answer "error category does apply" if:
    - the order of arguments does not match the intended roles, or
    - the identity of arguments does not match the entities in those roles.
    5. If the sentence contains ambiguous pronouns whose reference cannot be clearly determined from the sentence, answer "error category does apply." Do not treat clearly referable pronouns (e.g. “them,” “it,” “they”) as ambiguous.
    6. Ignore all other aspects such as quantifiers, logical operators, missing predicates, or extra details.
    7. If any predicate has mismatched argument positions or identities according to these rules, answer “error category does apply.” Otherwise, answer “error category does not apply.”


    Example:
    Sentence: An elevator transports people or goods between floors of a building.
    FOL: ∀x∀y (Elevator(x) ∧ Building(y) → TransportsBetweenFloors(x, y))
    Answer: error category does not apply

    Example:
    Sentence: Loving parents do not neglect their children.
    FOL: ∀x (Parent(x) ∧ Loving(x) → ¬Neglect(x, child))
    Answer: error category does not apply

    Example:
    Sentence: A musician plays an instrument and performs in a concert.
    FOL: ∀x∀y∀z (Musician(x) ∧ Instrument(y) ∧ Concert(z) → Plays(y, x) ∧ PerformsIn(x, z))
    Answer: error category does apply

    Example:
    Sentence: A scientist studies cells using a microscope.
    FOL: ∀x ∀y (Scientist(x) ∧ Microscope(y) → StudiesCellsWith(y, x))
    Answer: error category does apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Freie Variablen

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert nach allen Übersetzungen, dessen FOL-Formel mindestens eine Variable beinhalten
    def free_variable_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            (pl.col.FOL_expression.str.contains(r'\b[a-z]\b'))
        )
    return (free_variable_filtered,)


@app.cell
def _(free_variable_filtered, malls_changed):
    # Größe der Datensätze nach dem filtern
    # willow_changed: 15.421
    # malls_changed: 26.419

    df_free_variable_filtered = free_variable_filtered(malls_changed)
    df_free_variable_filtered
    return


@app.cell
def _():
    free_variable_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: A variable appears in the FOL formula without being introduced by a quantifier like ∀x or ∃x.

    How to decide:
    1. Consider only single lowercase letters as variables (e.g., x, y, z) following standard FOL naming conventions.
    2. Do not apply this category to:
    - predicate names
    - constants (e.g., lowercase words like alice or uppercase identifiers like Alice)
    - function terms (e.g., f(x)) — only check the variables themselves
    3. For each variable, check whether it is within the scope of a quantifier that binds it. A variable is bound if it appears after and within the scope of a corresponding ∀ or ∃ quantifier anywhere in the formula.
    4. If any variable appears in a predicate or logical expression without being bound by a quantifier anywhere in the formula, answer “error category does apply.”
    5. If all variables are properly bound according to their scopes, answer “error category does not apply.”
    6. Ignore all other aspects such as the number of quantifiers, predicates, or logical operators.

    Example:
    Sentence: Fossil fuels, such as coal, oil, and natural gas, release carbon dioxide when burned, contributing to climate change and global warming.
    FOL: ∀x (FossilFuel(x) ∧ Coal(c) ∧ Oil(o) ∧ NaturalGas(g) ∧ CarbonDioxide(d) ∧ Burn(b) → ContributesToClimateChangeAndGlobalWarming(x, c, o, g, d, b))
    Answer: error category does apply

    Example:
    Sentence: Telescopes use lenses or mirrors to observe distant objects.
    FOL: ∀x (Telescope(x) → (Use(y) ∧ Lenses(y) ∨ Mirrors(y) ∧ ToObserve(z) ∧ DistantObjects(z) ∧ With(x, y, z)))
    Answer: error category does apply

    Example:
    Sentence: Drones are used for aerial photography, surveillance, and package delivery.
    FOL: ∀x ∀y (Drone(x) → (UsedFor(y) ∧ AerialPhotography(y) ∨ Surveillance(y) ∨ PackageDelivery(y) ∧ In(x, y)))
    Answer: error category does not apply

    Example:
    Sentence: A pentagon is not orange and does not point to any blue object.
    FOL: ∀x (Pentagon(x) → (¬Orange(x) ∧ ¬∃y (Blue(y) ∧ PointsTo(x, y))))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Biimplikation statt Implikation

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert alle Übersetzungen heraus, bei denen die FOL-Formel das Biimplikationssymbol enthält
    def biconditional_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.contains("↔")
        )
    return (biconditional_filtered,)


@app.cell
def _(biconditional_filtered, malls_changed):
    df_biconditional_filtered = biconditional_filtered(malls_changed)
    df_biconditional_filtered
    return


@app.cell
def _():
    # IST FEHLERHAFT!

    # willow_changed: 1.176
    #df_willow_biconditional = process_dataset_classification3(biconditional_filtered(willow_changed), biconditional_prompt)

    # Als Excel speichern
    #df_willow_biconditional.to_pandas().to_excel("data/willow_biconditional.xlsx", index=False)

    #df_willow_biconditional
    return


@app.cell
def _():
    # IST FEHLERHAFT!

    # malls_changed: 1.707
    #df_malls_biconditional = process_dataset_classification3(biconditional_filtered(malls_changed), biconditional_prompt)

    # Als Excel speichern
    #df_malls_biconditional.to_pandas().to_excel("data/malls_biconditional.xlsx", index=False)

    #df_malls_biconditional
    return


@app.cell
def _():
    biconditional_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: A biconditional (↔) is used where only a one-way implication (→) is correct.

    How to decide:
    1. Identify the intended logical relationship in the natural language sentence.
    2. Check the FOL formula for the use of a biconditional (↔). If not, answer "error category does not apply" immediatly.
    3. If the NL sentence expresses a one-way conditional (e.g., “If A, then B”) rather than a true equivalence (“A if and only if B”) answer "error category does apply"
    4. Do not apply this category if the sentence actually implies mutual necessity (e.g., “A if and only if B”) or if the biconditional is correctly representing such an equivalence.
    5. Ignore all other aspects of the formula, including quantifiers, predicates, or additional logical operators.
    6. If the FOL formula incorrectly uses ↔ for a one-way implication, answer “error category does apply.” Otherwise, answer “error category does not apply.”

    Example:
    Sentence: An animal is a pet only if it is domesticated or it is a goldfish.
    FOL: ∀v (Animal(v) ∧ (Pet(v) ↔ (Domesticated(v) ∨ IsGoldfish(v))))
    Answer: error category does apply

    Example:
    Sentence: All kittens are not fierce or mean.
    FOL: ∀x (Kitten(x) ∧ (¬Fierce(x) ∨ ¬Mean(x)))
    Answer: error category does not apply

    Example:
    Sentence: An entity is a heavy cube only if it's not yellow.
    FOL: ∀v (Heavy(v) ∧ Cube(v) ↔ ¬Yellow(v))
    Answer: error category does apply

    Example:
    Sentence: All birds that are not both white and black are eagles.
    FOL: ∀x ((¬(White(x) ∧ Black(x))) → Eagles(x))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Konjunktion statt Implikation

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert nach allen Übersetzungen, deren FOL-Formel eine Konjunktion beinhaltet
    def conjunction_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.contains("∧")
        )
    return (conjunction_filtered,)


@app.cell
def _(conjunction_filtered, malls_changed):
    df_conjunction_filtered = conjunction_filtered(malls_changed)
    df_conjunction_filtered
    return


@app.cell
def _():
    conjunction_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: A conjunction (∧) is used instead of a conditional (→), changing the meaning to mutual truth instead of dependency.

    How to decide:
    1. Determine the intended logical relationship in the natural language sentence.
    2. Check if the FOL formula uses a conjunction (∧) where a one-way implication (→) is intended.
    3. Apply this error category only if the sentence expresses a conditional relationship (e.g., “If A, then B”) and the FOL incorrectly uses ∧ instead of →.
    3. Do not apply if the sentence actually expresses mutual truth or co-occurrence (e.g., “A and B are both true”) and ∧ is correct.
    4. Ignore all other elements of the formula, including quantifiers, predicates, or additional logical operators.
    5. Answer “error category does apply” if ∧ incorrectly replaces →; otherwise, answer “error category does not apply.”

    Example:
    Sentence: All kittens are not fierce or mean.
    FOL: ∀x (Kitten(x) ∧ (¬Fierce(x) ∨ ¬Mean(x)))
    Answer: error category does apply

    Example:
    Sentence: An entity is a heavy cube only if it's not yellow.
    FOL: ∀v (Heavy(v) ∧ Cube(v) ↔ ¬Yellow(v))
    Answer: error category does not apply

    Example:
    Sentence: If a house is neither big nor small, it's affordable.
    FOL: ∀x ((¬Big(x) ∨ ¬Small(x)) → Affordable(x))
    Answer: error category does not apply

    Example:
    Sentence: All kids are not troublesome or naughty.
    FOL: ∃x (Kids(x) ∧ (¬Troublesome(x) ∨ ¬Naughty(x)))
    Answer: error category does apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    XOR-Fehler

    FUNKTIONIERT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert nach allen Übersetzungen, deren FOL-Formel min. 2 XOR Operatoren enthalten
    def xor_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.count_matches("⊕") >= 2
        )
    return (xor_filtered,)


@app.cell
def _(test, xor_filtered):
    # Die noch einmal mit KI überprüfen!
    # Die Operatoren könnten unanhängig voneinander sein!
    # willow_changed: 78
    # malls_changed: 298

    df_xor_filtered = xor_filtered(test)
    df_xor_filtered
    return


@app.cell
def _():
    # FERTIG
    # willow_changed: 78
    #df_willow_xor = process_dataset_classification3(xor_filtered(willow_changed), xor_prompt)

    # Als Excel speichern
    #df_willow_xor.to_pandas().to_excel("data/willow_xor.xlsx", index=False)

    #df_willow_xor
    return


@app.cell
def _():
    # FERTIG
    # malls_changed: 298
    #df_malls_xor = process_dataset_classification3(xor_filtered(malls_changed), xor_prompt)

    # Als Excel speichern
    #df_malls_xor.to_pandas().to_excel("data/malls_xor.xlsx", index=False)

    #df_malls_xor
    return


@app.cell
def _():
    # FERTIG
    # willow_cleaned: 0
    return


@app.cell
def _():
    # FERTIG
    # malls_cleaned: 16 -> 9
    #df_malls_cleaned_xor = process_dataset_classification3(xor_filtered(malls_cleaned), xor_prompt)

    # Als Excel speichern
    #df_malls_cleaned_xor.to_pandas().to_excel("data/malls_cleaned_xor.xlsx", index=False)

    #df_malls_cleaned_xor
    return


@app.cell
def _():
    xor_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: The XOR operator (⊕) is used incorrectly in a formula involving more than two variables. XOR is only defined for two arguments at a time; applying it directly to three or more variables (e.g., A ⊕ B ⊕ C) is wrong.

    Example:
    Sentence: A movie can be awarded for best cinematography, best original score, or best costume design, but not all three simultaneously.
    FOL: ∀x (Movie(x) ∧ Awarded(x) → (BestCinematographyAward(x) ⊕ BestOriginalScoreAward(x) ⊕ BestCostumeDesignAward(x)))
    Answer: error category does apply

    Example:
    Sentence: A person is either alive or dead, never both.
    FOL: ∀x (Person(x) → (Alive(x) ⊕ Dead(x)))
    Answer: error category does not apply

    Example:
    Sentence: A chef creates dishes using ingredients like vegetables, fruits, or grains.
    FOL: ∀x∀y∀z∀w (Chef(x) ∧ (Vegetable(y) ∨ Fruit(z) ∨ Grain(w)) → CreatesDishWithIngredient(x, y) ⊕ CreatesDishWithIngredient(x, z) ⊕ CreatesDishWithIngredient(x, w))
    Answer: error category does apply

    Example:
    Sentence: A car is either a vintage Model-T or a modern Tesla, but not both.
    FOL: ∀x ((Vintage(x) ∧ ModelT(x)) ⊕ (Modern(x) ∧ Tesla(x)))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fehlende Klammern

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _():
    #df_parantheses = process_dataset_classification3(test, parantheses_prompt).with_row_index("row", 1)
    #df_parantheses
    return


@app.cell
def _():
    parantheses_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: Parentheses are missing or misplaced, changing the intended grouping of logical operations.

    How to decide:
    1. Examine the FOL formula for parentheses around logical expressions.
    2. Check the FOL formula and apply standard operator precedence rules:
    - ¬ binds strongest,
    - ∧,
    - ∨,
    - →,
    - ↔ binds weakest.
    3. Apply this error category only if the formula’s meaning changes compared to the intended grouping after applying the standard precedence rules, meaning that extra parentheses would be required to express the correct logic.
    4. Do not apply this category if the standard precedence rules already produce the correct intended grouping, even if parentheses are absent.
    5. Ignore all other aspects of the formula, including variable names, predicates, or quantifiers.
    6. Answer “error category does apply” if parentheses errors change the intended logical grouping; otherwise, answer “error category does not apply.”

    Example:
    Sentence: If an instrument is part of a symphony orchestra, it is a string, woodwind, or brass instrument.
    FOL: ∀x (Instrument(x) ∧ PartOfSymphonyOrchestra(x) → StringInstrument(x) ∨ WoodwindInstrument(x) ∨ BrassInstrument(x))
    Answer: error category does not apply

    Example:
    Sentence: A person is a musician if and only if they play an instrument or sing, but they do not dissonance.
    FOL: ∀x (Person(x) ∧ Musician(x) ↔ (PlayInstrument(x) ∨ Sing(x) ∧ ¬Dissonance(x)))
    Answer: error category does apply

    Example:
    Sentence: A machine is operational if and only if it uses electricity or solar energy, but not oil.
    FOL: ∀x (Machine(x) → (Operational(x) ↔ UseElectricity(x) ∨ UseSolar(x) ∧ ¬UseOil(x)))
    Answer: error category does apply

    Example:
    Sentence: A material can be metal, wood, or plastic.
    FOL: ∀x (Material(x) → Metal(x) ∨ Wood(x) ∨ Plastic(x))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Falsche Gesamtnegation

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _(pl):
    # Nach allen Übersetzungen filtern, dessen Fol-Fomreln mit einem Negationszeichen starten
    def global_negation_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.starts_with("¬")
        )
    return


@app.cell
def _():
    # willow_changed: 2.389
    # malls_changed: 160

    #df_global_negation_filtered = global_negation_filtered(malls_changed)
    #df_global_negation_filtered
    return


@app.cell
def _():
    #df_global_negation = process_dataset_classification3(test, global_negation_prompt).with_row_index("row", 1)
    #df_global_negation
    return


@app.cell
def _():
    global_negation_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: The entire formula is unnecessarily negated, inverting its meaning.

    How to decide:
    1. Examine the FOL formula to see if a single negation (¬) is applied to the entire formula.
    2. Determine the intended meaning of the natural language sentence.
    3. Apply this error category only if the top-level negation reverses the intended truth of the entire statement.
    4. Do not apply this category if the negation is part of a sub-expression or is necessary to represent the sentence correctly.
    5. Ignore other issues such as quantifiers, argument order, predicate names, or operator grouping.
    6. If the entire formula is negated and this changes the intended meaning, answer “error category does apply.” Otherwise, answer “error category does not apply.”

    Example:
    Sentence: If all humans admire John then there are people who do not respect Emma.
    FOL: ¬∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))
    Answer: error category does apply

    Example:
    Sentence: Not all humans follow David.
    FOL: ¬∀x (Human(x) → Follow(x, david))
    Answer: error category does not apply

    Example:
    Sentence: There is no book and Henry is a writer.
    FOL: ¬∃x (Book(x)) ∧ Writer(henry)
    Answer: error category does apply

    Example:
    Sentence: If all humans admire Sophia then there are people who do not respect Noah.
    FOL: ¬∀x (Human(x) → Admire(x, sophia)) → ∃x (Human(x) ∧ ¬Respect(x, noah))
    Answer: error category does apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fehlende Negation der Bedingung

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert die Übersetzungen nach FOL-Formeln, die eine Implikation enthalten
    def condition_negation_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.FOL_expression.str.contains("→")
        )
    return


@app.cell
def _():
    # willow_changed: 12.715
    # malls_changed: 23.702

    #df_condition_negation_filtered = condition_negation_filtered(willow_changed)
    #df_condition_negation_filtered
    return


@app.cell
def _():
    #df_condition_negation = process_dataset_classification3(test, condition_negation_prompt).with_row_index("row", 1)
    #df_condition_negation
    return


@app.cell
def _():
    condition_negation_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: A necessary negation in the condition part (antecedent) of an implication is missing.

    How to decide:
    1. Identify the antecedent (the “if” part) of the implication in the FOL formula.
    2. Determine whether the natural language sentence requires a negation in this condition.
    3. Apply this error category only if the negation is necessary to preserve the intended meaning of the sentence.
    4. Do not apply this category if the negation is optional, part of the consequent, or unrelated to the antecedent.
    5. Ignore other aspects of the formula, such as quantifiers, predicate names, or operator order.
    6. If the formula omits a required negation in the antecedent, answer “error category does apply.” Otherwise, answer “error category does not apply.”

    Example:
    Sentence: Unless a country is either poor or rich, it is a developed country.
    FOL: ∀v ((Poor(v) ∨ Rich(v)) → Developed(v))
    Answer: error category does apply

    Example:
    Sentence: An animal can be a bird or a mammal.
    FOL: ∀x (Animal(x) → (Bird(x) ⊕ Mammal(x)))
    Answer: error category does not apply

    Example:
    Sentence: Algebraic laws are not assumptions.
    FOL: ∀x (AlgebraicLaw(x) → ¬Assumption(x))
    Answer: error category does not apply

    Example:
    Sentence: A food that isn't spicy or sour is sweet.
    FOL: ∀w ((Spicy(w) ∨ Sour(w)) → Sweet(w))
    Answer: error category does apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Fehlerhafte Negation der Bedingung und Folge

    FUNKTIONIERT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert nach allen Übersetzungen, bei denen die FOL-Formel sowohl eine Implikation, als auch min. 2 Negationszeichen enthält
    def both_negated_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            (pl.col.FOL_expression.str.count_matches("¬") >= 2) &
            (pl.col.FOL_expression.str.contains("→"))
        )
    return (both_negated_filtered,)


@app.cell
def _(both_negated_filtered, willow_cleaned):
    df_both_negated_filtered = both_negated_filtered(willow_cleaned)
    df_both_negated_filtered
    return


@app.cell
def _():
    # FERTIG
    # willow_changed: 862
    #df_willow_both_negated = process_dataset_classification3(both_negated_filtered(willow_changed), both_negated_prompt)

    # Als Excel speichern
    #df_willow_both_negated.to_pandas().to_excel("data/willow_both_negated.xlsx", index=False)

    #df_willow_both_negated
    return


@app.cell
def _():
    # FERTIG
    # malls_changed: 496
    #df_malls_both_negated = process_dataset_classification3(both_negated_filtered(malls_changed), both_negated_prompt)

    # Als Excel speichern
    #df_malls_both_negated.to_pandas().to_excel("data/malls_both_negated.xlsx", index=False)

    #df_malls_both_negated
    return


@app.cell
def _():
    # FERTIG
    # willow_cleaned: 1.030
    #df_willow_cleaned_both_negated = process_dataset_classification3(both_negated_filtered(willow_cleaned), both_negated_prompt)

    # Als Excel speichern
    #df_willow_cleaned_both_negated.to_pandas().to_excel("data/willow_cleaned_both_negated.xlsx", index=False)

    #df_willow_cleaned_both_negated
    return


@app.cell
def _():
    # FERTIG
    # malls_cleaned: 975
    #df_malls_cleaned_both_negated = process_dataset_classification3(both_negated_filtered(malls_cleaned), both_negated_prompt)

    # Als Excel speichern
    #df_malls_cleaned_both_negated.to_pandas().to_excel("data/malls_cleaned_both_negated.xlsx", index=False)

    #df_malls_cleaned_both_negated
    return


@app.cell
def _():
    both_negated_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: Negation is applied to both the condition and the conclusion of an implication, incorrectly reversing the intended logical relationship.

    How to decide:
    1. Identify the implication in the FOL formula, separating the antecedent (condition) and consequent (conclusion).
    2. Check whether negations are applied to both the antecedent and the consequent.
    3. Apply this error category only if such double negation reverses the intended meaning of the natural language sentence.
    4. Do not apply this category if negation is correctly used in only one part, or if the double negation preserves the intended meaning.
    5. Ignore other aspects of the formula, such as quantifiers, predicate names, or additional operators.
    6. If the formula incorrectly negates both parts of the implication, answer “error category does apply.” Otherwise, answer “error category does not apply.”

    Example:
    Sentence: No squares are big.
    FOL: ∀y (Square(y) → ¬Big(y))
    Answer: error category does not apply

    Example:
    Sentence: All careful persons are alive.
    FOL: ∀x (¬Careful(x) → ¬Alive(x))
    Answer: error category does apply

    Example:
    Sentence: None but the strong survive the storm.
    FOL: ∀z (¬Strong(z) → ¬SurviveStorm(z))
    Answer: error category does apply

    Example:
    Sentence: Except for the honest, none deserve respect.
    FOL: ∀z (¬Honest(z) → ¬DeserveRespect(z))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    "Neither ... Nor"-Fehler

    FUNKTIONIERT
    """
    )
    return


@app.cell
def _(pl):
    # Filtert nach Übersetzungen, dessen NL-Sätze das Wort neither enthalten
    def neither_nor_filtered(data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            pl.col.NL_sentence.str.contains(r"(?i)\bneither\b", literal=False)
        )
    return (neither_nor_filtered,)


@app.cell
def _(neither_nor_filtered, willow_cleaned):
    df_neither_nor_filtered = neither_nor_filtered(willow_cleaned)
    df_neither_nor_filtered
    return


@app.cell
def _():
    # FERTIG
    # willow_changed: 154
    #df_willow_neither_nor = process_dataset_classification3(neither_nor_filtered(willow_changed), neither_nor_prompt)

    # Als Excel speichern
    #df_willow_neither_nor.to_pandas().to_excel("data/willow_neither_nor.xlsx", index=False)

    #df_willow_neither_nor
    return


@app.cell
def _():
    # FERTIG
    # malls_changed: 78
    #df_malls_neither_nor = process_dataset_classification3(neither_nor_filtered(malls_changed), neither_nor_prompt)

    # Als Excel speichern
    #df_malls_neither_nor.to_pandas().to_excel("data/malls_neither_nor.xlsx", index=False)

    #df_malls_neither_nor
    return


@app.cell
def _():
    # FERTIG
    # willow_cleaned: 154
    #df_willow_cleaned_neither_nor = process_dataset_classification3(neither_nor_filtered(willow_cleaned), neither_nor_prompt)

    # Als Excel speichern
    #df_willow_cleaned_neither_nor.to_pandas().to_excel("data/willow_cleaned_neither_nor.xlsx", index=False)

    #df_willow_cleaned_neither_nor
    return


@app.cell
def _():
    # FERTIG
    # malls_cleaned: 78
    #df_malls_cleaned_neither_nor = process_dataset_classification3(neither_nor_filtered(malls_cleaned), neither_nor_prompt)

    # Als Excel speichern
    #df_malls_cleaned_neither_nor.to_pandas().to_excel("data/malls_cleaned_neither_nor.xlsx", index=False)

    #df_malls_cleaned_neither_nor
    return


@app.cell
def _():
    neither_nor_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: ‘Neither A nor B’ is incorrectly translated using disjunction (∨) instead of conjunction (∧) of negations.

    How to decide:
    1. Check if the natural language sentence explicitly contains the phrase “neither … nor …” construction. If not, answer “error category does not apply” immediately.
    2. If the sentence does contain such a construction, examine the FOL formula to see how it is represented.
    3. In correct FOL, “Neither A nor B” should be expressed as: (¬𝐴 ∧ ¬𝐵) or equivalently: ¬(A ∨ B). If not, answer “error category does apply”

    Example:
    Sentence: If a house is neither big nor small, it's affordable.
    FOL: ∀x ((¬Big(x) ∨ ¬Small(x)) → Affordable(x))
    Answer: error category does apply

    Example:
    Sentence: All participants in the meeting speak neither French nor German.
    FOL: ∀w (Participant(w) ∧ AtMeeting(w) → ¬(Speak(w, french) ∨ Speak(w, german)))
    Answer: error category does not apply

    Example:
    Sentence: Unless a thing is neither blue nor yellow, it is a triangle.
    FOL: ∀x ((Blue(x) ∨ Yellow(x)) → Triangle(x))
    Answer: error category does apply

    Example:
    Sentence: All students in the auditorium neither heard Beethoven nor Mozart.
    FOL: ∀x (Student(x) ∧ InAuditorium(x) → ¬(Hear(x, beethoven) ∨ Hear(x, mozart)))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Falsche Prädikatsbenennung

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _():
    #df_name = process_dataset_classification3(test, name_prompt).with_row_index("row", 1)
    #df_name
    return


@app.cell
def _():
    name_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: The predicate names in the FOL formula do not accurately reflect the meaning or action described in the natural language sentence.

    How to decide:
    1. Identify all predicate names in the FOL formula.
    2. Identify the actions, states, or categories in the natural language sentence that these predicates are meant to represent.
    3. If all predicate names correspond in meaning to the intended actions, states, or categories in the sentence, answer “error category does not apply” immediately — ignore all other differences such as quantifier choice, logical operators, or argument structure.
    4. Only if one or more predicate names clearly describe a different action, state, or category than intended in the sentence, answer “error category does apply.”
    5. Do not consider:
    - the arguments or terms passed to the predicates,
    - whether all words from the sentence appear in the formula,
    - the logical structure or correctness of the formula beyond predicate naming.

    Example:
    Sentence: A car is electric if it uses an electric motor instead of an internal combustion engine.
    FOL: ∀x (Car(x) ∧ ElectricMotor(x) ∧ ¬InternalCombustionEngine(x) → ElectricCar(x))
    Answer: error category does apply

    Example:
    Sentence: All roses are flowers.
    FOL: ∀x (Roses(x) → Flowers(x))
    Answer: error category does not apply

    Example:
    Sentence: Not all birds can fly or swim.
    FOL: ¬∀x (Bird(x) → (Fly(x) ∨ Sing(x)))
    Answer: error category does apply

    Example:
    Sentence: No insect having wings lacks antennae.
    FOL: ∀x (Insect(x) → (HasWing(x) → ¬LackAntenna(x)))
    Answer: error category does not apply

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Prädikatsverschachtelung

    FUNKTIONIERT NICHT
    """
    )
    return


@app.cell
def _():
    #df_predicate_call = process_dataset_classification3(test, predicate_call_prompt).with_row_index("row", 1)
    #df_predicate_call
    return


@app.cell
def _():
    predicate_call_prompt = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Does the described error categorie apply?

    Error cagegory: A predicate incorrectly contains another predicate as an argument instead of a term.

    How to decide:
    1. Recall that in First-Order Logic, predicates must only have terms as arguments. Terms are variables (e.g., x, y), constants (e.g., Alice, book1), or function terms (e.g., f(x)).
    2. Go through each predicate in the FOL formula.
    3. For each argument of each predicate, check whether it is itself a predicate (e.g., P(Q(x))).
    4. If any argument of a predicate is a predicate, answer “error category does apply.”
    5. If all arguments are terms (variables, constants, or function terms), answer “error category does not apply.”

    Example:
    Sentence: Suppose no ones loves their commute, there is a driver who everyone praises.
    FOL: ¬∀x (Person(x) → Loves(x, Commute(x))) → ∃y (Driver(y) ∧ ∀z (Person(z) → Praises(z, y)))
    Answer: error category does apply

    Example:
    Sentence: If Alice trusts Bob then Bob trusts Alice.
    FOL: Trust(alice, bob) → Trust(bob, alice)
    Answer: error category does not apply

    """
    return


@app.cell
def _(pl):
    # Alle Datensätze zur Fehlererkennung der einzelnen Fehlergruppen wieder hochladen
    # Damit überprüft werden, kann wie häufig eine Fehlergruppe vorkommt und das nur die gewollten Ausgaben existieren

    willow_verallgemeinerung = pl.read_excel("data/willow_cleaned_overgeneralization.xlsx")
    malls_verallgemeinerung = pl.read_excel("data/malls_cleaned_overgeneralization.xlsx")

    willow_bindung = pl.read_excel("data/willow_cleaned_wrong_binding.xlsx")
    malls_bindung = pl.read_excel("data/malls_cleaned_wrong_binding.xlsx")

    malls_xor = pl.read_excel("data/malls_cleaned_xor.xlsx")

    willow_bedingung_implikation = pl.read_excel("data/willow_cleaned_both_negated.xlsx")
    malls_bedingung_implikation = pl.read_excel("data/malls_cleaned_both_negated.xlsx")

    willow_neither = pl.read_excel("data/willow_cleaned_neither_nor.xlsx")
    malls_neither = pl.read_excel("data/malls_cleaned_neither_nor.xlsx")

    error_data = malls_neither
    return (error_data,)


@app.cell
def _(error_data, pl):
    # Filtert nach allen Übersetzungen, wo die Fehlergruppe zutrifft, um diese zu Zählen
    error_filtered = error_data.filter(pl.col("error_type") == "error category does apply")
    error_filtered
    return


@app.cell
def _(error_data, pl):
    # Überprüft, ob es zu anderen Ausgaben als "error category does apply" und "error category does not apply" gekommen ist
    NOTerror_filtered = error_data.filter(
        (pl.col("error_type") != "error category does apply") &
        (pl.col("error_type") != "error category does not apply")
    )

    NOTerror_filtered
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Datensatz Bereinigung""")
    return


@app.cell
def _(SYSTEM_PROMPT_CORRECTION, few_shot_examples, llm, pl):
    # Für das Bereinigen der Datensätze

    def process_entries_correction(sentence: str, translation: str) -> str:

        # Gültige Kombinationen:
        # 1.Ansatz: promptVariant = few_shot_examples und systemPrompt = SYSTEM_PROMPT_CORRECTION
        # 2.Ansatz: promptVariant = few_shot_examples_explanation und systemPrompt = SYSTEM_PROMPT_CORRECTION
        # 3.Ansatz: promptVariant = few_shot_examples und systemPrompt = SYSTEM_PROMPT_CORRECTION_more

        promptVariant = few_shot_examples 
        systemPrompt = SYSTEM_PROMPT_CORRECTION

        prompt = (
            f"{promptVariant}"
            f"Example 16:\n"
            f"Sentence: {sentence}\n"
            f"Translation: {translation}\n"
            f"Corrected Translation:"

        )

        # Zur Kontrolle der Prompts
        #print("🧠 Prompt:\n", prompt)

        messages = [
        ("system", systemPrompt,),
        ("human", prompt),
        ]

        ai_msg = llm.invoke(messages)
        return ai_msg.content.strip()



    # Hier mit zufälligen Daten, später anpassen
    # Dann nur noch den TEST Datensatz mit den wichtigsten Fehlergruppen
    # Und danach für den kompleten Datensatz anwendbar
    def process_dataset_correction(data: pl.DataFrame) -> pl.DataFrame:

        translations = data.map_rows(
            lambda r: process_entries_correction(r[0], r[1])
        )

        # ursprüngliche Daten + neue Übersetzungen zusammenführen
        result = pl.concat((data, translations), how="horizontal")
        result = result.rename({"map": "FOL_LLM"})

        # Vergleichsspalte hinzufügen (nur zum Prüfen)
        result = result.with_columns(
            (pl.col("FOL_expression") == pl.col("FOL_LLM")).alias("is_equal")
        )

        return result
    return


@app.cell
def _():
    #df_test_cleaned = process_dataset_correction(test)

    # Als Excel speichern
    #df_test_cleaned.to_pandas().to_excel("data/test_cleaned.xlsx", index=False)

    #df_test_cleaned
    return


@app.cell
def _():
    #df_malls_cleaned = process_dataset_correction(malls_changed)

    # Als Excel speichern
    #df_malls_cleaned.to_pandas().to_excel("data/malls_cleaned.xlsx", index=False)

    #df_malls_cleaned
    return


@app.cell
def _():
    #df_willow_cleaned = process_dataset_correction(willow_changed)

    # Als Excel speichern
    #df_willow_cleaned.to_pandas().to_excel("data/willow_cleaned.xlsx", index=False)

    #df_willow_cleaned
    return


@app.cell
def _():
    few_shot_examples = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Detect and correct erroneous translations, bringing them into the correct FOL form using the following examples.

    Example 1: 
    Sentence: All birds that are not both white and black are eagles. 
    Translation: ∀x ((¬(White(x) ∧ Black(x))) → Eagles(x)) 
    Corrected Translation: ∀x((Bird(x) ∧ ¬(White(x) ∧ Black(x))) → Eagle(x)) 

    Example 2: 
    Sentence: A musician plays an instrument and performs in a concert. 
    Translation: ∀x∀y (Musician(x) ∧ Instrument(y) ∧ Concert(z) → Plays(y, x) ∧ PerformsIn(x, z)) 
    Corrected Translation: ∀x (Musician(x) → ∃y∃z (Instrument(y) ∧ Concert(z) ∧ Plays(x, y) ∧ PerformsIn(x, z))) 

    Example 3: 
    Sentence: All kittens are not fierce or mean. 
    Translation: ∀x (Kitten(x) ∧ (¬Fierce(x) ∨ ¬Mean(x))) 
    Corrected Translation: ∀x (Kitten(x) → ¬(Fierce(x) ∧ Mean(x))) 

    Example 4: 
    Sentence: An entity is a heavy cube only if it's not yellow. 
    Translation: ∀v (Heavy(v) ∧ Cube(v) ↔ ¬Yellow(v))
    Corrected Translation: ∀v ((Heavy(v) ∧ Cube(v)) → ¬Yellow(v)) 

    Example 5: 
    Sentence: If a house is neither big nor small, it's affordable. 
    Translation: ∀x ((¬Big(x) ∨ ¬Small(x)) → Affordable(x)) 
    Corrected Translation: ∀x((¬Big(x)∧¬Small(x))→Affordable(x)) 

    Example 6: 
    Sentence: If all humans admire John then there are people who do not respect Emma. 
    Translation: ¬∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma)) 
    Corrected Translation: ∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma)) 

    Example 7: 
    Sentence: Unless a country is either poor or rich, it is a developed country. 
    Translation: ∀v ((Poor(v) ∨ Rich(v)) → Developed(v)) 
    Corrected Translation: ∀v (Country(v) → ¬(Poor(v) ∨ Rich(v)) → Developed(v)) 

    Example 8: 
    Sentence: A person is a musician if and only if they play an instrument or sing, but they do not dissonance. 
    Translation: ∀x (Person(x) ∧ Musician(x) ↔ (PlayInstrument(x) ∨ Sing(x) ∧ ¬Dissonance(x))) 
    Corrected Translation: ∀x (Person(x) → (Musician(x) ↔ ((PlayInstrument(x) ∨ Sing(x)) ∧ ¬Dissonance(x)))) 

    Example 9: 
    Sentence: A worker can be a singer, or appreciate Da Vinci's sketches, but not both. 
    Translation: ∀x (Singer(x) ⊕ AppreciateDaVinci(x)) 
    Corrected Translation: ∀x (Worker(x) → (Singer(x) ⊕ AppreciateDaVincisSketches(x))) 

    Example 10: 
    Sentence: Loving parents do not neglect their children. 
    Translation: ∀x (Parent(x) ∧ Loving(x) → ¬Neglect(x, child)) 
    Corrected Translation: ∀x (Parent(x) ∧ Loving(x) → ∀y (ChildOf(y,x) → ¬Neglect(x,y))) 

    Example 11: 
    Sentence: A movie can be awarded for best cinematography, best original score, or best costume design, but not all three simultaneously. 
    Translation: ∀x (Movie(x) ∧ Awarded(x) → (BestCinematographyAward(x) ⊕ BestOriginalScoreAward(x) ⊕ BestCostumeDesignAward(x))) 
    Corrected Translation: ∀x (Movie(x) ∧ Awarded(x) → ((BestCinematographyAward(x) ∧ ¬BestOriginalScoreAward(x) ∧ ¬BestCostumeDesignAward(x)) ∨ (¬BestCinematographyAward(x) ∧ BestOriginalScoreAward(x) ∧ ¬BestCostumeDesignAward(x)) ∨ (¬BestCinematographyAward(x) ∧ ¬BestOriginalScoreAward(x) ∧ BestCostumeDesignAward(x)))) 

    Example 12: 
    Sentence: All careful persons are alive. 
    Translation: ∀x (¬Careful(x) → ¬Alive(x)) 
    Corrected Translation: ∀x (Person(x) ∧ Careful(x) → Alive(x))

    Example 13: 
    Sentence: A teacher gives a student a book. 
    Translation: ∀x∀y (Teacher(x) ∧ Student(y) ∧ Book(z) → Gives(x, y, z)) 
    Corrected Translation: ∀x (Teacher(x) → ∃y∃z (Student(y) ∧ Book(z) ∧ Gives(x, y, z))) 

    Example 14: 
    Sentence: All dogs chase a ball and then bite it. 
    Translation: ∀x∀y (Dog(x) ∧ Ball(y) → Chase(x, y) ∧ Bite(x, y)) 
    Corrected Translation: ∀x (Dog(x) → ∃y (Ball(y) ∧ Chases(x, y) ∧ Bites(x, y)))

    Example 15:
    Sentence: A student is not brilliant and not diligent.
    Translation: ∀x (Student(x) → ¬(Brilliant(x) ∧ Diligent(x)))
    Corrected Translation: ∀x (Student(x) → ¬Brilliant(x) ∧ ¬Diligent(x))

    """
    return (few_shot_examples,)


@app.cell
def _():
    few_shot_examples_explanation = """
    You are given natural language sentences and their translations into First-Order Logic (FOL). Detect and correct erroneous translations, bringing them into the correct FOL form using the following examples. Each correction is followed by a short explanation.

    Example 1:
    Sentence: All birds that are not both white and black are eagles.
    Translation: ∀x ((¬(White(x) ∧ Black(x))) → Eagles(x))
    Corrected Translation: ∀x((Bird(x) ∧ ¬(White(x) ∧ Black(x))) → Eagle(x))
    Explanation: The predicate Bird(x) was missing, and "Eagles(x)" must be singular "Eagle(x)".

    Example 2:
    Sentence: A musician plays an instrument and performs in a concert.
    Translation: ∀x∀y∀z (Musician(x) ∧ Instrument(y) ∧ Concert(z) → Plays(y, x) ∧ PerformsIn(x, z))
    Corrected Translation: ∀x (Musician(x) → ∃y∃z (Instrument(y) ∧ Concert(z) ∧ Plays(x, y) ∧ PerformsIn(x, z)))
    Explanation: The quantifiers must show that each musician plays some instrument and performs in some concert, not all universally.

    Example 3:
    Sentence: All kittens are not fierce or mean.
    Translation: ∀x (Kitten(x) ∧ (¬Fierce(x) ∨ ¬Mean(x)))
    Corrected Translation: ∀x (Kitten(x) → ¬(Fierce(x) ∧ Mean(x)))
    Explanation: The sentence means kittens cannot be both fierce and mean, not that each kitten must satisfy the disjunction.

    Example 4:
    Sentence: An entity is a heavy cube only if it's not yellow.
    Translation: ∀v (Heavy(v) ∧ Cube(v) ↔ ¬Yellow(v))
    Corrected Translation: ∀v ((Heavy(v) ∧ Cube(v)) → ¬Yellow(v))
    Explanation: "Only if" is a one-way implication, not a biconditional. Only "if and only if" is a biconditional

    Example 5:
    Sentence: If a house is neither big nor small, it's affordable.
    Translation: ∀x ((¬Big(x) ∨ ¬Small(x)) → Affordable(x))
    Corrected Translation: ∀x((¬Big(x)∧¬Small(x))→Affordable(x))
    Explanation: "Neither...nor" means conjunction of negations, not disjunction.

    Example 6:
    Sentence: If all humans admire John then there are people who do not respect Emma.
    Translation: ¬∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))
    Corrected Translation: ∀x (Human(x) → Admire(x, john)) → ∃x (Human(x) ∧ ¬Respect(x, emma))
    Explanation: The antecedent was wrongly negated; the sentence directly states the condition, not its negation.

    Example 7:
    Sentence: Unless a country is either poor or rich, it is a developed country.
    Translation: ∀v ((Poor(v) ∨ Rich(v)) → Developed(v))
    Corrected Translation: ∀v (Country(v) → ¬(Poor(v) ∨ Rich(v)) → Developed(v))
    Explanation: The "unless" structure means if not poor or rich, then developed. Also, Country(v) must be included.

    Example 8:
    Sentence: A person is a musician if and only if they play an instrument or sing, but they do not dissonance.
    Translation: ∀x (Person(x) ∧ Musician(x) ↔ (PlayInstrument(x) ∨ Sing(x) ∧ ¬Dissonance(x)))
    Corrected Translation: ∀x (Person(x) → (Musician(x) ↔ ((PlayInstrument(x) ∨ Sing(x)) ∧ ¬Dissonance(x))))
    Explanation: The biconditional applies inside the Person(x) condition, and parentheses were misplaced.

    Example 9:
    Sentence: A worker can be a singer, or appreciate Da Vinci's sketches, but not both.
    Translation: ∀x (Singer(x) ⊕ AppreciateDaVinci(x))
    Corrected Translation: ∀x (Worker(x) → (Singer(x) ⊕ AppreciateDaVincisSketches(x)))
    Explanation: The restriction to workers was missing, and the predicate name must match the sentence.

    Example 10:
    Sentence: Loving parents do not neglect their children.
    Translation: ∀x (Parent(x) ∧ Loving(x) → ¬Neglect(x, child))
    Corrected Translation: ∀x (Parent(x) ∧ Loving(x) → ∀y (ChildOf(y,x) → ¬Neglect(x,y)))
    Explanation: The relation to their children must be made explicit, not a generic "child".

    Example 11:
    Sentence: A movie can be awarded for best cinematography, best original score, or best costume design, but not all three simultaneously.
    Translation: ∀x (Movie(x) ∧ Awarded(x) → (BestCinematographyAward(x) ⊕ BestOriginalScoreAward(x) ⊕ BestCostumeDesignAward(x)))
    Corrected Translation: ∀x (Movie(x) ∧ Awarded(x) → ((BestCinematographyAward(x) ∧ ¬BestOriginalScoreAward(x) ∧ ¬BestCostumeDesignAward(x)) ∨ (¬BestCinematographyAward(x) ∧ BestOriginalScoreAward(x) ∧ ¬BestCostumeDesignAward(x)) ∨ (¬BestCinematographyAward(x) ∧ ¬BestOriginalScoreAward(x) ∧ BestCostumeDesignAward(x))))
    Explanation: Exclusive-or with three options must be expanded explicitly to prevent "all three".

    Example 12:
    Sentence: All careful persons are alive.
    Translation: ∀x (¬Careful(x) → ¬Alive(x))
    Corrected Translation: ∀x (Person(x) ∧ Careful(x) → Alive(x))
    Explanation: The original was contrapositive and lost the Person(x) condition; correction matches the natural meaning.

    """
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# LLM Setup""")
    return


@app.cell
def _(ChatOpenAI, os):
    # ES müssen noch eigene Keys hinzugefügt werden, bevor man den Code testen kann!

    if not os.environ.get("OPENAI_API_KEY"):
        print("API Key nicht gesetzt!")

    if not os.environ.get("CHAT_AI_API_KEY"):
        print("Chat AI API Key nicht gesetzt!")


    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        #api_key=os.environ.get("CHAT_AI_API_KEY"),
        #base_url="https://chat-ai.academiccloud.de/v1"
    )
    return (llm,)


@app.cell
def _():
    # System Prompt für die Fehlererkennung 
    # Gilt für die ersten beiden Ansätze

    SYSTEM_PROMPT_CLASSIFICATION_ALLE = """
    You are an expert in formal logic (First-Order Logic, FOL). Your task is to analyze natural language statements and their corresponding FOL translations to identify typical logical errors.

    Your task is to:
    - Examine each NL/FOL pair.
    - Determine whether any of the known error categories apply.
    - Return **a list** of matching error labels (e.g., ["quantifier_error", "missing_category"]).
    - Return ["none"] if the translation is correct and contains no identifiable error.

    Important guidelines:
    - **Multiple errors** can apply to a single translation.
    - Be consistent and cautious. If you're unsure, return ["none"].
    - Base your analysis on both the structure and meaning of the original sentence and the FOL expression.

    Always return a **list of error labels**. Do not explain or justify your decision.
    """.strip()
    return (SYSTEM_PROMPT_CLASSIFICATION_ALLE,)


@app.cell
def _():
    # System Prompt für die Fehlererkennung 
    # Gilt für den dritten Ansatz

    SYSTEM_PROMPT_CLASSIFICATION_EINZELN = """
    You are an expert in formal logic (First-Order Logic, FOL). Your task is to analyze natural language statements and their corresponding FOL translations to identify errors.

    Your task is to:
    - Examine each NL/FOL pair.
    - Determine whether the described error categorie applies.
    - Return "error category does apply" when the error categorie applies and "error category does apply" if not.

    Important guidelines:
    - Be consistent and cautious. If you're unsure, return "error category does not apply".
    - Base your analysis on both the structure and meaning of the original sentence and the FOL expression.

    Do not explain or justify your decision.
    """.strip()
    return (SYSTEM_PROMPT_CLASSIFICATION_EINZELN,)


@app.cell
def _():
    # System Prompt für die Bereinigung
    # Gilt für die ersten beiden Ansätze

    SYSTEM_PROMPT_CORRECTION = """
    You are an expert in formal logic (First-Order Logic, FOL). 
    Your task is to analyze natural language statements and their corresponding FOL translations, identify errors, and correct them.

    Guidelines:
    - Use both the structure and meaning of the original sentence and the FOL expression to guide your corrections.
    - Be consistent with syntax (parentheses, quantifiers, operators) and semantics (correct predicates, correct quantifiers, implications).
    - Follow the style demonstrated in the few-shot examples provided.
    - Every object mentioned must be quantified (no free variables).  

    Output:
    - If the translation is already correct, return it unchanged.  
    - If it contains errors, return the corrected FOL formula.  
    - Do not provide explanations or justifications.  
    """.strip()
    return (SYSTEM_PROMPT_CORRECTION,)


@app.cell
def _():
    # System Prompt für die Bereinigung
    # Gilt für den dritten Ansatz

    SYSTEM_PROMPT_CORRECTION_more = """
    You are an expert in formal logic (First-Order Logic, FOL). 
    Your task is to analyze natural language statements and their corresponding FOL translations, identify errors, and correct them.

    When correcting, pay special attention to common error types:
    - A category mentioned in the natural language sentence is missing in the FOL translation.
    - Over-generalization by excessive use of the universal quantifier (∀).
    - Variables used in the wrong order inside predicates.
    - Free variables left unbound.
    - Using a biconditional (↔) instead of an implication (→).
    - Using a conjunction (∧) instead of an implication (→).
    - Exclusive-or (⊕) applied with three or more operands.
    - Parentheses missing, changing the meaning.
    - The entire formula incorrectly negated.
    - The antecedent of an implication missing the required negation.
    - Both parts of an implication wrongly negated.
    - “Neither … nor …” mistranslated.
    - Predicate names that do not reflect the natural language meaning.
    - Predicates incorrectly nested inside other predicates.

    Guidelines:
    - Use both the structure and meaning of the original sentence and the FOL expression to guide your corrections.
    - Be consistent with syntax (parentheses, quantifiers, operators) and semantics (correct predicates, correct quantifiers, implications).
    - Follow the style demonstrated in the few-shot examples provided.
    - Every object mentioned must be quantified (no free variables).  

    Output:
    - If the translation is already correct, return it unchanged.  
    - If it contains errors, return the corrected FOL formula.  
    - Do not provide explanations or justifications.  
    """.strip()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Grafiken erstellen""")
    return


@app.cell
def _():
    import plotly.express as px
    import pandas as pd
    return (px,)


@app.cell
def _(pl):
    datensatz = pl.read_excel("data/Grafik_Verbesserung.xlsx")
    return (datensatz,)


@app.cell
def _(pl, px):
    # Code, um für die einzelnen Fehlergruppen (bei denen die Fehlererkennung funktioniert) eine Grafik mit einem Vergleich vor und nach der Bereinigung zu erstellen

    def plot_category(data: pl.DataFrame, category_name: str):
        """
        Erstellt eine horizontale Balkengrafik für eine Kategorie.
        Vergleicht mehrere Datensätze (z. B. Willow, Malls)
        mit jeweils zwei Versionen (Vorher/Nachher).
        """

        df = data.to_pandas()
        df_filtered = df[df["Kategorie"] == category_name]

        fig = px.bar(
            df_filtered,
            x="Anzahl",
            y="Datensatz",
            color="Version",
            orientation="h",
            barmode="group",
            text="Anzahl",
            color_discrete_map={
                "Vorher": "#ff7f0e",
                "Nachher": "#1f77b4"
            },
            category_orders={"Datensatz": ["Willow", "Malls"], "Version": ["Datensatz nach der Bereinigung", "Originaldatensatz"]}
            #width=950,
            #height=420
        )

        fig.update_traces(texttemplate="%{text}", textposition="outside", textfont_size=14)

        fig.update_layout(
        font=dict(size=14),
        margin=dict(l=100, r=50, t=90, b=50),
        uniformtext_minsize=8,
        uniformtext_mode='show',
        xaxis_title="Anzahl",
        yaxis_title="Datensatz",
        legend_title_text="",
        legend=dict(
            orientation="h",
            y=1.05,
            x=0.5,
            xanchor='center',
            yanchor='bottom'
        ),
        bargap=0.18,       # Abstand zwischen Balkengruppen
        bargroupgap=0.08,   # Abstand zwischen Balken innerhalb einer Gruppe
        legend_traceorder="reversed"  # kehrt die Reihenfolge aller Legenden-Einträge um
        )

        # Abstand zwischen Y-Achsenlabels und Balken
        fig.update_yaxes(
            automargin=True,        # lässt Plotly linken Rand automatisch erweitern, falls nötig
            ticklen=6,              # Länge der Tick-Striche
            ticklabelstandoff=18    # Abstand (px) zwischen Tick-Label und Achse/Balken
        )

        fig.show()
        return fig
    return (plot_category,)


@app.cell
def _(datensatz, plot_category):
    grafik_verallgemeinerung = plot_category(datensatz, "Verallgemeinerung")

    grafik_verallgemeinerung.write_image("grafik/grafik_verallgemeinerung.pdf", width=1000, height=600)
    return


@app.cell
def _(datensatz, plot_category):
    grafik_falsche_bindung = plot_category(datensatz, "Falsche Bindung")

    grafik_falsche_bindung.write_image("grafik/grafik_falsche_bindung.pdf", width=1000, height=600)
    return


@app.cell
def _(datensatz, plot_category):
    grafik_xor = plot_category(datensatz, "XOR Operator")

    grafik_xor.write_image("grafik/grafik_xor.pdf", width=1000, height=600)
    return


@app.cell
def _(datensatz, plot_category):
    grafik_negierung = plot_category(datensatz, "Fehlerhafte Negierung")

    grafik_negierung.write_image("grafik/grafik_negierung.pdf", width=1000, height=600)
    return


@app.cell
def _(datensatz, plot_category):
    grafik_neither = plot_category(datensatz, "Neither ... Nor")

    grafik_neither.write_image("grafik/grafik_neither.pdf", width=1000, height=600)
    return


if __name__ == "__main__":
    app.run()

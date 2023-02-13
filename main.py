from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel


def main():
    # Initialize model
    model_xlm_wiki = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_de-wiki", labels=["OK", "BAD"])

    # Sample source/target
    source, target =  "Most sharks eat fish and other small aquatic animals. ", "Die meisten Haie fressen Fische und andere größere Meerestiere."

    #Prediction output
    predictions, raw_outputs, tokens = model_xlm_wiki.predict([[source, target]])

    # Print statement
    print(predictions, raw_outputs, tokens) 


if __name__ == "__main__":
    main()
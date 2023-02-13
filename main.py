from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel, predict


def main():
    model_xlm_wiki = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_de-wiki", labels=["OK", "BAD"])
    source, target =  "Most sharks eat fish and other small aquatic animals. ", "Die meisten Haie fressen Fische und andere größere Meerestiere."
    predictions, raw_outputs, tokens = model_xlm_wiki.predict([[source, target]])
    print(predictions, raw_outputs, tokens) 

if __name__ == "__main__":
    main()
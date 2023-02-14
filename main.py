from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel


def main():
    # Initialize model
    model_xlm_wiki = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_de-wiki", labels=["OK", "BAD"])

    # Sample source/target
    source, target =  "Most sharks eat fish and other small aquatic animals. ", "Die meisten Haie fressen Fische und andere größere Meerestiere."
    source1 = "Mehrere Rettungsorganisationen aus Deutschland schließen ihren Einsatz in der Türkei ab."
    target1 = "Several rescue organizations from Germany complete their mission in Turkey."

    # Text Source: https://www.tagesschau.de/wirtschaft/erzeugerpreise-dezember-gesamtjahr-2022-101.html
    # DE - EN via Google Translate
    source2 = "Im Gesamtjahr 2022 legten die Preise landwirtschaftlicher Produkte wie Kartoffeln, Getreide oder Milch wegen des Ukraine-Kriegs um ein Drittel im Vorjahresvergleich zu, wie das Statistische Bundesamt in Wiesbaden mitteilte. Das war der höchste Anstieg seit 1961. Zum Vergleich: 2021 waren die Preise noch um 8,8 Prozent gestiegen."
    target2 = "In 2022 as a whole, the prices of agricultural products such as potatoes, grain or milk increased by a third compared to the previous year due to the Ukraine war, as reported by the Federal Statistical Office in Wiesbaden. That was the highest increase since 1961. For comparison: in 2021, prices had risen by 8.8 percent."

    #Prediction output
    predictions, raw_outputs, tokens = model_xlm_wiki.predict([[source2, target2]])

    # Print statement
    print(predictions[0]) 
    print(raw_outputs)

    # Use raw_output indices to find words to correct
    for i in range(len(raw_outputs[0])):
      if raw_outputs[0][i] == 'BAD':
        print(tokens[i])


if __name__ == "__main__":
    main()
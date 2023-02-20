from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
from datasets import load_dataset


def main():
    # Initialize model and load dataset
    model_xlm_wiki = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_de-wiki", labels=["OK", "BAD"])
    dataset = load_dataset("GEM/xwikis")

    # Text Source: https://www.tagesschau.de/wirtschaft/erzeugerpreise-dezember-gesamtjahr-2022-101.html
    # DE - EN via Google Translate
    source2 = "Im Gesamtjahr 2022 legten die Preise landwirtschaftlicher Produkte wie Kartoffeln, Getreide oder Milch wegen des Ukraine-Kriegs um ein Drittel im Vorjahresvergleich zu, wie das Statistische Bundesamt in Wiesbaden mitteilte. Das war der höchste Anstieg seit 1961. Zum Vergleich: 2021 waren die Preise noch um 8,8 Prozent gestiegen."
    target2 = "In 2022 as a whole, the prices of agricultural products such as potatoes, grain or milk increased by a third compared to the previous year due to the Ukraine war, as reported by the Federal Statistical Office in Wiesbaden. That was the highest increase since 1961. For comparison: in 2021, prices had risen by 8.8 percent."

    # Dataset source/target analysis
    for j in range(10):
      data_predictions, data_raw_outputs, data_tokens = model_xlm_wiki.predict([[dataset['train']['src_summary'][j], dataset['train']['tgt_summary'][j]]])
      print(j)
      print(dataset['train']['src_summary'][j])
      print(dataset['train']['tgt_summary'][j])
      for k in range(len(data_raw_outputs[0])):
        if data_raw_outputs[0][k] == 'BAD':
          print(data_tokens[k])


    #Prediction output
    #predictions, raw_outputs, tokens = model_xlm_wiki.predict([[source2, target2]])
    predictions, raw_outputs, tokens = model_xlm_wiki.predict([[dataset['train']['src_summary'][1], dataset['train']['tgt_summary'][1]]])

    # Print statement
    print(dataset['train']['src_summary'][1])
    print(dataset['train']['tgt_summary'][1])
    print(predictions[0]) 
    print(raw_outputs)


    #Use raw_output indices to find words to correct
    for i in range(len(raw_outputs[0])):
      if raw_outputs[0][i] == 'BAD':
        print(tokens[i])


if __name__ == "__main__":
    main()
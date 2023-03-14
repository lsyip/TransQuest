from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
from datasets import load_dataset


def main():
    # Initialize model and load dataset
    model_xlm_wiki = MicroTransQuestModel("xlmroberta", "TransQuest/microtransquest-en_de-wiki", labels=["OK", "BAD"])
    dataset = load_dataset("wmt19", "de-en", split='train', streaming=True)
    dataset_head = dataset.take(10000)

    # Dataset source/target analysis
    counter = 0
    for item in dataset_head:
      de_src = item.get("translation").get("de")
      en_tgt = item.get("translation").get("en")
      data_predictions, data_raw_outputs, data_tokens = model_xlm_wiki.predict([[de_src, en_tgt]])
      print(counter)
      print(de_src)
      print(en_tgt)
      print("Raw QE Output:")
      print(data_predictions)
      print(data_raw_outputs)
      print(data_tokens)
      counter = counter + 1
      for k in range(len(data_raw_outputs[0])):
        if data_raw_outputs[0][k] == 'BAD':
          print(data_tokens[k])

      print()


    #Use raw_output indices to find words to correct
    for i in range(len(data_raw_outputs[0])):
      if data_raw_outputs[0][i] == 'BAD':
        print(data_tokens[i])


if __name__ == "__main__":
    main()
from transquest.algo.sentence_level.siamesetransquest.run_model import SiameseTransQuestModel
from datasets import load_dataset


def main():
    # Initialize model and load dataset
    model_sentence = SiameseTransQuestModel("TransQuest/siamesetransquest-da-ro_en-wiki")
    dataset = load_dataset("wmt19", "de-en", split='train', streaming=True)
    dataset_head = dataset.take(10000)

    # Dataset source/target analysis
    counter = 0
    for item in dataset_head:
      de_src = item.get("translation").get("de")
      en_tgt = item.get("translation").get("en")
      predictions = model_sentence.predict([[de_src, en_tgt]])
      print(counter)
      print(de_src)S
      print(en_tgt)

      print(predictions)


if __name__ == "__main__":
    main()
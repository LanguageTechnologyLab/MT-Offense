import os
import shutil
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from TestInstance import TestInstance
from evaluation import macro_f1, weighted_f1
from label_converter import decode, encode
from print_stat import print_information
from fine_tune_config import TEMP_DIRECTORY, \
    MODEL_TYPE, MODEL_NAME, args, SEED, RESULT_FOLDER

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(RESULT_FOLDER): os.makedirs(RESULT_FOLDER)

train = pd.read_csv('./train data translated/hindi_translated_training.csv').dropna()
test_translated = pd.read_csv('./test data/translated_hindi.tsv', sep='\t').dropna()
test_original = pd.read_csv('./test data/hindi_test.tsv', sep='\t').dropna()

test_translated_model_1 = test_translated.rename(columns={'Helsinki-NLP/opus-mt-en-hi--hi': 'text', 'labels': 'labels'})
test_translated_model_2 = test_translated.rename(columns={'facebook/m2m100_1.2B--hi': 'text', 'labels': 'labels'})
test_translated_model_3 = test_translated.rename(columns={'facebook/nllb-200-3.3B--hin_Deva': 'text', 'labels': 'labels'})
test_original = test_original.rename(columns={'hindi-text': 'text', 'labels': 'labels'})

train = train.rename(columns={'model-1': 'text', 'labels': 'labels'})
train = train[['text', 'labels']]
train['labels'] = encode(train["labels"])

test_files_dict={
    "MODEL-1": test_translated_model_1[['text', 'labels']],
    "MODEL-2": test_translated_model_2[['text', 'labels']],
    "MODEL-3": test_translated_model_3[['text', 'labels']],
    "ORIGINAL": test_original[['text', 'labels']],
}

test_instances = []

for name, file in test_files_dict.items():
    test_instance = TestInstance(file, args, name)
    test_instances.append(test_instance)

# Train the model
print("Started Training")

if args["evaluate_during_training"]:
    for i in range(args["n_fold"]):
        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])
        print("Started Fold {}".format(i))
        train_df, eval_df = train_test_split(train, test_size=0.2, random_state=SEED*i)
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                    use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
                                    use_cuda=torch.cuda.is_available())
        
        for test_instance in test_instances:
            predictions, raw_outputs = model.predict(test_instance.get_sentences())
            test_instance.test_preds[:, i] = predictions

        model = None
        print("Completed Fold {}".format(i))

    for test_instance in test_instances:
        final_predictions = []
        for row in test_instance.test_preds:
            row = row.tolist()
            final_predictions.append(int(max(set(row), key=row.count)))
        test_instance.df['predictions'] = final_predictions
else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                    use_cuda=torch.cuda.is_available())
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    for test_instance in test_instances:
        predictions, raw_outputs = model.predict(test_instance.get_sentences())
        test_instance.df['predictions'] = predictions

for test_instance in test_instances:
    print()
    print("==================== Results for " + test_instance.name + "========================")
    test_instance.df['predictions'] = decode(test_instance.df['predictions'])
    test_instance.df['labels'] = decode(test_instance.df['labels'])
    test_instance.df.to_csv(RESULT_FOLDER+test_instance.name+'.csv', sep='\t',index=False)
    print_information(test_instance.df, "predictions", "labels")
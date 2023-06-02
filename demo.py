# from ts.components.data_ingestion import DataIngestion
# from ts.components.model_trainer import ModelTrainer
# # from ts.components.model_evaluation import ModelEvaluation
# # from ts.components.model_pusher import ModelPusher

# from ts.components.data_loader import NewsDataLoader
# from ts.components.model_finetuner import T5smallFinetuner
# # from ts.entity.artifact_entity import ModelTrainerArtifacts
# # from ts.constants import *

# from ts.entity.config_entity import DataIngestionConfig, ModelTrainerConfig
# # , ModelEvaluationConfig
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# data_ingestion = DataIngestion(DataIngestionConfig())

# data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()

# model_name = PRETRAINED_MODEL_NAME

# t5tokenizer = AutoTokenizer.from_pretrained(model_name)
# t5small_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# train_data_path = data_ingestion_artifacts.train_file_path
# val_data_path = data_ingestion_artifacts.test_file_path
# dataloader = NewsDataLoader(train_file_path= train_data_path,
#                             val_file_path= val_data_path,
#                             tokenizer=t5tokenizer,
#                             batch_size=BATCH_SIZE,
#                             columns_name=COLUMNS_NAME
#                             )
# dataloader.prepare_data()
# dataloader.setup()

# model = T5smallFinetuner(model=t5small_model, tokenizer=t5tokenizer)

# model_trainer = ModelTrainer(ModelTrainerConfig(), model, dataloader)

# model_trainer_artifacts = model_trainer.initiate_model_trainer()

# trained_model_path=r'D:\\Projects\\text-summarization\\artifacts\\12_19_2022_20_01_08\\model_training_artifacts\\trained_model\\model.pt'
# model_loss=3.397435188293457

# model_trainer_artifact = ModelTrainerArtifacts(trained_model_path=trained_model_path, model_loss=model_loss)

# model_evaluation = ModelEvaluation(ModelEvaluationConfig(),model_trainer_artifacts=model_trainer_artifact)

# model_evaluation_artifacts = model_evaluation.initiate_evaluation()

# model_pusher = ModelPusher(model_evaluation_artifacts)

# model_pusher_artifacts = model_pusher.initiate_model_pusher()

from ts.pipeline.training_pipeline import TrainingPipeline

tp = TrainingPipeline()

tp.run_pipeline()

from ts.pipeline.prediction_pipeline import SinglePrediction

pred = SinglePrediction()

text ="""Actor Sushant Singh Rajput has said that he doesn't mind men flirting with him and takes it as a compliment. However, he added that he doesn't usually expect attention from men. Meanwhile, it has been rumoured that Sushant has been dating actress Kriti Sanon since they started filming for their upcoming film 'Raabta'. """

result = pred.predict(text)
print(result)
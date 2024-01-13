Vijay Kumaravelrajan vkumarav@usc.edu

In one or two sentences, present your project to us!
Using transfer learning, I train BERTforSequenceClassification on the News Headlines for Sarcasm Detection Dataset to classify sarcastic and non-sarcastic news headlines from The Onion and Huffington Post, respectively.

Dataset: Indicate the dataset you chose to use, any preprocessing steps that were applied, as well as the reasoning behind these choices:
The dataset used was News Headlines for Sacrasm Detection. The dataset was split into training, validation and test sets, with 80% being used for training, 10% for validation and 10% for evaluating model performance. Before fine-tuning, each headline had to be tokenized using the Bert-Base-Uncased tokenizer and augmented with classification, ending and padding tokens.

Model Development and Training: Discuss your model implementation choices, the training procedure, the conditions you settled on (e.g., hyperparameters), and discuss why these are a good set for your task:
The model was implemented using the pre-trained Bert-base-uncased model on a V100 GPU made available through Google Colab. The batch size was 32, as that was the highest able to fit into the GPU memory available. The model was trained for three epochs, as recommended for text classification tasks through transfer learning. The optimizer used was ADAM with a learning rate of 5e-5. These values were chosen with reference author recommendations from the BERT paper.

Model Evaluation/Results: Present the metrics you chose and your model evaluation results. 
The metrics chosen to evaluate model results were Accuracy, Precision, Recall, F1 Score, and Matthew's correlation coefficient in order to provide the most complete picture of the model's classification capabilites.
Accuracy: 94.2%
Precision: 96.1%
Recall: 91.2%
F1 Score: 93.6%
MCC: 0.884


How well does your dataset, model architecture, training procedures, and chosen metrics fit the task at hand? 
The model architecture and training procedures fit well with the task at hand. Using a pre-trained model like BERT allows a much smaller amount of compute to be used in order to train this otherwise complex classification task, as the final layer of the model can be used to generate a feature-rich embedding of the input text to train a feed-forward classification network. The training procedure used was based of the recommendations of the BERT paper, and the metrics chosen appropriately characterize the model's classification performance.

Can your efforts be extended to wider implications or contribute to social good? Are there any limitations in your methods that should be considered before doing so?
Detecting sarcasm in writing is a limited use-case, however, more broadly emotion and tone recognition in written text would be useful, for instance, detecting patient mood in the context of autonomous therapy for mental illness. Before tackling that task, it would be necessary to build a diverse training dataset with high-quality examples to train the model. This dataset could be sourced either from real-world data or synthetically generated data.

If you were to continue this project, what would be your next steps?
My next step for this project would be to attempt to construct a more diverse dataset to classify a wider array of tones and emotions by using an LLM such as Gemini or GPT-4 to generate high-quality synthetic data. After this, I would train another sequence classification network on top of BERT. As BERT is a language model with a relatively small amount of parameters at 110 million, I think an exciting extension and use-case for this project could be a version of this classifier running locally on user machines and classifying the tone of the writing they produce to see if it matches their intention. 

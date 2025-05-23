# AI_Detection
Text Classification: Human vs AI

The challenge proposed by Tradrly in the field of AI/ML:

Create a classification model capable of analyzing a text and predicting whether it was written by a human or generated by artificial intelligence.

Our proposal:

A deep learning model capable of classifying text responses as either human-written or ChatGPT-written.
We use the HC3 (Human ChatGPT Comparison Corpus) dataset to power our model.

Technologies used: Python, TensorFlow / Keras, scikit-learn, NLTK, Matplotlib, Seaborn, Pandas, NumPy

Dataset used:  Hello-SimpleAI/HC3 (split: train) takenn from the plateform HuggingFace which mainly presents
answers to questions written by humans and others generated by ChatGPT from several domains.

Preprocessing: We made some modifications to this dataset to adapt it to our needs: creation of a new dataset form
some processing using NTLK

Dividing the dataset into training and test sets

Text vectorization with TextVectorization (Keras):

Maximum of 20,000 tokens

Model architecture: Keras sequential model

Training:

Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation: Predictions on the test set

Computation of the confusion matrix, classification ratio, and overall accuracy.

Early Stopping on the val_accuracy with a patience of 2 epochs.

Visualizations possible with Matplotlib and Seaborn.

The trained model is saved in a file: model.keras

This model was then used for a simple simulation using a web interface called 'InterfaceDetect.py'. The model receives its input from the text bar. Once the button is clicked, the detection result is immediately displayed. This interface was created using the open-source 'Streamlit' library.

Expected results:

Adequate accuracy (to be adjusted based on observed results).

Good ability to differentiate human text from text generated by ChatGPT.

import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer

### TASK 1 ###
BASE_MODEL_NAME = 'bert-base-uncased'

class SentenceTransformer(nn.Module):
    def __init__(self):
        super(SentenceTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(BASE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.output_size = self.transformer.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        # Max pooling
        expandedMask = attention_mask.unsqueeze(-1).expand(embeddings.size())
        embeddings[expandedMask == 0] = -1e9
        maxPool = torch.max(embeddings, dim=1).values
        return maxPool


# TESTING SHOWCASE
testModel = SentenceTransformer()

examples = ['I hate dogs!', 'I love cats!', 'I do not mind humans.']
encoded = testModel.tokenizer(examples, padding=True, truncation=True, return_tensors='pt')

embeddings = testModel(encoded['input_ids'], encoded['attention_mask'])


print('### Sample outputs for Task 1 ###')
for i, sentence in enumerate(examples):
    print(f'Sentence: {sentence}')
    print(f'Size: {embeddings[i].shape}')
    print(f'Embedding (Sliced): {embeddings[i][:10]}\n') # Sliced to first 10 for brevity, shape shown for concept understanding





### TASK 2/4 ###

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_classification, num_classes_sentiment):
        """
        Multi-Task Learning model for Sentence Classification and Sentiment Analysis (Task 2)

        Args:
            num_classes_classification: Number of classes for sentence classification task
            num_classes_sentiment: Number of classes for sentiment analysis
        """
        super(MultiTaskModel, self).__init__()
        self.sentence_transformer = SentenceTransformer()
        
        # Classification head
        self.classification_head = nn.Linear(self.sentence_transformer.output_size, num_classes_classification)
        
        # Sentiment analysis head
        self.sentiment_head = nn.Linear(self.sentence_transformer.output_size, num_classes_sentiment)

    def forward(self, input_sentences,  attention_mask):
        embeddings = self.sentence_transformer(input_sentences, attention_mask)
        
        # Output for classification task
        classificationLogits = self.classification_head(embeddings)
        
        # Output for sentiment task
        sentimentLogits = self.sentiment_head(embeddings)
        
        return classificationLogits, sentimentLogits
    


def mulitTaskTraining(model, training_data, epochs = 5, lr = 2e-5):
    """
    Training loop for multi-task model (Task 4)

    Args:
        model: Model to be trained
        training_data: Tokenized data in batches including true class labels
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterionClassification = nn.CrossEntropyLoss()
    criterionSentiment = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        totalLoss = 0

        # Training loop
        for batch in training_data:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            optimizer.zero_grad()

            # Forward pass
            classificationLogits, sentimentLogits = model(input_ids, attention_mask)

            lossClassification = criterionClassification(classificationLogits, batch["true_classes"])
            lossSentiment = criterionSentiment(sentimentLogits, batch["true_sentiments"])
            loss = lossClassification + lossSentiment

            # Backward pass
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

        totalLoss = totalLoss / len(training_data) # total loss / # of batches
        print(f"Training Loss: {totalLoss}")


# TESTING SHOWCASE

"""
For the purposes of showing that the model can train, I created a very small fake dataset.

For the purposes of how it could actually be trained, a real dataset could be loaded with the pytorch dataloader
"""

classLabels = {
    0: 'Does not include a commmon type of pet',
    1: 'Does include a common type of pet'
}
sentimentLabels = {
    0: 'Positive',
    1: 'Negative',
    2: 'Indifferent'
}

examples = ['I love dogs!', 'I hate cats!', 'I do not mind humans.']
trueClasses = torch.tensor([1, 1, 0])
trueSentiments = torch.tensor([0, 1, 2])

multiTask = MultiTaskModel(num_classes_classification = 2, num_classes_sentiment = 3)

encoded = multiTask.sentence_transformer.tokenizer(examples, padding=True, truncation=True, return_tensors='pt')
encoded['true_classes'] = trueClasses
encoded['true_sentiments'] = trueSentiments
encoded = [encoded] # Wrapped in a list to simulate "batches" in training. Since this simulated dataset is so small, this represents a single batch

print("### Sample outputs for combined tasks 2 & 4 ###")
mulitTaskTraining(multiTask, encoded)
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
#
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# Define a new example sentence with multiple meanings of the word "bank"

def get_bert_embedding(text, method, test_same_word):
    '''
    Text: The input text
    Method: For different downstream methods, we can produce different word embeddings.
    While concatenation of the last four layers produced the best results on this specific task,
    many of the other methods come in a close second and in general it is advisable to test different
    versions for your specific application: results may vary.
    test_same_word: To test whether the same words with different meanings would have similar embeddings.
    '''
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Display the words with their indices.
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    print(segments_ids)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-chinese')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    print("Number of layers:", len(encoded_layers))
    layer_i = 0

    print("Number of batches:", len(encoded_layers[layer_i]))
    batch_i = 0

    print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    token_i = 0

    print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))
    # `encoded_layers` is a Python list.
    print('Type of encoded_layers: ', type(encoded_layers))

    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', encoded_layers[0].size())

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings.size()
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings.size()
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)
    token_embeddings.size()
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    if method == 'concat':
        # `token_embeddings` is a [22 x 12 x 768] tensor.
        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Concatenate the vectors (that is, append them together) from the last
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)
        print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
        # print("The embedding vector is", token_vecs_cat)
        return token_vecs_cat
    if method == 'mean':
        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
    if test_same_word:
        ######test for word embedding
        print('First 15 vector values for one instance appearing in different places')

        print("Mean_1:", str(token_vecs_sum[6][:15]))
        print("Mean_2:", str(token_vecs_sum[11][:15]))

        # Calculate the cosine similarity between the same word with different meanings
        diff_mean = 1 - cosine(token_vecs_sum[6], token_vecs_sum[11])
        print('Vector similarity for  *similar*  word:  %.2f' % diff_mean)

        return token_vecs_sum


if __name__ == "__main__":
    text = "你到底什么意思 " \
           "我就是意思意思"
    # execute only if run as a script
    text_bert_embedding = get_bert_embedding(text, method='mean', test_same_word=True)

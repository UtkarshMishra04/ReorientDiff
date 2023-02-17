import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers import CLIPConfig, CLIPProcessor, CLIPTokenizer, CLIPModel

class HuggingFaceEncoding(nn.Module):

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.embedding_dim = 1024

    def single_forward(self, image, text):
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**image_inputs)
        text_inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(**text_inputs)
        return torch.cat([image_features, text_features], dim=1)

    def get_image_features(self, image):
        image_batch = [image[i] for i in range(image.shape[0])]
        image_inputs = self.processor(images=image_batch, return_tensors="pt")
        image_features = self.model.get_image_features(**image_inputs)
        return image_features

    def get_text_features(self, text):
        text_inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(**text_inputs)
        return text_features

    def forward(self, image, text):
        # batched input
        # image: [batch_size, 3, 240, 320]
        # text: [batch_size, 1]
        device = image.device

        # convert image batch to a list of image tensors
        image_batch = image #[image[i] for i in range(image.shape[0])]
        all_embeddings = self.single_forward(image_batch, text)

        return all_embeddings.to(device)


def main():
    model = HuggingFaceEncoding()
    batch_size = 5
    image = torch.rand(3, 240, 320).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    text = ["a dog", "a cat", "a bat", "a ball", "a mug"]
    embedding = model(image, text).detach().numpy()
    print(embedding.shape)

    # plot cosine similarity between embeddings

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity

    # compute cosine similarity
    cos_sim = cosine_similarity(embedding, embedding)
    # plot
    sns.heatmap(cos_sim, annot=True, xticklabels=text, yticklabels=text)
    plt.show()



if __name__ == '__main__':
    main()
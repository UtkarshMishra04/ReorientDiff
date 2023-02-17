import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F


class SegmentationModel(nn.Module):

    def __init__(self, image_shape, object_dim, output_dim):
        super().__init__()
        # Load the ResNet50 model for RGBD images
        self.model = models.segmentation.fcn_resnet50()
        self.model.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last layer with a new one
        self.model.classifier[4] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))

        # get an embedscp ding vector with downsample from (b, 10, 240, 320) to (b, 512)
        self.downsample_net11 = nn.Sequential(
            nn.Conv2d(10, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )

        self.downsample_net12 = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.downsample_net2 = nn.Sequential(
            nn.Linear(60*80 + output_dim, output_dim),
            nn.ReLU()
        )

        self.upsample_net1 = nn.Sequential(
            nn.Linear(output_dim, 60*80),
            nn.ReLU()
        )

        self.upsample_net21 = nn.Sequential(
            nn.ConvTranspose2d(2, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )

        self.upsample_net22 = nn.Sequential(
            nn.ConvTranspose2d(10, 10, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        self.output_net = nn.Sequential(
            nn.Conv2d(10, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, rgbd, obj):
        # Run the model
        lats = []
        x = self.model(rgbd)['out']
        x = self.downsample_net11(x)
        lats.append(x)
        x = self.downsample_net12(x)
        lats.append(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, obj), dim=1)
        embed = self.downsample_net2(x)
        x = self.upsample_net1(embed)
        x = x.view(x.shape[0], 1, 60, 80)
        x = self.upsample_net21(torch.cat((x, lats[1]), dim=1))
        x = self.upsample_net22(torch.cat((x, lats[0]), dim=1))
        x = self.output_net(x)
        return x, embed

    def get_embedding(self, rgbd, obj):
        x = self.model(rgbd)['out']
        x = self.downsample_net11(x)
        x = self.downsample_net12(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, obj), dim=1)
        x = self.downsample_net2(x)
        return x

    

class PoseObjPredictor(nn.Module):
    def __init__(self, image_shape, pose_dim, object_dim, output_dim, pose_hdims, object_hdims):
        super().__init__()
        self.image_shape = image_shape
        self.pose_dim = 3
        self.orientation_dim = 2
        self.object_dim = object_dim
        self.output_dim = output_dim
        self.pose_hdims = pose_hdims
        self.object_hdims = object_hdims
        self.segmentation_model = SegmentationModel(image_shape, object_dim, output_dim)

        self.pose_predictor = nn.Sequential(
            nn.Linear(output_dim + output_dim + output_dim, pose_hdims[0]),
            nn.ReLU(),
            nn.Linear(pose_hdims[0], pose_hdims[1]),
            nn.ReLU(),
            nn.Linear(pose_hdims[1], pose_hdims[2]),
            nn.ReLU(),
            nn.Linear(pose_hdims[2], self.pose_dim)
        )

        self.orientation_predictor = nn.Sequential(
            nn.Linear(output_dim + output_dim + output_dim, pose_hdims[0]),
            nn.ReLU(),
            nn.Linear(pose_hdims[0], pose_hdims[1]),
            nn.ReLU(),
            nn.Linear(pose_hdims[1], pose_hdims[2]),
            nn.ReLU(),
            nn.Linear(pose_hdims[2], self.orientation_dim)
        )

        self.object_predictor = nn.Sequential(
            nn.Linear(output_dim + output_dim + output_dim, object_hdims[0]),
            nn.ReLU(),
            nn.Linear(object_hdims[0], object_hdims[1]),
            nn.ReLU(),
            nn.Linear(object_hdims[1], self.object_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, clip_image, clip_text, rgbd):
        # Get the segmentation mask
        # clip_image = self.clip_model.get_image_features(rgb).to(rgbd.device)
        # clip_text = self.clip_model.get_text_features(text).to(rgbd.device)

        # print(clip_image.shape)
        # print(clip_text.shape)
        # print(rgbd.shape)

        segmentation_mask, image_embeddding = self.segmentation_model(rgbd, clip_text)
        # image_embeddding = self.segmentation_model.get_embedding(rgbd, clip_text)

        joint_embedding = torch.cat((clip_image, clip_text, image_embeddding), dim=1)

        # Predict the pose
        pose = self.pose_predictor(joint_embedding)
        orientation = self.orientation_predictor(joint_embedding)
        object_pred = self.object_predictor(joint_embedding)

        return pose, orientation, object_pred, segmentation_mask

    def get_embedding(self, clip_image, clip_text, rgbd):

        with torch.no_grad():
            segmentation_mask, image_embeddding = self.segmentation_model(rgbd, clip_text)
            # image_embeddding = self.segmentation_model.get_embedding(rgbd, clip_text)

            joint_embedding = torch.cat((clip_image, clip_text, image_embeddding), dim=1)

        return joint_embedding

    def get_loss(self, clip_image, clip_text, rgbd, pose, object, segm):

        pred_pose, pred_orientation, pred_object, pred_mask = self.forward(clip_image, clip_text, rgbd)

        # Calculate the loss
        pose_loss = F.mse_loss(pred_pose, pose[:, :3])
        orientation_loss = F.mse_loss(pred_orientation, pose[:, :-2])
        object_loss = F.mse_loss(pred_object, object)
        mask_loss = F.mse_loss(pred_mask, segm)

        loss = pose_loss + orientation_loss + object_loss + mask_loss

        return loss, pose_loss + orientation_loss, object_loss, mask_loss

    def get_predictions(self, clip_image, clip_text, rgbd):
        with torch.no_grad():
            pred_pose, pred_orientation, pred_object, pred_mask = self.forward(clip_image, clip_text, rgbd)
            pred_pose = torch.cat((pred_pose, torch.zeros_like(pred_orientation), pred_orientation), dim=1)
        return pred_pose, pred_object, pred_mask
        

def main():
    # Load the model
    model = PoseObjPredictor(None, (4, 240, 320), 7, 6, 512, None, None)
    model = model.to('cuda')
    model.eval()

    # Load the image
    image = torch.randn(10, 4, 240, 320).to('cuda')
    object = torch.randn(10, 6).to('cuda')
    print(image.shape)

    # Run the model
    mask = model(image, object)
    print(mask.shape)

if __name__ == '__main__':
    main()



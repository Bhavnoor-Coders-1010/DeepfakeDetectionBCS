import streamlit as st
import warnings
import logging
import absl.logging
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image


class DeepfakeNet(nn.Module):
    def __init__(self):
        super(DeepfakeNet, self).__init__()
#         self.drp = nn.Dropout(0.4)#------------------------------------------------------------------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList([Bottleneck(64, 256, first=True)])
        self.out_size = [256,512,1024,2048]
        self.blocks = [1,2,2,1]
        for i in range(len(self.out_size)):
            if i > 0:
                self.layers.append(Bottleneck(self.out_size[i-1], self.out_size[i], 2))
            for extraLayers in range(self.blocks[i]-1):
                self.layers.append(Bottleneck(self.out_size[i], self.out_size[i]))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1)



    # def _make_layer(self, in_channels, out_channels, blocks):
    #     layers = []
    #     for _ in range(blocks):
    #         layers.append(Bottleneck(in_channels, out_channels))
    #     return nn.Sequential(*layers)

    def forward(self, x):
#         x = self.drp(x)#------------------------------------------------------------------------------------------------------------------------------------
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, first=False):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels//2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1, groups=32, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = stride == 2 or first
        if self.downsample:
            self.changeInputC2D = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            self.changeInputBn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.changeInputC2D(residual)
            residual = self.changeInputBn(residual)
        out = torch.add(out, residual)
        out = self.relu(out)
        return out

# Example usage
device = torch.device("cpu")
newModel = torch.load("DeepfakeBCSLatest.pth", weights_only=False, map_location=torch.device('cpu'))
newModel.to(device)


from PIL import Image



logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")



def predict_image(image_path):
    newModel.eval()
    path = image_path
    img = Image.open(path)
    # plt.imshow(img)
    basic = v2.Compose([v2.Resize((224,224)),v2.ToTensor()])
    img = basic(img)

    img = img.to(device)
    img = img.view((1,3,224,224))
    if torch.round(newModel(img)).item()==1:
        return "FAKE", newModel(img)[0][0].item()
    else:
        return "REAL", 1-newModel(img)[0][0].item()


st.set_page_config(page_title="DeepFake Detector", page_icon="💪", layout="centered")

st.title('🧐 DeepFake Detector 🕵️‍♀️')
st.write("Upload an image...")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.subheader("Image Preview")
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    st.write("")
    st.markdown("<h3 style='color:white;'>Classifying...</h3>", unsafe_allow_html=True)

    predicted_label, prediction_confidence = predict_image("uploaded_image.jpg")
    st.subheader("Prediction")
    st.write(
        f"Confidence Score: **{prediction_confidence * 100:.1f}%**\nPredicted Label: {predicted_label}")

# Additional CSS for black background
st.markdown("""
<style>
    .stApp {
        background-color: black;
        color: white;
    }
    .st-bf {
        font-size: 1.5rem;
    }
    .st-df {
        color: white;
    }
</style>
""", unsafe_allow_html=True)